
from typing import Any, List, Union
import numpy as np
import torch
from torch import nn


from src.modules import metrics

from omegaconf import DictConfig
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric

from src import utils
from src.tasks import TaskLitModule, register_task
from src.datamodules.data_utils import Alphabet
from src.utils.config import compose_config as Cfg

DEBUG=False
log = utils.get_logger(__name__)

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


@register_task('cmlm')
class CMLM(TaskLitModule):

    _DEFAULT_CFG: DictConfig = Cfg(
        learning=Cfg(
            noise='no_noise',  # ['full_mask', 'random_mask']
            num_unroll=0,
        ),
        generator=Cfg(
            max_iter=1,
            strategy='denoise',  # ['denoise' | 'mask_predict']
            noise='full_mask',  # ['full_mask' | 'selected mask']
            replace_visible_tokens=False,
            temperature=0,
            eval_sc=False,
        )
    )

    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        alphabet: DictConfig,
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
        *,
        learning=_DEFAULT_CFG.learning,
        generator=_DEFAULT_CFG.generator
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(ignore=['model', 'criterion'], logger=False)
        self.save_hyperparameters(logger=True)

        self.alphabet = Alphabet(**alphabet)
        self.build_model() 

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == 'fit':
            log.info(f'\n{self.model}')

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(cfg=self.hparams.model, group='model')

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(cfg=self.hparams.criterion) 
        self.criterion.ignore_index = self.alphabet.padding_idx

    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_nll_loss = MeanMetric()

        self.val_ppl_best = MinMetric()

        self.acc = MeanMetric()
        self.acc_best = MaxMetric()

        self.acc_median = CatMetric()
        self.acc_median_best = MaxMetric()

    def load_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_epoch_start(self) -> None:
        pass

    # -------# Training #-------- #
    @torch.no_grad()
    def inject_noise(self, tokens, coord_mask, noise='random_mask', sel_mask=None, mask_by_unk=False):
        padding_idx = self.alphabet.padding_idx
        if mask_by_unk:
            mask_idx = self.alphabet.unk_idx
        else:
            mask_idx = self.alphabet.mask_idx

        def _full_mask(target_tokens):
            target_mask = (
                target_tokens.ne(padding_idx)  # & mask
                & target_tokens.ne(self.alphabet.cls_idx)
                & target_tokens.ne(self.alphabet.eos_idx)
            )
            # masked_target_tokens = target_tokens.masked_fill(~target_mask, mask_idx)
            masked_target_tokens = target_tokens.masked_fill(target_mask, mask_idx)
            return masked_target_tokens

        def _random_mask(target_tokens):
            target_masks = (
                target_tokens.ne(padding_idx) & coord_mask
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            masked_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), mask_idx
            )
            return masked_target_tokens 

        def _selected_mask(target_tokens, sel_mask):
            # for inpainting custom selected positions
            masked_target_tokens = torch.masked_fill(target_tokens, mask=sel_mask, value=mask_idx)
            return masked_target_tokens

        def _adaptive_mask(target_tokens):
            raise NotImplementedError

        noise = noise or self.hparams.noise

        if noise == 'full_mask':
            masked_tokens = _full_mask(tokens)
        elif noise == 'random_mask':
            masked_tokens = _random_mask(tokens)
        elif noise == 'selected_mask':
            masked_tokens = _selected_mask(tokens, sel_mask=sel_mask)
        elif noise == 'no_noise':
            masked_tokens = tokens
        else:
            raise ValueError(f"Noise type ({noise}) not defined.")

        prev_tokens = masked_tokens
        prev_token_mask = prev_tokens.eq(mask_idx) & coord_mask
        # target_mask = prev_token_mask & coord_mask

        return prev_tokens, prev_token_mask  # , target_mask

    def step(self, batch):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids     
        """
        coords = batch['coords']
        coord_mask = batch['coord_mask']
        tokens = batch['tokens']

        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, coord_mask, noise=self.hparams.learning.noise)
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = label_mask = prev_token_mask

        logits = self.model(batch)
        # logits = self.model.forward_encoder(batch)

        if isinstance(logits, tuple):
            logits, encoder_logits = logits
            # loss, logging_output = self.criterion(logits, tokens, label_mask=label_mask)
            # NOTE: use fullseq loss for pLM prediction
            loss, logging_output = self.criterion(
                logits, tokens,
                # hack to calculate ppl over coord_mask in test as same other methods
                label_mask=label_mask if self.stage == 'test' else None
            )
            encoder_loss, encoder_logging_output = self.criterion(encoder_logits, tokens, label_mask=label_mask)

            loss = loss + encoder_loss
            logging_output['encoder/nll_loss'] = encoder_logging_output['nll_loss']
            logging_output['encoder/ppl'] = encoder_logging_output['ppl']
        else:
            loss, logging_output = self.criterion(logits, tokens, label_mask=label_mask)

        return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)

        # log train metrics
        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log('lr', self.lrate, on_step=True, on_epoch=False, prog_bar=True)

        for log_key in logging_output:
            log_value = logging_output[log_key]
            self.log(f"train/{log_key}", log_value, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    # -------# Evaluating #-------- #
    def on_test_epoch_start(self) -> None:
        self.hparams.noise = 'full_mask'

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)

        # log other metrics
        sample_size = logging_output['sample_size']
        self.eval_loss.update(loss, weight=sample_size)
        self.eval_nll_loss.update(logging_output['nll_loss'], weight=sample_size)

        if self.stage == 'fit':
            pred_outs = self.predict_step(batch, batch_idx)
        return {"loss": loss}

    def on_validation_epoch_end(self):
        log_key = 'test' if self.stage == 'test' else 'val'

        # compute metrics averaged over the whole dataset
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()

        eval_nll_loss = self.eval_nll_loss.compute()
        self.eval_nll_loss.reset()
        
        eval_ppl = torch.exp(eval_nll_loss)
        # 2. log them
        self.log(f"{log_key}/loss",     eval_loss,     prog_bar=True)
        self.log(f"{log_key}/nll_loss", eval_nll_loss, prog_bar=True)
        self.log(f"{log_key}/ppl",      eval_ppl,      prog_bar=True)

        if self.stage == 'fit':
            self.val_ppl_best.update(eval_ppl)
            self.log("val/ppl_best", self.val_ppl_best.compute(), prog_bar=True)

            self.on_predict_epoch_end()

        super().on_validation_epoch_end()

    # -------# Inference/Prediction #-------- #
    def forward(self, batch, return_ids=False):
        # In testing, remove target tokens to ensure no data leakage!
        # or you can just use the following one if you really know what you are doing:
        #   tokens = batch['tokens']
        tokens = batch.pop('tokens')

        prev_tokens, prev_token_mask = self.inject_noise(
            tokens, batch['coord_mask'],
            noise=self.hparams.generator.noise,  # NOTE: 'random_mask' by default. Set to 'selected_mask' when doing inpainting.
        )
        batch['prev_tokens'] = prev_tokens
        batch['prev_token_mask'] = prev_tokens.eq(self.alphabet.mask_idx)

        # Simple argmax prediction instead of iterative generation
        with torch.no_grad():
            logits = self.model(batch)
            if isinstance(logits, tuple):
                logits = logits[0]  # Use main logits, not encoder logits
            
            # Get predictions for masked positions only
            mask_positions = prev_tokens.eq(self.alphabet.mask_idx)
            predicted_tokens = torch.argmax(logits, dim=-1)
            
            # Replace masked positions with predictions, keep original tokens elsewhere
            output_tokens = prev_tokens.clone()
            output_tokens[mask_positions] = predicted_tokens[mask_positions]
        if DEBUG:
            # Log token information for debugging
            print(f"Original tokens: {tokens[0][:10]}")  # First 10 tokens of first sequence
            print(f"Masked tokens: {prev_tokens[0][:10]}")  # First 10 tokens of first sequence  
            print(f"Predicted tokens: {predicted_tokens[0][:10]}")  # First 10 tokens of first sequence
            print(f"Output tokens: {output_tokens[0][:10]}")  # First 10 tokens of first sequence

        
        if not return_ids:
            return self.alphabet.decode(output_tokens)
        return output_tokens

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True) -> Any:
        coord_mask = batch['coord_mask']
        tokens = batch['tokens']

        pred_tokens = self.forward(batch, return_ids=True)

        # NOTE: use esm-1b to refine
        # pred_tokens = self.esm_refine(
        #     pred_ids=torch.where(coord_mask, pred_tokens, prev_tokens))
        # # decode(pred_tokens[0:1], self.alphabet)

        if log_metrics:
            # per-sample accuracy
            recovery_acc_per_sample = metrics.accuracy_per_sample(pred_tokens, tokens, mask=coord_mask)
            self.acc_median.update(recovery_acc_per_sample)

            # # global accuracy
            recovery_acc = metrics.accuracy(pred_tokens, tokens, mask=coord_mask)
            self.acc.update(recovery_acc, weight=coord_mask.sum())

        results = {
            'pred_tokens': pred_tokens,
            'names': batch['names'],
            'native': batch['seqs'],
            'recovery': recovery_acc_per_sample,
            'sc_tmscores': np.zeros(pred_tokens.shape[0])
        }

        if self.hparams.generator.eval_sc:
            torch.cuda.empty_cache()
            sc_tmscores = self.eval_self_consistency(pred_tokens, batch['coords'], mask=tokens.ne(self.alphabet.padding_idx))
            results['sc_tmscores'] = sc_tmscores

        return results

    def on_predict_epoch_end(self) -> None:
        log_key = 'test' if self.stage == 'test' else 'val'

        acc = self.acc.compute() * 100
        self.acc.reset()
        self.log(f"{log_key}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        acc_median = torch.median(self.acc_median.compute()) * 100
        self.acc_median.reset()
        self.log(f"{log_key}/acc_median", acc_median, on_step=False, on_epoch=True, prog_bar=True)

        if self.stage == 'fit':
            self.acc_best.update(acc)
            self.log(f"{log_key}/acc_best", self.acc_best.compute(), on_epoch=True, prog_bar=True)

            self.acc_median_best.update(acc_median)
            self.log(f"{log_key}/acc_median_best", self.acc_median_best.compute(), on_epoch=True, prog_bar=True)
        # else:
        #     if self.hparams.generator.eval_sc:
        #         import itertools
        #         sc_tmscores = list(itertools.chain(*[result['sc_tmscores'] for result in results]))
        #         self.log(f"{log_key}/sc_tmscores", np.mean(sc_tmscores), on_epoch=True, prog_bar=True)
        #     self.save_prediction(results, saveto=f'./test_tau{self.hparams.generator.temperature}.fasta')

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def eval_self_consistency(self, pred_ids, positions, mask=None):
        pred_seqs = self.alphabet.decode(pred_ids, remove_special=True)

        # run_folding:
        sc_tmscores = []
        with torch.no_grad():
            output = self._folding_model.infer(sequences=pred_seqs, num_recycles=4)
            pred_seqs = self.alphabet.decode(output['aatype'], remove_special=True)
            for i in range(positions.shape[0]):
                pred_seq = pred_seqs[i]
                seqlen = len(pred_seq)
                _, sc_tmscore = metrics.calc_tm_score(
                    positions[i, 1:seqlen + 1, :3, :].cpu().numpy(),
                    output['positions'][-1, i, :seqlen, :3, :].cpu().numpy(),
                    pred_seq, pred_seq
                )
                sc_tmscores.append(sc_tmscore)
        return sc_tmscores


def convert_by_alphabets(ids, alphabet1, alphabet2, relpace_unk_to_mask=True):
    sizes = ids.size()
    mapped_flat = ids.new_tensor(
        [alphabet2.get_idx(alphabet1.get_tok(ind)) for ind in ids.flatten().tolist()]
    )
    if relpace_unk_to_mask:
        mapped_flat[mapped_flat.eq(alphabet2.unk_idx)] = alphabet2.mask_idx
    return mapped_flat.reshape(*sizes)
