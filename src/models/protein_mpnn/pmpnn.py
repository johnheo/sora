from dataclasses import dataclass

from src.models import register_model
from src.models.protein_mpnn import FixedBackboneDesignEncoderDecoder
from src.datamodules.data_utils import Alphabet

from .encoder import MPNNEncoder


@dataclass
class ProteinMPNNConfig:
    d_model: int = 128
    d_node_feats: int = 128
    d_edge_feats: int = 128
    k_neighbors: int = 48
    augment_eps: float = 0.0
    n_enc_layers: int = 3
    dropout: float = 0.1

    # decoder-only
    n_vocab: int = 22
    n_dec_layers: int = 3
    random_decoding_order: bool = True
    nar: bool = True
    crf: bool = False
    use_esm_alphabet: bool = False


@register_model('protein_mpnn')
class ProteinMPNNCMLM(FixedBackboneDesignEncoderDecoder):
    _default_cfg = ProteinMPNNConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.encoder = MPNNEncoder(
            node_features=self.cfg.d_node_feats,
            edge_features=self.cfg.d_edge_feats,
            hidden_dim=self.cfg.d_model,
            num_encoder_layers=self.cfg.n_enc_layers,
            k_neighbors=self.cfg.k_neighbors,
            augment_eps=self.cfg.augment_eps,
            dropout=self.cfg.dropout
        )
        
        self.decoder = None # only use the encoder

        if self.cfg.use_esm_alphabet:
            alphabet = Alphabet('esm', 'cath')
            self.padding_idx = alphabet.padding_idx
            self.mask_idx = alphabet.mask_idx
        else:
            alphabet = None
            self.padding_idx = 0
            self.mask_idx = 1

    def _forward(self, coords, coord_mask, prev_tokens, token_padding_mask=None, target_tokens=None, return_feats=False, **kwargs):
        coord_mask = coord_mask.float()
        encoder_out = self.encoder(X=coords, mask=coord_mask)

        logits, feats = self.decoder(
            prev_tokens=prev_tokens,
            memory=encoder_out, memory_mask=coord_mask,
            target_tokens=target_tokens,
            **kwargs
        )

        if return_feats:
            return logits, feats
        return logits

    def forward(self, batch, return_feats=False, **kwargs):
        coord_mask = batch['coord_mask'].float()

        encoder_out = self.encoder(
            X=batch['coords'],
            mask=coord_mask,
            residue_idx=batch.get('residue_idx', None),
            chain_idx=batch.get('chain_idx', None)
        )

        # logits, feats = self.decoder(
        #     prev_tokens=batch['prev_tokens'],
        #     memory=encoder_out, 
        #     memory_mask=coord_mask,
        #     target_tokens=batch.get('tokens'),
        #     **kwargs
        # )

        # if return_feats:
        #     return logits, feats
        return encoder_out

    def forward_encoder(self, batch):
        encoder_out = self.encoder(
            X=batch['coords'],
            mask=batch['coord_mask'].float(),
            residue_idx=batch.get('residue_idx', None),
            chain_idx=batch.get('chain_idx', None)
        )
        encoder_out['coord_mask'] = batch['coord_mask'].float()

        return encoder_out