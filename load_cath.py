from typing import List, Optional
import torch
from torch.utils.data import DataLoader, Dataset

from cath import CATH
from data_utils import Alphabet, MaxTokensBatchSampler

from pytorch_lightning import LightningDataModule

# @register_datamodule('cath')
class CATHDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        chain_set_jsonl: str = 'chain_set.jsonl',
        chain_set_splits_json: str = 'chain_set_splits.json',
        max_length: int = 500,
        atoms: List[str] = ('N', 'CA', 'C', 'O'),
        alphabet_name: str = 'esm2',
        batch_size: int = 64,
        max_tokens: int = 6000,
        sort: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_split: str = 'train',
        valid_split: str = 'valid',
        test_split: str = 'test',
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.alphabet = None # alphabet class

        self.train_data: Optional[Dataset] = None
        self.valid_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if stage == 'fit':
            (train, valid), alphabet = CATH(
                self.hparams.data_dir,
                chain_set_jsonl=self.hparams.chain_set_jsonl,
                chain_set_splits_json=self.hparams.chain_set_splits_json,
                max_length=self.hparams.max_length,
                split=(self.hparams.train_split, self.hparams.valid_split),
            )
            self.train_dataset = train
            self.valid_dataset = valid
        elif stage == 'test' or stage == 'predict':
            test, alphabet = CATH(
                self.hparams.data_dir,
                chain_set_jsonl=self.hparams.chain_set_jsonl,
                chain_set_splits_json=self.hparams.chain_set_splits_json,
                split=(self.hparams.test_split, )
            )
            self.test_dataset = test
        else:
            raise ValueError(f"Invalid stage: {stage}.")

        self.alphabet = Alphabet(name=self.hparams.alphabet_name, featurizer='cath')

        self.collate_batch = self.alphabet.featurizer

    def _build_batch_sampler(self, dataset, max_tokens, shuffle=False, distributed=True):
        is_distributed = distributed and torch.distributed.is_initialized()

        batch_sampler = MaxTokensBatchSampler(
            dataset=dataset,
            shuffle=shuffle,
            distributed=is_distributed,
            batch_size=self.hparams.batch_size,
            max_tokens=max_tokens,
            sort=self.hparams.sort,
            drop_last=False,
            sort_key=lambda i: len(dataset[i]['seq']))
        return batch_sampler

    def train_dataloader(self):
        if not hasattr(self, 'train_batch_sampler'):
            self.train_batch_sampler = self._build_batch_sampler(
                self.train_dataset,
                max_tokens=self.hparams.max_tokens,
                shuffle=True
            )
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_sampler=self._build_batch_sampler(
                self.valid_dataset, max_tokens=self.hparams.max_tokens, distributed=False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_sampler=self._build_batch_sampler(
                self.test_dataset, max_tokens=self.hparams.max_tokens, distributed=False),
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
        )

def main():
    """
    Main function to demonstrate how to initialize the CATH data module
    and create a dataloader for training.
    """
    # Initialize the CATH data module
    cath_dm = CATHDataModule(
        data_dir="data/cath_4.3/",
        chain_set_jsonl="chain_set.jsonl",
        chain_set_splits_json="chain_set_splits.json",
        max_length=500,
        atoms=('N', 'CA', 'C', 'O'),
        alphabet_name='esm2',  # Use ESM-2 vocabulary
        batch_size=32,
        max_tokens=6000,
        sort=True,
        num_workers=4,
        pin_memory=True,
        train_split='train',
        valid_split='valid',
        test_split='test'
    )
    
    # Setup the data module for training
    cath_dm.setup(stage='fit')
    
    # Get the training dataloader
    train_dataloader = cath_dm.train_dataloader()
    val_dataloader = cath_dm.val_dataloader()
    
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    
    # Example: iterate through a few batches
    print("\nExample batch iteration:")
    for i, batch in enumerate(train_dataloader):
        if i >= 3:  # Only show first 3 batches
            break

        B, L, _, _ = batch['coords'].shape

        print(f"Batch {i+1}:")
        print(f"  Batch size (B): {B}")
        print(f"  Sequence lengths (L): {[len(seq) for seq in batch['seqs']]}")
        if 'coords' in batch:
            print(f"  Coordinates shape: {batch['coords'].shape}")
        print()

if __name__ == "__main__":
    main()



"""
Loaded data size: 22401/22507. Discarded: {'bad_chars': 107, 'too_long': 0}.
Size. train: 16631, validation: 1516
Number of training batches: 649
Number of validation batches: 50

Example batch iteration:
Batch 1:
  Batch size (B): 48
  Sequence lengths (L): [124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125, 125]
  Coordinates shape: torch.Size([48, 125, 4, 3])

Batch 2:
  Batch size (B): 57
  Sequence lengths (L): [105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106]
  Coordinates shape: torch.Size([57, 106, 4, 3])

Batch 3:
  Batch size (B): 25
  Sequence lengths (L): [231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231, 231]
  Coordinates shape: torch.Size([25, 231, 4, 3])

"""