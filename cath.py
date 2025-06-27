import json
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.datapipes.map import SequenceWrapper
from torch.utils.data.dataset import Subset

from data_utils import Alphabet

import esm

def CATH(
    root=".data/cath_4.2",
    chain_set_jsonl='chain_set.jsonl',
    chain_set_splits_json='chain_set_splits.json',
    split=("train", "validation", "test"),
    truncate=None, max_length=500,
    alphabet="ACDEFGHIKLMNPQRSTVWY", # Will use ESM-2 alphabet by default
    verbose=False
):
    alphabet_set = set([a for a in alphabet])

    discard_count = {
        'bad_chars': 0,
        'too_long': 0,
    }

    chain_set_jsonl_fullpath = os.path.join(root, chain_set_jsonl)
    chain_set_splits_json_fullpath = os.path.join(root, chain_set_splits_json)

    # 1) load the dataset
    with open(chain_set_jsonl_fullpath) as f:
        # NOTE: dataset is a list of mapping
        # each mapping has columns:
        #   name: str
        #   seq: str. sequence of amino acids
        #   coords: Dict[str, List[1d-array]]). e.g., {"N": [[0, 0, 0], [0.1, 0.1, 0.1], ..], "Ca": [...], ..}

        dataset: List[Dict] = []

        lines = f.readlines()
        for i, line in enumerate(lines):
            # if i > 300: break
            entry = json.loads(line)
            seq = entry['seq']
            name = entry['name']

            # Convert raw coords to np arrays
            for key, val in entry['coords'].items():
                entry['coords'][key] = np.asarray(val, dtype=np.float32)

            # Check if in alphabet
            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    dataset.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                # print(name, bad_chars, entry['seq'])
                discard_count['bad_chars'] += 1

            if verbose and (i + 1) % 100000 == 0:
                print('{} entries ({} loaded)'.format(len(dataset), i + 1))

            # Truncate early
            if truncate is not None and len(dataset) == truncate:
                break
        total_size = i

        dataset = SequenceWrapper(dataset)
        print(f'Loaded data size: {len(dataset)}/{total_size}. Discarded: {discard_count}.')

        # 2) split the dataset
        dataset_indices = {entry['name']: i for i, entry in enumerate(dataset)}
        with open(chain_set_splits_json_fullpath) as f:
            dataset_splits = json.load(f)

        # compatible with cath data
        split = ['validation' if s == 'valid' else s for s in split]
        dataset_splits = [
            Subset(dataset, [
                dataset_indices[chain_name] for chain_name in dataset_splits[key]
                if chain_name in dataset_indices
            ])
            for key in split
        ]
        sizes = [f'{split[i]}: {len(dataset_splits[i])}' for i in range(len(split))]
        msg_sizes = ', '.join(sizes)
        print(f'Size. {msg_sizes}')  
        if len(dataset_splits) == 1:
            dataset_splits = dataset_splits[0]

        # for split in dataset_splits:

        return dataset_splits, alphabet_set


# NOTE: batch is a list mapping
# each mapping has columns:
#   name: str
#   seq: str. sequence of amino acids
#   coords: Dict[str, List[1d-array]]). e.g., {"N": [[0, 0, 0], [0.1, 0.1, 0.1], ..], "Ca": [...], ..}
def collate_batch(
    batch: List[Dict[str, Any]],
    batch_converter,
    transform=None,
    atoms=('N', 'CA', 'C', 'O')
):
    seqs, coords = [], []
    names = []
    for entry in batch:
        _seq, _coords = entry['seq'], entry['coords']
        seqs.append(_seq)
        # [L, 3] x 4 -> [L, 4, 3]
        coords.append(
            # np.stack([_coords[c] for c in ['N', 'CA', 'C', 'O']], 1)
            np.stack([_coords[c] for c in atoms], 1)
        )
        names.append(entry['name'])

    coords, confidence, strs, tokens, lengths, coord_mask = batch_converter.from_lists(
        coords_list=coords, confidence_list=None, seq_list=seqs
    )

    # coords, tokens, coord_mask, lengths = featurize(batch, torch.device('cpu'), 0)
    # coord_mask = coord_mask > 0.5
    batch_data = {
        'coords': coords,
        'tokens': tokens,
        'confidence': confidence,
        'coord_mask': coord_mask,
        'lengths': lengths,
        'seqs': seqs,
        'names': names
    }

    if transform is not None:
        batch_data = transform(batch_data)

    return batch_data


class CoordBatchConverter(esm.data.BatchConverter):
    def __init__(self, alphabet, coord_pad_inf=False, coord_nan_to_zero=True, to_pifold_format=False):
        super().__init__(alphabet)
        self.coord_pad_inf = coord_pad_inf
        self.to_pifold_format = to_pifold_format
        self.coord_nan_to_zero = coord_nan_to_zero

    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        """
        Args:
            raw_batch: List of tuples (coords, confidence, seq)
            In each tuple,
                coords: list of floats, shape L x n_atoms x 3
                confidence: list of floats, shape L; or scalar float; or None
                seq: string of length L
        Returns:
            coords: Tensor of shape batch_size x L x n_atoms x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        # self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, float) or isinstance(confidence, int):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        if self.coord_pad_inf:
            # pad beginning and end of each protein due to legacy reasons
            coords = [
                F.pad(torch.tensor(cd), (0, 0, 0, 0, 1, 1), value=np.nan)
                for cd, _ in coords_and_confidence
            ]
            confidence = [
                F.pad(torch.tensor(cf), (1, 1), value=-1.)
                for _, cf in coords_and_confidence
            ]
        else:
            coords = [
                torch.tensor(cd) for cd, _ in coords_and_confidence
            ]
            confidence = [
                torch.tensor(cf) for _, cf in coords_and_confidence
            ]
        # coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)
        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)

        if self.to_pifold_format:
            coords, tokens, confidence = ToPiFoldFormat(X=coords, S=tokens, cfd=confidence)

        lengths = tokens.ne(self.alphabet.padding_idx).sum(1).long()
        if device is not None:
            coords = coords.to(device)
            confidence = confidence.to(device)
            tokens = tokens.to(device)
            lengths = lengths.to(device)

        coord_padding_mask = torch.isnan(coords[:, :, 0, 0])
        coord_mask = torch.isfinite(coords.sum([-2, -1]))
        confidence = confidence * coord_mask + (-1.) * coord_padding_mask

        if self.coord_nan_to_zero:
            coords[torch.isnan(coords)] = 0.

        return coords, confidence, strs, tokens, lengths, coord_mask

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        """
        Args:
            coords_list: list of length batch_size, each item is a list of
            floats in shape L x 3 x 3 to describe a backbone
            confidence_list: one of
                - None, default to highest confidence
                - list of length batch_size, each item is a scalar
                - list of length batch_size, each item is a list of floats of
                    length L to describe the confidence scores for the backbone
                    with values between 0. and 1.
            seq_list: either None or a list of strings
        Returns:
            coords: Tensor of shape batch_size x L x 3 x 3
            confidence: Tensor of shape batch_size x L
            strs: list of strings
            tokens: LongTensor of shape batch_size x L
            padding_mask: ByteTensor of shape batch_size x L
        """
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """
        Takes a list of tensors with the following dimensions:
            [(d_11,       ...,           d_1K),
             (d_21,       ...,           d_2K),
             ...,
             (d_N1,       ...,           d_NK)]
        and stack + pads them into a single tensor of:
        (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
        """
        if len(samples) == 0:
            return torch.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]
        result = torch.empty(
            len(samples), *max_shape, dtype=samples[0].dtype, device=device
        )
        result.fill_(pad_v)
        for i in range(len(samples)):
            result_i = result[i]
            t = samples[i]
            result_i[tuple(slice(0, k) for k in t.shape)] = t
        return result


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()



class Featurizer(object):
    def __init__(self, alphabet: Alphabet, 
                 to_pifold_format=False, 
                 coord_nan_to_zero=True,
                 atoms=('N', 'CA', 'C', 'O')):
        self.alphabet = alphabet
        self.batcher = CoordBatchConverter(
            alphabet=alphabet,
            coord_pad_inf=alphabet.add_special_tokens,
            to_pifold_format=to_pifold_format, 
            coord_nan_to_zero=coord_nan_to_zero
        )

        self.atoms = atoms

    def __call__(self, raw_batch: dict):
        seqs, coords, names = [], [], []
        for entry in raw_batch:
            # [L, 3] x 4 -> [L, 4, 3]
            if isinstance(entry['coords'], dict):
                coords.append(np.stack([entry['coords'][atom] for atom in self.atoms], 1))
            else:
                coords.append(entry['coords'])
            seqs.append(entry['seq'])
            names.append(entry['name'])

        coords, confidence, strs, tokens, lengths, coord_mask = self.batcher.from_lists(
            coords_list=coords, confidence_list=None, seq_list=seqs
        )

        # coord_mask = coord_mask > 0.5
        batch = {
            'coords': coords,
            'tokens': tokens,
            'confidence': confidence,
            'coord_mask': coord_mask,
            'lengths': lengths,
            'seqs': seqs,
            'names': names
        }
        return batch
