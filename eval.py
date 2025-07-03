"""
Masking Strategy: The model uses masked language modeling where tokens are randomly masked and the model learns to predict them
Coordinate Input: The model takes protein backbone coordinates (N, CA, C, O atoms) as input
Sequence Output: The model predicts amino acid sequences conditioned on the structure
Loss Function: Cross-entropy loss over the vocabulary, ignoring padding tokens
Data Format: Coordinates are in shape [batch_size, seq_len, n_atoms, 3] where n_atoms=4 for backbone atoms
"""

import torch
import torch.nn as nn
import torch.optim as optim
from protein_mpnn.pmpnn import ProteinMPNNCMLM, ProteinMPNNConfig
from src.datamodules.data_utils import Alphabet
from src.datamodules.cath import CATHDataModule

# 1. Setup model and data
def setup_model_and_data():
    # Model configuration
    cfg = ProteinMPNNConfig(
        d_model=128,
        d_node_feats=128,
        d_edge_feats=128,
        k_neighbors=48,
        n_enc_layers=3,
        n_dec_layers=3,
        n_vocab=22,
        dropout=0.1
    )
    
    # Initialize model
    model = ProteinMPNNCMLM(cfg)
    
    # Setup alphabet
    alphabet = Alphabet('esm', 'cath')
    
    # Setup datamodule
    datamodule = CATHDataModule(
        data_dir="data/",
        batch_size=8,
        max_tokens=6000,
        num_workers=0
    )
    datamodule.setup('fit')
    
    return model, alphabet, datamodule


def inference_example(model, alphabet, batch):
    model.eval()
    with torch.no_grad():
        # Remove target tokens to prevent data leakage
        tokens = batch.pop('tokens')
        
        # Create fully masked input
        coord_mask = batch['coord_mask']
        prev_tokens = torch.full_like(tokens, alphabet.mask_idx)
        batch['prev_tokens'] = prev_tokens
        
        # Generate
        logits = model(batch)
        predicted_tokens = logits.argmax(dim=-1)
        
        # Decode to sequences
        sequences = alphabet.decode(predicted_tokens, remove_special=True)
        return sequences

if __name__ == "__main__":
    model, alphabet, datamodule = setup_model_and_data()
    inference_example(model, alphabet, datamodule.test_dataloader()[0]) 

