#!/usr/bin/env python3
"""
Test script to verify ESM-2 vocabulary compatibility with CATH data loader
"""

import torch
from transformers import EsmTokenizer, EsmModel
from src.datamodules.data_utils import Alphabet
from load_cath import CATHDataModule

def test_vocab_compatibility():
    """Test if the current data loader is compatible with ESM-2 vocabulary"""
    
    print("=== ESM-2 Vocabulary Analysis ===")
    
    # Load ESM-2 tokenizer
    esm2_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    esm2_vocab = esm2_tokenizer.get_vocab()
    
    print(f"ESM-2 vocabulary size: {len(esm2_vocab)}")
    print(f"ESM-2 special tokens: {[k for k in esm2_vocab.keys() if k.startswith('<')]}")
    print(f"ESM-2 amino acids: {[k for k in esm2_vocab.keys() if k.isalpha() and len(k) == 1]}")
    
    # Print ESM-2 token mappings
    print("\nESM-2 token mappings:")
    for token, idx in sorted(esm2_vocab.items(), key=lambda x: x[1]):
        print(f"  {token}: {idx}")
    
    print("\n=== Current Data Loader Analysis ===")
    
    # Test current data loader setup
    cath_dm = CATHDataModule(
        data_dir="data/cath_4.3/",
        chain_set_jsonl="chain_set.jsonl",
        chain_set_splits_json="chain_set_splits.json",
        alphabet_name='esm2',
        max_length=500,
        batch_size=32,
        max_tokens=6000,
        sort=True,
        num_workers=0,  # Use 0 for testing
        pin_memory=False
    )
    
    # Setup the data module
    cath_dm.setup(stage='fit')
    
    print(f"Current alphabet name: {cath_dm.alphabet.name}")
    print(f"Current alphabet size: {len(cath_dm.alphabet)}")
    print(f"Current standard tokens: {cath_dm.alphabet.standard_toks}")
    print(f"Current prepend tokens: {cath_dm.alphabet.prepend_toks}")
    print(f"Current append tokens: {cath_dm.alphabet.append_toks}")
    
    # Print current token mappings
    print("\nCurrent token mappings:")
    for i in range(len(cath_dm.alphabet)):
        print(f"  {cath_dm.alphabet.get_tok(i)}: {i}")
    
    # Get a sample batch
    train_dataloader = cath_dm.train_dataloader()
    sample_batch = next(iter(train_dataloader))
    
    print(f"\nSample batch tokens shape: {sample_batch['tokens'].shape}")
    print(f"Sample batch token values range: {sample_batch['tokens'].min().item()} to {sample_batch['tokens'].max().item()}")
    
    # Check if tokens are within ESM-2 vocabulary range
    max_esm2_token_id = max(esm2_vocab.values())
    min_esm2_token_id = min(esm2_vocab.values())
    
    print(f"\nESM-2 token ID range: {min_esm2_token_id} to {max_esm2_token_id}")
    
    # Check if current tokens are compatible
    current_max_token = sample_batch['tokens'].max().item()
    current_min_token = sample_batch['tokens'].min().item()
    
    print(f"Current token ID range: {current_min_token} to {current_max_token}")
    
    if current_max_token <= max_esm2_token_id and current_min_token >= min_esm2_token_id:
        print("✅ COMPATIBLE: Current tokens are within ESM-2 vocabulary range")
    else:
        print("❌ INCOMPATIBLE: Current tokens exceed ESM-2 vocabulary range")
    
    # Test token conversion
    print("\n=== Token Conversion Test ===")
    
    # Get a sample sequence
    sample_seq = sample_batch['seqs'][0]
    print(f"Sample sequence: {sample_seq}")
    
    # Tokenize with ESM-2
    esm2_tokens = esm2_tokenizer.encode(sample_seq, add_special_tokens=False)
    print(f"ESM-2 tokenization: {esm2_tokens}")
    
    # Get current tokenization
    current_tokens = sample_batch['tokens'][0].tolist()
    # Remove padding tokens
    current_tokens = [t for t in current_tokens if t != cath_dm.alphabet.padding_idx]
    print(f"Current tokenization: {current_tokens}")
    
    # Decode both
    esm2_decoded = esm2_tokenizer.decode(esm2_tokens)
    current_decoded = ''.join([cath_dm.alphabet.get_tok(t) for t in current_tokens])
    
    print(f"ESM-2 decoded: {esm2_decoded}")
    print(f"Current decoded: {current_decoded}")
    
    if esm2_decoded == current_decoded:
        print("✅ Token conversion is consistent")
    else:
        print("❌ Token conversion is inconsistent")
        
        # Investigate the difference
        print("\nInvestigating token differences:")
        for i, (esm2_tok, curr_tok) in enumerate(zip(esm2_tokens, current_tokens)):
            esm2_char = esm2_tokenizer.decode([esm2_tok])
            curr_char = cath_dm.alphabet.get_tok(curr_tok)
            if esm2_char != curr_char:
                print(f"  Position {i}: ESM-2 '{esm2_char}' (token {esm2_tok}) vs Current '{curr_char}' (token {curr_tok})")
    
    # Test next token prediction compatibility
    print("\n=== Next Token Prediction Test ===")
    
    # Load ESM-2 model
    esm2_model = EsmModel.from_pretrained("facebook/esm2_t30_150M_UR50D")
    
    # Test with a short sequence
    test_seq = "MKTAYIAKQRQISFVKSHFSRQ"
    test_tokens = esm2_tokenizer.encode(test_seq, add_special_tokens=False, return_tensors="pt")
    
    print(f"Test sequence: {test_seq}")
    print(f"Test tokens: {test_tokens}")
    
    # Get current tokenization
    # test_my_tokens = sample_batch['tokens'][0]

    with torch.no_grad():
        # outputs = esm2_model(test_tokens)
        outputs = esm2_model(sample_batch['tokens'])
        # ESM-2 model returns hidden states, not logits directly
        hidden_states = outputs.last_hidden_state
    
    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Hidden state dimension: {hidden_states.shape[-1]}")
    
    # For next token prediction, we need to add a language modeling head
    # This would typically be done with EsmForMaskedLM or by adding a linear layer
    print("✅ ESM-2 model can be used for next token prediction with proper head")
    
    return True

if __name__ == "__main__":
    test_vocab_compatibility() 