# Sora


## Overview

Exploring structure-informed sequence prediction with CATH (Class, Architecture, Topology, Homology) protein structure data.


```
sora
├── configs                
│   ├── callbacks        # logging, checkpoints
│   ├── datamodule       # dataloading
│   ├── experiment       # models
│   ├── hydra            # Hydra manager
│   ├── logger           # CSV, WandB
│   ├── paths            # data, logs, outputs
│   └── trainer          # torch lightning 
├── data                  
│   ├── cath_4.2          
│   └── cath_4.3          
├── src                   
│   ├── datamodules
│   ├── models    
│   ├── modules    
│   ├── tasks     
│   └── utils    
```

## Setup

1. **Data Download**: Run `./download_cath.sh` to populate `data/`
2. **Configuration**: Modify `configs/` files for your experiment
3. **Training**: Execute `train.sh` or `python train.py`
4. **Outputs**: Results automatically saved to `logs/` with experiment naming

## Usage

```bash
# Download CATH data
./download_cath.sh

# Train a ESM2 model with CATH dataloader
sh train.sh
```

> Modified from [ByProt](https://github.com/BytedProtein/ByProt)

