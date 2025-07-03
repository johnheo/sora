# Sora


## Overview

Exploring structure-informed sequence prediction with CATH (Class, Architecture, Topology, Homology) protein structure data.

## Setup

1. Clone this repository
2. Install required dependencies
3. Run the download script to get CATH data

## Usage

```bash
# Download CATH data
./download_cath.sh

# Train a ESM2 model with CATH dataloader
sh train.sh
```

> Modified from [ByProt](https://github.com/BytedProtein/ByProt)

