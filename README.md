# Sora

A Python project for processing and analyzing CATH protein structure data.

## Overview

This project provides utilities for downloading, loading, and processing CATH (Class, Architecture, Topology, Homology) protein structure data. CATH is a hierarchical classification of protein domains based on their structure and sequence.

## Features

- Download CATH data from official sources
- Load and process CATH chain sets
- Data utilities for protein structure analysis
- Support for CATH versions 4.2 and 4.3

## Files

- `cath.py` - Main CATH data processing module
- `data_utils.py` - Utility functions for data handling
- `load_cath.py` - CATH data loading functionality
- `download_cath.sh` - Script to download CATH data
- `data/` - Directory containing CATH data files

## Setup

1. Clone this repository
2. Install required dependencies
3. Run the download script to get CATH data

## Usage

```bash
# Download CATH data
./download_cath.sh

# Load and process CATH data
python load_cath.py
```

## Data Structure

The project supports CATH data in JSONL format with chain set information and splits for training/validation/testing.

## License

[Add your license information here] 