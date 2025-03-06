#!/bin/bash

# Loading the required module
module load anaconda/Python-ML-2025a
pip install -e scribe_agent

# Run the script
python data_creator.py