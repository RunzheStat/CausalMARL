# Spatiotemporal Causal Effects Evaluation: A Multi-Agent Reinforcement Learning Framework

This repository is the official implementation of the paper "Spatiotemporal Causal Effects Evaluation: A Multi-Agent Reinforcement Learning Framework" submitted to NeurIPS 2020. 


## Requirements
* Python version: Python 3.6.8 :: Anaconda custom (64-bit)
* Main packages for the proposed estimator
    - numpy
    - scipy
    - sklearn
    - itertools
* Additional packages for experiments
    - pickle
    - multiprocessing
    - os
    - time
    - sys
    - logging
    - warnings


## File Overview
* `main.py`: main function for the proposed estimator and its components
* `weight.py`: neural network for the weight estimation part
* `utils.py`: helper functions for `main.py` and simulation parts
* `simu.py`: experiment script for reproducing the results in the paper
* `simu_funs.py`: main functions for the simulation experiment
* `simu_DGP.py`: data generating functions for the simulation experiment

## How to reproduce results

Simply run `simu.py` to reproduce the simulation results presented in Figure 2. 

