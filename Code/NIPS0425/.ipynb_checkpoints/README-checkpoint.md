# My Paper Title

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

> 📋Optional: include a graphic [that one?]
explaining your approach/main result

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

Simply run `simu.py` and pickle.load the output files to get the simulation results.
1. Figure 2: a / b
说清楚 ec2 output 的结构(两类 + 那个 grid + 那些图)，怎么结合起来
N_targets: a list (len-N_target) of list of [bias, std, MSE, ...] (each of these measures is a vector)
see my plot.ipynb

> 📋Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 

## References

If you find this code is useful for your research, please cite our paper

```
@article{lu2019vilbert,
  title={ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  journal={arXiv preprint arXiv:1908.02265},
  year={2019}
}
```