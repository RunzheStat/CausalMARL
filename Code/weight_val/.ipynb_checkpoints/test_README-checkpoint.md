## TestMA
This repository contains the code for the paper "Does the Markov Decision Process Fit the Data: Testing for the Markov Property in Sequential Decision Making", submitted to ICML 2020. 

## Installation
* Python version: Python 3.6.8 :: Anaconda custom (64-bit)
* Packages use in this project are included in common Python distributions. 
    * Main packages include `numpy, sklearn, scipy, pandas, random, pickle, statsmodels, operator, itertools` and `multiprocessing`. 
    * Additional packages include `time, warnings, smtplib, ssl, os` and `sys`.
    

## File Overview
1. Files in the main folder: scripts to reproduce results in our paper. 
2. Files in the `/code_data` folder: main functions for the proposed test, our experiments and some supporting functions.
    1. The proposed test
        1. `_core_test_fun.py`: main functions for the proposed test, including Algorithm 1 and 2 in the paper, and their componnets.
        5. `_QRF.py`: the random forests regressor used in our experiments.
    2. Experiments
        2. `_DGP_Ohio.py`: simulate data and evaluate policies for the HMDP synthetic data section.
        3. `_DGP_TIGER.py`: simulate data for the POMDP synthetic data section.
        4. `_Funcs_Real_Ohio.py`: functions used in our HMDP real data experiment
        7. `_utility_RL.py`: RL algorithms used in the experiments, including FQI, FQE and related functions.
    6. `_uti_basic.py` and `_utility.py`: helper functions
    9. `Data_Ohio.csv`: The OhioT1DM dataset used in our experiment, processed from the dataset provided by the paper "The OhioT1DM dataset for blood glucose level prediction". 

## How to reproduce results
Simply run the corresponding scripts. Relative file paths may be needed.

1. Figure 2: `Ohio_simu_testing.py`
2. Figure 3: `Ohio_simu_values.py` and `Ohio_simu_seq_lags.py`
3. Figure 4: `Tiger_simu.py`
4. Table 1: `Ohio_real.py`
