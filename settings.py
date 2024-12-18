__author__ = "Luofeng Zhou"
"""
This file contains settings of this project
"""
import numpy as np
import pandas as pd
import os
import re
import time
import datetime
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle
import glob
from torch.utils.data.dataset import Dataset
from functools import reduce
from pandas.tseries.offsets import MonthEnd
from sklearn.metrics import r2_score
import lzma
import sys

# from torch.masked import masked_tensor, as_masked_tensor

# System variables
import matplotlib
terminal_run = 'pycharm'  # other choice: 'pycharm', 'grid', 'local'
if terminal_run == 'local':
    matplotlib.use('TkAgg')

torch.autograd.set_detect_anomaly(True)

parameters = {
    'learning_rate': 0.0025,  # in model helper  # 0.001
    'learning_rate_holdings': 0.00001,  # for holdings NN
    'optimizer': 'adam',  # in model helper
    'intermediate_dims': (128, 64, 32, 16),  # in model definition
    # 'intermediate_dims_holdings': (32, 16),
    # 'intermediate_dims': (32, 16, 8),  # in model definition
    'dropout': 0.05,  # in model definition
    'l1_regularization': 1e-8,  # in model helper
    'l2_regularization': 1e-6,  # in model helper
    'epoch': 64,  # 64
    # 'epoch_holdings': 4,
}

# define directory
base = 'C:/Users/34308/Documents/Doc/CS/6998_013/bond_fund/'
# base = 'E:/bond_fund_project_data/'

log_process_name = base + 'log.log'
input_directory = base + 'Input/'
intermediate_directory = base + "Intermediate/"
network_directory = intermediate_directory + "Networks/"

# char_file = input_directory + 'char_fundonly_4rxfactor_v6.csv'
char_file = input_directory + "char_withholdings_4rxfactor_v8_20240919.csv"
char_file_triple = char_file.split(".csv")[0] + "_tri.csv"

# macro files
char_file_macro_cfnai = 'cfnai-data-series-xlsx.xlsx'
char_file_sentiment = 'Investor_Sentiment_Data_20220821_POST.xlsx'
char_file_fisd_longterm = 'fisd_longterm.csv'
char_file_fisd_highyield = 'fisd_highyield.csv'
char_file_inflation_exp = 'Inflation expectations.xlsx'
char_file_cpi = 'CPIAUCSL.csv'
char_file_spread = 'ebp_csv.csv'

# when the batch size is very large, then it is simply a full batch mode
nn_batch_size = 10240000

holdings_char_flag = False
chronological_flag = True

random_seed = 1

fund_char_variables = ['FundId', 'Month', 'flow', 'aum', 'dividend_yield', 'exp_ratio_net', 'fee_rate',
                       'turnover_ratio',
                       'st_reversal', 'st_momentum', 'momentum', 'int_momentum', 'lt_momentum', 'lt_reversal',
                       'age', 'R_squared', 'fund_st_reversal', 'fund_st_momentum', 'fund_momentum', 'family_flow',
                       'family_aum', 'family_age', 'family_fund_momentum', 'no_funds',
                       'future_ab_monthly_return']

holdings_char_variables = ['FundId', 'Month', 'future_ab_monthly_return',
                           'maturity',
                           'maturity_pos',
                           'coupon',
                           'coupon_pos',
                           'toptenshare',
                           'topfiveshare',
                           'toponeshare',
                           'hhi',
                           'holdings_num',
                           'gov_share',
                           'abs_share',
                           'corp_share',
                           'muni_share',
                           'cash_share',
                           'convert_share',
                           'rule144a_share',
                           'shortterm_share',
                           'ultrashortterm_share',
                           'usd_share',
                           'eur_share',
                           'nonusa_share',
                           'zerocoupon_share',
                           'ownership_market_val',
                           'price_count',
                           'Coupon Frequency Description_1',
                           'Coupon Frequency Description_2',
                           'Coupon Type_1',
                           'Coupon Type_2',
                           'Issue Price',
                           'Is Callable_1',
                           'Is Putable_1',
                           'Offering Type Description_1',
                           'Offering Type Description_2',
                           'Offering Type Description_3',
                           'Offering Type Description_4',
                           'Offering Type Description_5',
                           'Purpose Description_1',
                           'Purpose Description_2',
                           'Purpose Description_3',
                           'Purpose Description_4',
                           'Purpose Description_5',
                           'Is Convertible_1',
                           'volume_aggexternal',
                           'spread_aggexternal',
                           'yield_aggexternal',
                           'zero_trading_day_aggexternal']

