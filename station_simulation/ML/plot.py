import sys
sys.path.append("../")

import numpy as np
import simulator
import config
import copy
import argparse

from Infections.Agent import Symptoms, Category
from Infections.InfectionExecution import InfectionExecution
from pprint import pprint
import os.path
from os import path
from ML.mlutils import progressbar, get_loss_function

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random
import tensorflow as tf
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras import layers
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import load_model
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm as cm


from learn_simple import plot_run, load_NN


# ===================================
NETWORK_FILE = f'my_neural_network-{config.T}-{config.N}-new.h5'

# testing with small N
EXPEXTED_TEST_PER_DAY = config.N / (4*config.N)
EXPEXTED_QUARANTINES_PER_DAY = config.N / 10
QUARANTINE_DAYS = 7

# tests:
mytests = {}
mytests[0] = {
    'type': 'TestNull',
    'parameters': {}
}
mytests[1] = {
    'type': 'TestRandom',
    'parameters': {
        'prob_test': EXPEXTED_TEST_PER_DAY / config.N,
        'prob_quarantine': EXPEXTED_QUARANTINES_PER_DAY / config.N,
        'quarantine_days': QUARANTINE_DAYS }
}
# =========================================

# observation: 12 fields + contact graph
OBS_LENGTH = (12 * config.N) + (config.N * config.N)
TRAINING_TIME_DAYS = 40

# main
if __name__ == "__main__":
    # parsing
    OUTPUT_FIGURES_NUM_default = 10
    T_default = 4*7*3
    parser = argparse.ArgumentParser(description='Plot some infection predictions.')
    parser.add_argument('--figs', type=int, help=f'number of figures to generate (default {OUTPUT_FIGURES_NUM_default})')
    parser.add_argument('--T', type=int, help=f'time to predict in days (default {T_default})')
    args = parser.parse_args()

    if args.figs:
        OUTPUT_FIGURES_NUM = args.figs
    else:
        OUTPUT_FIGURES_NUM = OUTPUT_FIGURES_NUM_default

    if args.T:
        config.T = args.T
    else:
        config.T = T_default
    print(f"Generating {OUTPUT_FIGURES_NUM} figures.")

    # turn off standard output
    config.plotting = False
    config.output_singleruns = False
    config.output_summary = False
    # choose test
    usetest = mytests[1]

    try:
        model = load_NN(NETWORK_FILE)
        for r in range(OUTPUT_FIGURES_NUM):
            plot_run(model, usetest, nogui=True, filename=f"plot-{r:03d}-out.png", T=config.T,
                initial_infections={'T_range': (TRAINING_TIME_DAYS,TRAINING_TIME_DAYS+7), 'N_range': (1,2)})

    except KeyboardInterrupt as e:
        print(f'Stopped plotting by user.')
