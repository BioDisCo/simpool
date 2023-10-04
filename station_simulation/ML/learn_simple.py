import sys
sys.path.append("../")

import numpy as np
import simulator
import config
import copy
import argparse
import math
import joblib
from joblib import Parallel, delayed
import contextlib
from tqdm import tqdm

from Infections.Agent import Symptoms, Category
from Infections.InfectionExecution import InfectionExecution
from pprint import pprint
import os.path
from os import path
from ML.mlutils import progressbar, get_loss_function 


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# -- weighting
CLASS_WEIGHTS = { 0 : 1.0 , 1 : 500.0 } # misclassifying 1 is worse
LINEAR_DISCOUNT_FACTOR = 500.0
# -- dataset
SIMULATED_TIME_DAYS = 100
TRAINING_TIME_DAYS = 40
# --
BATCH_SIZE = 10
DATASET_SIZE_IN_BATCHES = 10
EPOCHS_BEFORE_REGENERATE = 10
# -- how much training
REPETIONS = 1
EPOCHS = 10**6 # do not stop
STEPS_PER_EPOCH = DATASET_SIZE_IN_BATCHES # ok?
# -- multiprocessing
JOBNUMBER = -1 # -1 means all CPUs
MULTIPROCESSING_DATAGEN = True
#VALIDATION_STEPS = 2


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


# ===================================

config.T = TRAINING_TIME_DAYS
NETWORK_FILE = f'my_neural_network-{config.T}-{config.N}-new.h5'
AUTOMATIC_RESUME = True

#EXPEXTED_TEST_PER_DAY = 3
#EXPEXTED_QUARANTINES_PER_DAY = 1.5
#QUARANTINE_DAYS = 7

# testing with small N
EXPEXTED_TEST_PER_DAY = config.N / 5
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

# observation: 12 fields + contact graph
OBS_LENGTH = (12 * config.N) + (config.N * config.N)

def boolvec2str(vec):
    return ''.join(list(map( lambda b: '1' if b else '0',
                             vec )))

def get_properties(state_vec, infex):
    #  12 observables + observable contact graph
    symptom_none = []
    symptom_mild = []
    symptom_severe = []
    symptom_dead = []
    category_MED = []
    category_NUR = []
    category_ADM = []
    category_PAT = []
    working = []
    quarantined = []
    test_positive = []
    test_negative = []
    contactgraphs = []
    # non observable
    infected = []
    spreading = []
    # convert
    current_T = len(state_vec)
    current_N = len(state_vec[0])
    for t in range(current_T):
        state_dict = state_vec[t]
        vec = [ state_dict[agent_id] for agent_id in range(current_N) ]

        # symptom classes (1-hot)
        symptom_none += [ list(map( lambda x: x['symptoms'] is Symptoms.NONE,
                                    vec )) ]
        symptom_mild += [ list(map( lambda x: x['symptoms'] is Symptoms.MILD,
                                    vec )) ]
        symptom_severe += [ list(map( lambda x: x['symptoms'] is Symptoms.SEVERE,
                                      vec )) ]
        symptom_dead += [ list(map( lambda x: x['symptoms'] is Symptoms.DEAD,
                                    vec )) ]

        # category classes (1-hot)
        category_MED += [ list(map( lambda x: x['symptoms'] is Category.DOCTOR,
                                    vec )) ]
        category_NUR += [ list(map( lambda x: x['symptoms'] is Category.NURSE,
                                    vec )) ]
        category_ADM += [ list(map( lambda x: x['symptoms'] is Category.ADMIN,
                                      vec )) ]
        category_PAT += [ list(map( lambda x: x['symptoms'] is Category.PATIENT,
                                    vec )) ]

        # desease progress properties
        infected += [ list(map( lambda x: x['infected'],
                                    vec )) ]
        spreading += [ list(map( lambda x: x['spreading'],
                                    vec )) ]
        working += [ list(map( lambda x: x['working'],
                                    vec )) ]
        quarantined += [ list(map( lambda x: x['quarantined'],
                                    vec )) ]

        # test results
        test_positive += [ list(map( lambda x: x['testresult'] == True,
                                    vec )) ]
        test_negative += [ list(map( lambda x: x['testresult'] == False,
                                    vec )) ]

        # get contact graphs
        contactgraphs += [ infex.get_contactgraph_attime_withduration(t=t) ]
        #print('spreading')
        #pprint(spreading)
        #print(test_positive)
        #pprint(test_positive)

    return symptom_none, symptom_mild, symptom_severe, symptom_dead, \
           category_MED, category_NUR, category_ADM, category_PAT, \
           infected, spreading, \
           working, quarantined, \
           contactgraphs, \
           test_positive, test_negative


def get_run(usetest, initial_infections={'T_range': (0,10), 'N_range': (0,2)}):
    """
    run a simulation with the specified test.

    Arguments:
    Test as 'usetest'
    
    Returns:
    the run
    """
    results, infex, test, state_vec = simulator.run_sim(usetest,
        return_state_vec=True,
        initial_infections=initial_infections)
    #  12 observables
    symptom_none = []
    symptom_mild = []
    symptom_severe = []
    symptom_dead = []
    category_MED = []
    category_NUR = []
    category_ADM = []
    category_PAT = []
    working = []
    quarantined = []
    test_positive = []
    test_negative = []
    # contact graphs
    contactgraphs = []
    # non observable
    infected = []
    spreading = []
    # get properties
    symptom_none, symptom_mild, symptom_severe, symptom_dead, \
           category_MED, category_NUR, category_ADM, category_PAT, \
           infected, spreading, \
           working, quarantined, \
           contactgraphs, \
           test_positive, test_negative = get_properties(state_vec, infex)
    return symptom_none, symptom_mild, symptom_severe, symptom_dead, \
           category_MED, category_NUR, category_ADM, category_PAT, \
           infected, spreading, \
           working, quarantined, \
           contactgraphs, \
           test_positive, test_negative


def contactgraphs2matrix(contactgraphs):
    """
    Input:
    contact graph

    Returns:
    matrix : [0 .. T - 1] [0 .. config.N**2 - 1] -> weight of contact
    """
    # init
    current_T = len(contactgraphs)
    current_N = len(contactgraphs[0].keys())
    #print(current_N)
    matrix = [ [0 for _ in range(current_N * current_N)] for t in range(current_T) ]
    # fill
    for t in range(current_T):
        for i in range(current_N):
            for j in contactgraphs[t][i].keys():
                matrix[t][i * current_N + j] = contactgraphs[t][i][j]
    return matrix


def swap_agents(i, j, graphs, *fields):
    """
    swaps agents i and j
    <graphs> : the contact graphs
    <*fields> : properties of the agents like infected etc.

    No returned value, modifies graphs and fields directly.
    """
    current_T = len(graphs)
    #print(current_T)
    for t in range(current_T):
        # swap fields
        for field in fields:
            #assert ( i <= len(field)-1 and j <= len(field)-1 ), f"field={field}\ncurrent_T={current_T}\ni,j={i},{j}"
            #print(field)
            #print(i)
            #print(j)
            field[t][j], field[t][i] = field[t][i], field[t][j]
        # swap graphs
        graphs[t][j], graphs[t][i] = graphs[t][i], graphs[t][j]
        for k in graphs[t].keys():
            graphs[t][k] = { (i if out_neigh == j else (j if out_neigh == i else out_neigh)): value \
                            for (out_neigh,value) in graphs[t][k].items() }



class SimulationGenerator(object):

    def __init__(self, traces_in_dataset, usetest):
        self.traces_in_dataset = traces_in_dataset
        self.usetest = usetest

    def generate_data(self):
        #print('Generate data')
        # switch to a larger time that is to be simulated
        config.T = SIMULATED_TIME_DAYS
        # simulate
        symptom_none, symptom_mild, symptom_severe, symptom_dead, \
               category_MED, category_NUR, category_ADM, category_PAT, \
               infected, spreading, \
               working, quarantined, \
               contactgraphs, \
               test_positive, test_negative = get_run(usetest=self.usetest)
        # switch back to training time days
        config.T = TRAINING_TIME_DAYS
        # cut out TRAINING_TIME_DAYS from the SIMULATED_TIME_DAYS long run
        start_time = random.randrange(SIMULATED_TIME_DAYS - TRAINING_TIME_DAYS)
        end_time = start_time + TRAINING_TIME_DAYS
        # cutting
        symptom_none = symptom_none[start_time:end_time]
        symptom_mild = symptom_mild[start_time:end_time]
        symptom_severe = symptom_severe[start_time:end_time]
        symptom_dead = symptom_dead[start_time:end_time]
        category_MED = category_MED[start_time:end_time]
        category_NUR = category_NUR[start_time:end_time]
        category_ADM = category_ADM[start_time:end_time]
        category_PAT = category_PAT[start_time:end_time]
        infected = infected[start_time:end_time]
        spreading = spreading[start_time:end_time]
        working = working[start_time:end_time]
        quarantined = quarantined[start_time:end_time]
        contactgraphs = contactgraphs[start_time:end_time]
        test_positive = test_positive[start_time:end_time]
        test_negative = test_negative[start_time:end_time]
        # a spreading agent has prob 0.5 to be the special agent 0
        if random.random() < 0.5:
            # choose an arbitrary one as agent 0
            i = random.randrange(config.N)
        else:
            # choose a spreading one as agent 0
            spreading_list = [ j for j in range(config.N) if sum( spreading[t][j] for t in range(config.T) ) > 0 ]
            if spreading_list == []:
                # there is noone srpeading -> arbitrary choice
                i = random.randrange(config.N)
            else:
                i = random.choice(spreading_list)

        # swap agent i into agent 0
        swap_agents(i, 0, contactgraphs,
            symptom_none,
            symptom_mild,
            symptom_severe,
            symptom_dead,
            category_MED,
            category_NUR,
            category_ADM,
            category_PAT,
            infected,
            spreading,
            working,
            quarantined,
            test_positive,
            test_negative)
        # generate x and y
        # symptom_none : [0 .. config.T - 1] [0 .. config.N - 1] -> {0,1}
        #   etc.
        # concatenate along 2nd component (axis=1)
        x = np.concatenate((symptom_none, symptom_mild, symptom_severe, symptom_dead, \
                    category_MED, category_NUR, category_ADM, category_PAT, \
                    working, quarantined, test_positive, test_negative, contactgraphs2matrix(contactgraphs)), axis=1)
        # x : [0 .. config.T - 1] [0 .. (config.N - 1) * K] -> attributes
        #   where K = OBS_LENGTH
        #
        # speading : [0 .. config.T - 1] [0 .. config.N - 1] -> {0,1}
        y = [ agent_t[0] for agent_t in spreading ]
        # y : [0 .. config.T - 1] -> {0,1}
        #   if agent 0 is spreading at time t
        #print('[done generate data]')
        return x, y

    def generate_traces(self):
        #print('Generate batch')
        X_data = []
        Y_data = []
        weights = []
        
        if MULTIPROCESSING_DATAGEN:
            #results = Parallel(n_jobs=JOBNUMBER)( delayed(self.generate_data)() for i in range(self.traces_in_dataset) )
            with tqdm_joblib(tqdm(desc=f"Dataset with {joblib.cpu_count()} CPUs: ", total=self.traces_in_dataset)) as progress_bar:
                results = Parallel(n_jobs=JOBNUMBER)( delayed(self.generate_data)() for i in range(self.traces_in_dataset) )
            
            for i in range(self.traces_in_dataset):
                x = results[i][0]
                y = results[i][1]
                w = [ CLASS_WEIGHTS[y[t]] + y[t] * sum(y[:t]) for t in range(len(y)) ]
                X_data.append(x)
                Y_data.append(y)
                weights.append(w)
        else:
            assert (False), "TODO: needs to be adapted!"
            for _ in progressbar(range(self.traces_in_dataset), "Dataset: "):
                x, y = self.generate_data()
                # ATTENTION: This assume that the same agent cannot be spreading again.
                w = [ CLASS_WEIGHTS[y[t]] + y[t] * sum(y[:t]) for t in range(len(y)) ]
                X_data.append(x)
                Y_data.append(y)
                weights.append(w)
        # just convert to same shape np-array
        X_data = np.array(X_data).reshape(self.traces_in_dataset, config.T, OBS_LENGTH)
        # convert true,false to 0,1. Then also convert to same shape np-array
        Y_data = np.array(Y_data).astype(int).reshape(self.traces_in_dataset, config.T, 1)
        weights = np.array(weights).reshape(self.traces_in_dataset, config.T, 1)
        #print('[done generate batch]')
        return X_data, Y_data, weights


    """
    def data_generator(self):
        while True:
            X_data, Y_data, weights = self.generate_batch()
            yield X_data, Y_data, weights
    """


class DataGenerator(Sequence):
    def __init__(self, batch_size, dataset_size, epochs_before_regenerate, usetest):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.epochs_before_regenerate = epochs_before_regenerate
        self.simulation_generator = SimulationGenerator(traces_in_dataset=batch_size * dataset_size, usetest=usetest)
        self.current_epoch = 0
        # for shuffling and setpping
        self.indices = np.arange( batch_size * dataset_size )
        # our dataset
        self.__generate_dataset()

    def __len__(self):
        return self.dataset_size

    def __generate_dataset(self):
        print('Generate new dataset')
        # get traces for dataset
        self.x, self.y, self.weights =  self.simulation_generator.generate_traces()
        # reshape them
        traces_nr = self.batch_size * self.dataset_size
        self.x = np.array(self.x).reshape(traces_nr, config.T, OBS_LENGTH)
        self.y = np.array(self.y).reshape(traces_nr, config.T, 1)
        self.weights = np.array(self.weights).reshape(traces_nr, config.T, 1)
        #print("exit")

    def __getitem__(self, idx):
        #print(f'get {idx}')
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y = self.y[inds]
        batch_weights = self.weights[inds]
        return batch_x, batch_y, batch_weights
    
    def on_epoch_end(self):
        #print('epoch end')
        self.current_epoch += 1
        if (self.current_epoch >= self.epochs_before_regenerate):
            # generate new dataset
            self.__generate_dataset()
            self.current_epoch = 0
        else:
            # just reshuffle
            print('Shuffling dataset')
            np.random.shuffle(self.indices)
            pass


def predict_agent0(model, input_vector, window_size):
    """
    ensure:
      input_vector shape = (1, config.T, OBS_LENGTH)
      DATA_SIZE is 1 since has only 1 run to predict
    """
    T = input_vector.shape[1]
    # intially predict no infection
    pred_timeline = np.zeros( (window_size,1) )
    for current_time in range(T - window_size):
        # cut out current time window
        T_data = [ input_vector[:,current_time:current_time+window_size,:] ]
        # predict single output for sliding window
        pred = np.array( model(T_data) )
        pred = pred.reshape(1,1)
        pred_timeline = np.concatenate((pred_timeline, pred), axis=0)
    
    return pred_timeline



def plot_run(model, usetest, nogui=False, filename="learn_simple-out.png", T=SIMULATED_TIME_DAYS,
             initial_infections={'T_range': (0,10), 'N_range': (0,2)}):
    print('Plotting run...')
    
    config.T = T

    symptom_none, symptom_mild, symptom_severe, symptom_dead, \
    category_MED, category_NUR, category_ADM, category_PAT, \
    infected, spreading, \
    working, quarantined, \
    contactgraphs, \
    test_positive, test_negative = get_run(usetest=usetest, initial_infections=initial_infections)
    # prepare info for plotting
    info = (np.array(symptom_none) * 0.2 + np.array(symptom_mild) * 0.5 + np.array(symptom_severe) * 0.8) * np.array(spreading)
    # prediction for all agents
    all_pred = None
    for i in progressbar(range(config.N), "Predicting agents: "):
        # bring agent i into position 0
        swap_agents(i, 0, contactgraphs,
            symptom_none,
            symptom_mild,
            symptom_severe,
            symptom_dead,
            category_MED,
            category_NUR,
            category_ADM,
            category_PAT,
            infected,
            spreading,
            working,
            quarantined,
            test_positive,
            test_negative)
        # build input        
        x = np.concatenate((symptom_none, symptom_mild, symptom_severe, symptom_dead, \
                category_MED, category_NUR, category_ADM, category_PAT, \
                working, quarantined, test_positive, test_negative, contactgraphs2matrix(contactgraphs)), axis=1)
        T_data = [x]
        # DATA_SIZE is 1 since has only 1 run to predict
        T_data = np.array(T_data).reshape(1, config.T, OBS_LENGTH)
        # predict agent
        pred = predict_agent0(model=model,
                              input_vector=T_data,
                              window_size=TRAINING_TIME_DAYS)
        # add to all predictions
        if all_pred is None:
            all_pred = pred
        else:
            all_pred = np.concatenate((all_pred, pred), axis=1)
        # swap back! Otherwise this will mess with the vectors that are diplayed next!
        swap_agents(i, 0, contactgraphs,
            symptom_none,
            symptom_mild,
            symptom_severe,
            symptom_dead,
            category_MED,
            category_NUR,
            category_ADM,
            category_PAT,
            infected,
            spreading,
            working,
            quarantined,
            test_positive,
            test_negative)

    #print(info)
    plt.figure(figsize=(9,9))
    
    # spreading
    plt.subplot(1, 2, 1)
    plt.imshow(info, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    # tested
    my_reds = copy.copy(cm.get_cmap("Reds"))
    my_reds.set_under('k', alpha=0)
    plt.imshow( 0.5 * np.array(test_positive), cmap=my_reds, vmin=0.1, vmax=1, interpolation='none')
    my_greens = copy.copy(cm.get_cmap("Greens"))
    my_greens.set_under('k', alpha=0)
    plt.imshow( 0.5 * np.array(test_negative), cmap=my_greens, vmin=0.1, vmax=1, interpolation='none')

    # predicted
    plt.subplot(1, 2, 2)
    plt.imshow(1-all_pred, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    
    plt.tight_layout()
    print('[done plotting]')
    if not nogui:
        plt.show(block=True)
    else:
        plt.savefig(filename, dpi=300)


def load_NN(nn_filename):
    """
    loads an NN model and returns it.
    """
    print(f'Load model weights from {nn_filename} ...')
    try:
        model = create_NN()
        model.load_weights(nn_filename)
        print('[done loading]')
    except OSError as e:
        print(f'ERROR: Could not open model: {e}')
        exit(1)
    model.summary()
    return model


def create_NN():
    """
    creates a new NN model and returns it.
    """
    batchparam = None
    model = Sequential()
    model.add( LSTM(1024,
               name="input",
               return_sequences=True,
               activation="tanh",
               input_shape=(None,OBS_LENGTH),
               dropout=0.2,
               batch_input_shape=(batchparam,None,OBS_LENGTH) )
             )
    model.add( LSTM(128,
               name="hidden",
               return_sequences=False,
               dropout=0.2,
               activation="tanh" )
              )
    model.add( Dense(1,
               name="output",
               activation="sigmoid" )
             )
    # -- optimizer
    # since binary output using binary_crossentropy
    model.compile( optimizer="adam",
                   loss="binary_crossentropy",
                   sample_weight_mode="temporal",
                 )
    return model


def resume_NN(nn_filename, automatic_resume=True):
    """
    opens an existing NN model from <nn_filename>
    if it exists and <automatic_resume> == True,
    else creates a new model.

    Returns the model.
    """
    # check if trained network exists and take it
    if path.exists(nn_filename) and automatic_resume:
        print('Model found. Resume training...')
        model = load_NN(nn_filename)
    else:
        print('Model not found or no automatic resume. Create model...')
        model = create_NN()
        print('[done create model]')
        model.summary()
    return model




# =========================================


# main
if __name__ == "__main__":
    # parsing
    parser = argparse.ArgumentParser(description='Learn infection models.')
    parser.add_argument('--nogui', action='store_true',
                        help='disable the gui')
    parser.add_argument('--fresh', action='store_true',
                        help='start training from scratch and reset the NN file')
    parser.add_argument('--onlyplot', action='store_true',
                        help='only plot, no training')
    args = parser.parse_args()

    if args.fresh:
        print("Warning: will overwrite NN file if it exists.")
        AUTOMATIC_RESUME = False

    # turn off standard output
    print('Learning started...')
    config.plotting = False
    config.output_singleruns = False
    config.output_summary = False
    config.T = TRAINING_TIME_DAYS # Important: train on shorter timeframe
    # choose test
    usetest = mytests[1]
    # number of runs to get
    N_runs = 1
    model = resume_NN(NETWORK_FILE, AUTOMATIC_RESUME)
    
    if not args.onlyplot:
        #train
        try:
            for i in range(REPETIONS):
                # train
                print('Fitting...')
                mydatagenerator = DataGenerator(batch_size=BATCH_SIZE,
                                                dataset_size=DATASET_SIZE_IN_BATCHES,
                                                epochs_before_regenerate=EPOCHS_BEFORE_REGENERATE,
                                                usetest=usetest)
                history = model.fit(mydatagenerator,
                                    #use_multiprocessing=True, workers=2, 
                                    epochs=EPOCHS,
                                    steps_per_epoch=STEPS_PER_EPOCH,
                                    batch_size=BATCH_SIZE,
                                    verbose=1,
                                    #validation_data=data_generator(),
                                    #validation_steps=VALIDATION_STEPS,
                                    callbacks=[
                                        ModelCheckpoint(filepath=NETWORK_FILE,
                                            save_weights_only=True,
                                            monitor='loss',
                                            mode='min',
                                            save_best_only=False)])
                print('[done fitting]')
                # save
                #print(f'Save after {i+1} repetitions...')
                #model.save(NETWORK_FILE)
                #print('[done saving]')
                # plot
                #print(history.history)
                plt.figure(figsize=(9,9))
                labels = ["loss",]
                for lab in labels:
                    plt.plot( history.history[lab],label=f"{lab} model" )
                plt.yscale("log")
                plt.legend()
                if not args.nogui:
                    #plt.show(block=True)
                    pass
                else:
                    plt.savefig(f'hist-{config.T}-{config.N}.png', dpi=300)

        except KeyboardInterrupt as e:
            print(f'Stopped training by user.')

    # plot
    try:
        plot_run(model, usetest, nogui=args.nogui)
    except KeyboardInterrupt as e:
        print(f'Stopped plotting by user.')
