# version as of April 3, 2021

import numpy as np
import enum
import random
from random import shuffle
import torch
from matplotlib import pylab as plt
from copy import deepcopy, copy
#import dill as pickle
import pickle



n_agents = 20
CAPACITY = 5 # number of available tests per day
LOCAL_ACTIONS_NR = 2

class Ctrl_Action(enum.Enum):
    skip = 0
    test = 1
#    quarantine = 2
    def __str__(self):
        return str(self.name)


class Env_Action(enum.Enum):
    skip = 0
    def __str__(self):
        return str(self.name)


class Player(enum.Enum):
    Ctrl = 0
    Env = 1


# ==========================================
# Definition of model
# ==========================================

'''
L = [0,1] x {0,1}
    probability of being infected x selected/not yet selected 
C = {0,...,capacity}
    number of remaining tests and per agent
G = {'dummy'}
'''

class Glob_State:
    def __init__(self):
        #self.value = 'dummy'
        self.undetected = [0] * n_agents
        
    def __str__(self):
        return f"""{self.undetected}"""

class Ctrl_State:
    def __init__(self):
        self.capacity = CAPACITY

    def vector(self):
        return [self.capacity]

    def __str__(self):
        return f"""{self.capacity}"""


class Loc_State:
    def __init__(self):
        self.prob = round(random.uniform(0, 1),1)
        self.test = 0

    def vector(self):
        return [self.prob, self.test]
    
    def __str__(self):
        return f"""< {self.prob}, {self.test} >"""
    
    def __lt__(self,other):
        return(self.prob <= other.prob)



class State: # \mathcal{G}
    def __init__(self):
        self.glob = Glob_State()
        self.ctrl = Ctrl_State()
        self.loc = sorted([Loc_State() for _ in range(n_agents)])
        #probs = ([1] * CAPACITY) + ([0] * (n_agents - CAPACITY))
        ##shuffle(probs)
        #for i in range(n_agents):
        #    self.loc[i].prob = probs[i]

    @property
    def observation(self):
        return self.ctrl.vector() + [item for sublist in [self.loc[i].vector() for i in range(n_agents)] for item in sublist]
    
    
    def __str__(self):
        probs = [self.loc[i].prob for i in range(n_agents)]
        tests = [self.loc[i].test for i in range(n_agents)]
        return \
f"""\
===== state =====
Undetected (G): {self.glob}
Capacity (C): {self.ctrl}
Infected: {probs}
Tested__: {tests}
================="""


class Seq_State:
    def __init__(self):
        self.state = State()
        self.permutation = random.sample(list(range(n_agents)), n_agents)
        self.index = 0
        self.seen = set()

    @property
    def observation(self):
        index_list = [0] * n_agents
        index_list[self.current_agent] = 1
        binary_seen = [1 if i in self.seen else 0 for i in range(n_agents)]
        return np.array(self.state.observation + index_list + binary_seen)
    
    @property
    def actions(self):
        ctrl = self.state.ctrl
        ell = self.state.loc[self.current_agent]
        return actions_odot(ctrl, ell)
   
    @property
    def current_agent(self):
        return self.permutation[self.index]
    
    def __str__(self):
        return \
f"""{self.state}
Perm: {self.permutation}
Agent: {self.current_agent}
Seen: {self.seen}
================="""


def odot(ctrl: Ctrl_State, ell: Loc_State, action: Ctrl_Action):
    next_ctrl = deepcopy(ctrl)
    if next_ctrl.capacity > 0 and action == Ctrl_Action.test:
        next_ctrl.capacity -= 1
    return next_ctrl

def actions_odot(ctrl: Ctrl_State, ell: Loc_State):
    #return set([Ctrl_Action.skip.value, Ctrl_Action.test.value])
    actions = [Ctrl_Action.skip.value]
    if ctrl.capacity > 0:
        actions = actions + [Ctrl_Action.test.value]
    return set(actions)

def delta_loc(ell: Loc_State, action: Ctrl_Action):
    next_ell = deepcopy(ell)
    next_ell.test = action.value
    return next_ell


def reward_loc(ctrl: Ctrl_State, ell: Loc_State, action: Ctrl_Action):
    if ctrl.capacity <= 0 and action == Ctrl_Action.test:
        return -100
    return 0


def delta_glob(state: State, action: Env_Action):
    undetected = [0] * n_agents
    for i in range(n_agents):
        if random.uniform(0, 1) <= state.loc[i].prob:
            undetected[i] = 1 - state.loc[i].test
    next_state = deepcopy(state)
    next_state.glob.undetected = undetected
    return next_state
        

def reward_glob(state: State, action, next_state: State):
    return sum(next_state.glob.undetected) * -100


# ==========================================
# Sequential POMDP
# ==========================================



def tau(state: State, i, action: Ctrl_Action):
    next_state = deepcopy(state)
    next_state.ctrl = odot(next_state.ctrl, next_state.loc[i], action)
    next_state.loc[i] = delta_loc(next_state.loc[i], action)
    return next_state
  

def seq_delta_ctrl(seq_state: Seq_State, action: Ctrl_Action):
    next_seq_state = deepcopy(seq_state)
    next_seq_state.state = tau(next_seq_state.state, next_seq_state.current_agent, action)
    if next_seq_state.index < n_agents - 1:
        next_seq_state.seen.add(next_seq_state.current_agent)
        next_seq_state.index += 1
        return next_seq_state
    return next_seq_state.state


def seq_reward_ctrl(seq_state: Seq_State, action: Ctrl_Action):
    i = seq_state.current_agent
    ctrl = seq_state.state.ctrl
    ell = seq_state.state.loc[i]
    return reward_loc(ctrl, ell, action)


def seq_delta_env(state: State, action: Env_Action):
    seq_state = Seq_State()
    seq_state.state = delta_glob(state, action)
    return seq_state


def seq_reward_env(state: State, action, next_state: Seq_State):
    return reward_glob(state, action, next_state.state)


'''
seq_state = Seq_State()

seq_state.seen()

reward_glob(seq_state.state, Env_Action.skip, seq_state.state)

odot(seq_state.state.ctrl, seq_state.state.loc[1], Ctrl_Action.test)

reward_glob(seq_state.state, None, None)


seq_delta_ctrl(seq_state, Ctrl_Action.test)

seq_state.observation

seq_reward_ctrl(seq_state, Ctrl_Action.test)
'''

def simulate_step(model):
    current_state = Seq_State()
    #print(current_state)
    
    act = [None] * n_agents
    
    for i in range(n_agents):
    
        observation_ = current_state.observation.reshape(1,input_length) + np.random.rand(1,input_length)/1000.0
        observation1 = torch.from_numpy(observation_).float()
            
        qval = model(observation1)
        qval_ = qval.data.numpy()
    
        filtered_qvals = [ qval_[0][act] if act in current_state.actions \
                              else -float('Inf') for act in range(LOCAL_ACTIONS_NR) ]
        action_ = np.argmax(filtered_qvals)
    
        action = Ctrl_Action(action_)
        
        #print('>>>>>>>>>> Local action:', action)
        act[current_state.current_agent] = action.value
        
        current_state = seq_delta_ctrl(current_state, action)
        
        #print(current_state)
        
    #print(act)
    
    next_state = seq_delta_env(current_state, Env_Action.skip)
    print(next_state)
    print('Missed:', sum(next_state.state.glob.undetected))



# ========================================
# Training
# ========================================

gamma = 0.95
steps = 2 * n_agents
epsilon = 1.0
epochs = 200000
EPSRATE = 10000

seq_state = Seq_State()
len(seq_state.observation)
input_length = len(seq_state.observation)
l1 = input_length
#l2 = 300
#l3 = 100
l2 = 100
l3 = 50
l4 = LOCAL_ACTIONS_NR

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)
loss_fn = torch.nn.MSELoss()

learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(epochs):

    if epoch % 1000 == 0:
        print('Epoch: ', epoch)
        simulate_step(model)
        
    current_state = Seq_State()
    observation_ = current_state.observation.reshape(1,input_length) + np.random.rand(1,input_length)/1000.0
    observation1 = torch.from_numpy(observation_).float()
    
    for step in range(steps):
            
        qval = model(observation1)
        qval_ = qval.data.numpy()
    
        if random.random() < epsilon:
            action_ = random.choice(tuple(current_state.actions))
        else:
            #action_ = np.argmax(qval_)
            filtered_qvals = [ qval_[0][act] if act in current_state.actions \
                              else -float('Inf') for act in range(LOCAL_ACTIONS_NR) ]
            action_ = np.argmax(filtered_qvals)
    
        action = Ctrl_Action(action_)
        
        reward = seq_reward_ctrl(current_state, action)
        reward_env = 0
    
        current_state = seq_delta_ctrl(current_state, action)
    
        #print('is instance? ', isinstance(current_state, State))
    
        if isinstance(current_state, State):
                    
            next_state = seq_delta_env(current_state, Env_Action.skip)
            reward_env = seq_reward_env(current_state, Env_Action.skip, next_state)
            current_state = next_state
        
        reward = reward + reward_env
            
        observation2_ = current_state.observation.reshape(1,input_length) + np.random.rand(1,input_length)/1000.0
        observation2 = torch.from_numpy(observation2_).float()
        
        with torch.no_grad():
            newQ = model(observation2)
            newQ_ = newQ.data.numpy()
            filtered_qvals = [ newQ_[0][act] if act in current_state.actions else -float('Inf') for act in range(LOCAL_ACTIONS_NR) ]
        maxQ = np.max(filtered_qvals)
            
        Y = reward + (gamma * maxQ)
    
        Y = torch.Tensor([Y]).detach().squeeze()
        X = qval.squeeze()[action_]
        loss = loss_fn(X, Y)
    
        optimizer.zero_grad()
        loss.backward()
        #losses.append(loss.item())
        optimizer.step()
    
        observation1 = observation2
    
        if epsilon > 0.1:
            epsilon -= (1/EPSRATE)


# ========================================
# Simulation
# ========================================

current_state = Seq_State()
print(current_state)


act = [None] * n_agents

for i in range(n_agents):

    observation_ = current_state.observation.reshape(1,input_length) + np.random.rand(1,input_length)/1000.0
    observation1 = torch.from_numpy(observation_).float()
        
    qval = model(observation1)
    qval_ = qval.data.numpy()

    filtered_qvals = [ qval_[0][act] if act in current_state.actions \
                          else -float('Inf') for act in range(LOCAL_ACTIONS_NR) ]
    action_ = np.argmax(filtered_qvals)

    action = Ctrl_Action(action_)
    
    print('>>>>>>>>>> Local action:', action)
    act[current_state.current_agent] = action.value
    
    current_state = seq_delta_ctrl(current_state, action)
    
    print(current_state)
    
print(act)

next_state = seq_delta_env(current_state, Env_Action.skip)
print(next_state)
print('Missed:', sum(next_state.state.glob.undetected))


#torch.save(model, '/Users/bollig/Git/covid19/station_simulation/ML/model-20-05.pt')
#model = torch.load('/Users/bollig/Git/covid19/station_simulation/ML/model-15-05.pt')






