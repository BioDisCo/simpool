import numpy as np
import enum
import random
import torch
from matplotlib import pylab as plt
from copy import deepcopy, copy

EPOCHS = 1000000
LOCAL_ACTIONS_NR = 3
N_AGENTS = 3
N_ROUNDS = 7
CAPACITY = 3
PENALTY = -100

class Action(enum.Enum):
    skip = 0
    buy = 1
    sell = 2

    def __str__(self):
        return str(self.name)


class State:
    
    def __init__(self, n_agents, capacity):
        self.n_agents = n_agents
        self.round = 1
        self.permutation = random.sample(list(range(n_agents)), n_agents)
        self.index = 0
        self.seen = set()
        # game specific
        self.holding = [False] * n_agents
        self.values = [1] * n_agents
        self.last_buy_cost = [0] * n_agents
        self.capacity = capacity
        self.last_action = [None] * n_agents


    @property
    def actions(self):
        allowed = [Action.skip.value]

        if self.holding[self.agent]:
            allowed += [Action.sell.value]

        else:
            sell_list = [self.last_action[i] == Action.sell for i in range(self.n_agents)]
            if self.capacity + np.dot(sell_list, self.values) >= self.values[self.agent]:
                allowed += [Action.buy.value]
                
        allowed = set(allowed)

        # to check:
        # CHANGE HERE
        allowed = set([0,1,2])

        return allowed


    @property
    def agent(self):
        return self.permutation[self.index]

    @property
    def old_observation(self):
        binary_index = [1 if i == self.agent else 0 for i in range(self.n_agents)]
        binary_seen = [1 if i in self.seen else 0 for i in range(self.n_agents)]
        # game specific binary
        binary_holding = [1 if self.holding[i] else 0 for i in range(self.n_agents)]
        # return
        return np.array(binary_index + binary_seen +
            binary_holding +
            self.last_buy_cost +
            self.values +
            [self.capacity])
    

    @property
    def observation(self):
        binary_seen = [1 if i in self.seen else 0 for i in range(self.n_agents)]
        binary_seen[0], binary_seen[self.agent] = binary_seen[self.agent], binary_seen[0]
        
        # game specific binary
        binary_holding = [1 if self.holding[i] else 0 for i in range(self.n_agents)]
        binary_holding[0], binary_holding[self.agent] = binary_holding[self.agent], binary_holding[0]
        
        last_cost = self.last_buy_cost.copy()
        last_cost[0], last_cost[self.agent] = last_cost[self.agent], last_cost[0]
        
        all_list = list(zip(binary_seen, binary_holding, last_cost))
        current_tuple = list(all_list[0])
        rest = sorted(all_list[1:])
        rest = list(sum(rest, ()))
        
        return np.array(current_tuple + rest + self.values + [self.capacity])
        
    
        # return
        #return np.array([self.round] +
        #    binary_seen +
        #    binary_holding +
        #    last_cost +
        #    self.values +
        #    [self.capacity])


    def __str__(self):
        return \
f"""Round: {self.round} ({self.permutation})
  Seen: {self.seen}
  Holding={self.holding}
  Last buy={self.last_buy_cost}
  Values={self.values}
  Actions={self.actions}
  ## {self.capacity} ##"""


class Game:
    
    def __init__(self, n_agents = N_AGENTS, n_rounds = N_ROUNDS, capacity = CAPACITY, penalty = PENALTY):
        self.n_agents = n_agents
        self.n_rounds = n_rounds
        self.penalty = penalty
        self.state = State(n_agents, capacity)
        self.initial_capacity = capacity


    def new_state(self, action: Action):
        newState = deepcopy(self.state)
        current_agent = self.state.agent
        newState.last_action[current_agent] = action

        if action == Action.buy:
            newState.holding[current_agent] = True
            newState.last_buy_cost[current_agent] = self.state.values[current_agent]
            newState.capacity -= self.state.values[current_agent]

        elif action == Action.sell:
            newState.holding[current_agent] = False
            newState.last_buy_cost[current_agent] = 0 # does not need to be

        elif action == Action.skip:
            pass

        else:
            assert False, "unknown action"

        # end of agent action
        if self.state.index < self.n_agents - 1:
            newState.seen.add(current_agent)
            newState.index = self.state.index + 1

        else:
            # make sells
            sell_list = [newState.last_action[i] == Action.sell for i in range(self.n_agents)]
            newState.capacity += np.dot(sell_list, self.state.values)

            # new round
            newState.values = [1 + self.state.round] * self.n_agents
            newState.permutation = random.sample(list(range(self.n_agents)), self.n_agents)
            newState.index = 0
            newState.seen = set()
            newState.round += 1
            newState.last_action = [None] * self.n_agents

        return newState


    def apply_action(self, action: Action):
        newState = self.new_state(action=action)
        self.state = newState
    

    @property
    def running(self):
        return self.state.round < self.n_rounds
                
    def reward(self, action: Action):
        current_agent = self.state.agent
        asset = self.state.capacity + np.dot(self.state.holding, self.state.values)        
        next_state = self.new_state(action)
        next_asset = next_state.capacity + np.dot(next_state.holding, next_state.values)        
         
        if next_state.capacity < 0:
            #assert False, "this shoud not happen"
            return self.penalty
           
        if action == Action.buy and self.state.holding[current_agent]:
            #assert False, "this shoud not happen"
            return self.penalty

        elif action == Action.sell and not self.state.holding[current_agent]:
            #assert False, "this shoud not happen"
            return self.penalty
                                                
        if self.state.round == self.n_rounds - 1 and self.state.index == game.n_agents - 1:
            return (next_state.capacity - self.initial_capacity) * 100

        return next_asset - asset


game = Game(n_agents = N_AGENTS, n_rounds = N_ROUNDS, capacity = CAPACITY, penalty = PENALTY)

input_length = len(game.state.observation)
l1 = input_length
l2 = 64
l3 = 32
l4 = LOCAL_ACTIONS_NR

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
)
loss_fn = torch.nn.MSELoss()

gamma = 0.9
epsilon = 1.0
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# try trained game : short version

def simulate_game(use_best_model=False):
    game = Game(n_agents = N_AGENTS, n_rounds = N_ROUNDS, capacity = CAPACITY, penalty = PENALTY)
    
    action_list = [''] * game.state.n_agents
    reward_list = [0] * game.state.n_agents
    
    print('--------')
    print(game.state)
    while(game.running):
        
        #if game.state.index == 0:
        #    action_list = [''] * game.state.n_agents
        #    reward_list = [0] * game.state.n_agents
    
        observation_ = game.state.observation.reshape(1,input_length)
        observation = torch.from_numpy(observation_).float()
    
        if use_best_model:
            qval = best_model(observation)
        else:
            qval = model(observation)
        qval_ = qval.data.numpy()
        filtered_qvals = [ qval_[0][act] if act in game.state.actions else -float('Inf') for act in range(LOCAL_ACTIONS_NR) ]

        action_ = np.argmax(filtered_qvals)
        action = Action(action_)
        
        action_list[game.state.agent] = action.name
        reward_list[game.state.agent] = game.reward(action)
        
        if game.state.index == game.n_agents - 1:
            print('========================================')
            print('>>>>>>>>>>', action_list)
            print('>>>>>>>>>>', reward_list)
            print('========================================')
    
        game.apply_action(action)
        
        if game.state.index == 0:
            print(game.state)


epochs = EPOCHS
losses = []
best_capacity = 0
best_model = None
for i in range(epochs):
        
    game = Game(n_agents = N_AGENTS, n_rounds = N_ROUNDS, capacity = CAPACITY, penalty = PENALTY)

    observation_ = game.state.observation.reshape(1,input_length) + np.random.rand(1,input_length)/1000.0 #D
    observation1 = torch.from_numpy(observation_).float()
    
    while(game.running):
        qval = model(observation1)
        qval_ = qval.data.numpy()
        
        if random.random() < epsilon:
            action_ = random.choice(tuple(game.state.actions))
        else:
            action_ = np.argmax(qval_)
            #filtered_qvals = [ qval_[0][act] if act in game.state.actions else -float('Inf') for act in range(LOCAL_ACTIONS_NR) ]
            #action_ = np.argmax(filtered_qvals)
        
        action = Action(action_)
        reward = game.reward(action)
        game.apply_action(action)
        observation2_ = game.state.observation.reshape(1,input_length) + np.random.rand(1,input_length)/1000.0
        observation2 = torch.from_numpy(observation2_).float()
        
        with torch.no_grad():
            newQ = model(observation2)
        maxQ = torch.max(newQ)
        if game.running:
            Y = reward + (gamma * maxQ)
        else:
            Y = reward

        Y = torch.Tensor([Y]).detach().squeeze()
        X = qval.squeeze()[action_]
        loss = loss_fn(X, Y)


        optimizer.zero_grad()
        loss.backward()
        #losses.append(loss.item())
        optimizer.step()
        observation1 = observation2

    if epsilon > 0.1:
        epsilon -= (1/epochs)
        
    if game.state.capacity >= best_capacity:
        best_capacity = game.state.capacity
        best_model = model
    
    if i % 1000 == 0:
        print("Epoch:", i, "; Capacity:", game.state.capacity, "; Best capacity:", best_capacity)
        
    if i > 0 and i % 5000 == 0:
        simulate_game(use_best_model=True)


simulate_game(use_best_model=True)


#torch.save(model.state_dict(), '/Users/bollig/Git/covid19/station_simulation/ML/model.pth')

#torch.save(model, '/Users/bollig/Git/covid19/station_simulation/ML/model.pt')

#model = torch.load('/Users/bollig/Git/covid19/station_simulation/ML/model.pt')

