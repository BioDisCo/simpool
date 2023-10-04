#!/usr/bin/env python3

import random
from matplotlib import collections as mc

import config

from Infections.Agent import Symptoms, Agent, Category
from Infections.Hospital import SingleWard


class InfectionExecution(object):
    def __init__(self, initial_infections={'T_range': (0,10), 'N_range': (0,1)}):
        """
        Create an InfectionExecution with initial infections:
        <N_range> infections (uniformly distributed)
          during days <T_range> (uniformly distributed) 
        """
        self.infections = {'internal': [],
                           'external': []}
        self.contacted = dict()
        self.testresult = dict()
        self.agents = dict()
        myid = 0
        for cat in config.nb_agents.keys():
            num = config.nb_agents[cat]
            for i in range(num):
                self.agents[myid+i] = Agent(myid+i, cat, self)
            myid += num
        self.hospital = SingleWard(self)
        self.states : list = [] #self.get_state_vec()
        self.t : int = 0

        # --- initial infection model ------
        # model: infect agents
        self.scheduled_infections = []
        Num_to_infect = random.choice(range(
            min(config.N,initial_infections['N_range'][0]),
            min(config.N,initial_infections['N_range'][1])+1 ))
        # choose ids (without repetition) from population
        myids = random.sample(range(config.N), Num_to_infect)
        for i in myids:
            t = int( random.uniform(initial_infections['T_range'][0], initial_infections['T_range'][1]) )
            self.scheduled_infections += [ (i,t) ]
        #print(self.scheduled_infections)
        # --- end initial infection model ---

    def log_external_infection(self, t, id1):
        self.infections['external'] += [(t, id1)]

    def log_internal_infection(self, t1, t2, id1, id2):
        self.infections['internal'] += [(t1, t2, id1, id2)]

    """
    def get_symptomatic(self, t, symptoms):
        sym_leq_t = set([sym[1]
                         for sym in self.symptomatic if sym[0] <= t and sym[2]])
        healed_leq_t = set(
            [sym[1] for sym in self.symptomatic if sym[0] <= t and not sym[2]])
        return list(sym_leq_t - healed_leq_t)
    """

    def log_contacts(self, t: int, agent_id: int, contacts):
        if agent_id in self.contacted.keys():
            self.contacted[agent_id] += [{'time': t, 'contacts': contacts}]
        else:
            self.contacted[agent_id] = [{'time': t, 'contacts': contacts}]

    #def log_testresult(self, t: int, agent_id: int, test_result):
    #    if agent_id in self.testresult.keys():
    #        self.testresult[agent_id] += [{'time': t, 'test_result': test_result}]
    #    else:
    #        self.testresult[agent_id] = [{'time': t, 'test_result': test_result}]

    def get_state_vec(self):
        state = {
            agent_id: agent.state for agent_id, agent in self.agents.items()
        }
        return [ state ]

    def get_contacts_attime(self, agent_id, t, duration_threshold=config.DURATION_TH):
        if agent_id in self.contacted.keys():
            contacts_at_time = [c['contacts'] for c in self.contacted[agent_id] if c['time'] == t]
            assert(len(contacts_at_time) <= 1)
            if contacts_at_time:
                thresholded_contacts = [other for other in contacts_at_time[0].keys() if contacts_at_time[0][other] >= duration_threshold]
                return thresholded_contacts
            else:
                return []
        else:
            return []

    def get_contacts_attime_withduration(self, agent_id, t):
        if agent_id in self.contacted.keys():
            contacts_at_time = [c['contacts'] for c in self.contacted[agent_id] if c['time'] == t]
            assert(len(contacts_at_time) <= 1)
            if contacts_at_time:
                contacts_withduration = {other: contacts_at_time[0][other] for other in contacts_at_time[0].keys()}
                return contacts_withduration
            else:
                return {}
        else:
            return {}


    def get_contactgraph_attime(self, t):
        graph = {}
        for i in config.person_ids:
            graph[i] = self.get_contacts_attime(agent_id=i, t=t)
        return graph

    def get_contactgraph_attime_withduration(self, t):
        graph = {}
        for i in config.person_ids:
            graph[i] = self.get_contacts_attime_withduration(agent_id=i, t=t)
        return graph

    def get_contacts_fromtime(self, agent_id, t_from, duration_threshold=config.DURATION_TH):
        ret = set()

        for t in range(t_from, self.t+1):
            ret |= set(self.get_contacts_attime(agent_id, t, duration_threshold))

        return list(ret)

    def get_nb_staff(self, t):
        return len([i for i, state in self.states[t].items() if state['staff']])

    def get_nb_working(self, t):
        return len([i for i, state in self.states[t].items() if state['working'] and state['staff']])

    def get_nb_quarantined(self, t):
        return len([i for i, state in self.states[t].items() if state['quarantined'] and state['staff']])

    def get_nb_working_spreading(self, t):
        return len([i for i, state in self.states[t].items() if state['working'] and state['spreading'] and state['staff']])

    def get_nb_dead(self, t):
        return len([i for i, state in self.states[t].items() if state['dead'] and state['staff']])

    def get_nb_sick_leave(self, t):
        return len([i for i, state in self.states[t].items() if state['sick_leave'] and state['staff']])

    def get_nb_infected(self, t):
        return len([i for i, state in self.states[t].items() if state['infected'] and state['staff']])

    def get_nb_working_immune(self, t):
        return len([i for i, state in self.states[t].items() if state['working'] and state['infected'] and state['staff']])

    def plot(self, myax, cmap='Blues', alpha=0.2):
        lines = []
        for inf in self.infections['internal']:
            lines += [[(inf[0], inf[2]), (inf[1], inf[3])]]
        for inf in self.infections['external']:
            lines += [[(inf[0], inf[1]), (inf[0]+1, inf[1])]]

        lc = mc.LineCollection(lines, linewidths=1, alpha=alpha, cmap=cmap)
        myax.add_collection(lc)
        myax.autoscale()

    def get_infections_per_infected(self):
        # to compute R
        was_infecting = set()
        for inf in self.infections['internal']:
            was_infecting.add(inf[2])  # source
            was_infecting.add(inf[3])  # sink
        for inf in self.infections['external']:
            was_infecting.add(inf[1])  # sink
        was_infecting = list(was_infecting)

        infected_nums = []
        for myid in was_infecting:
            li = list(
                filter(lambda inf: inf[2] == myid, self.infections['internal']))
            infected_nums += [len(li)]

        return infected_nums

    def get_serial_intervals(self):
        #TODO: adapt to new implementation
        return []

        # from onset of symptoms to onset of symptoms along causal chains
        # at the moment only for severe cases - not mild symptomatic cases
        """
        serial_intervals = []

        for inf in self.infections['internal']:
            source = inf[2]
            sink = inf[3]
            source_symptomatic_onset = list(
                filter(lambda sym: sym[1] == source and sym[2], self.symptomatic))
            sink_symptomatic_onset = list(
                filter(lambda sym: sym[1] == sink and sym[2],   self.symptomatic))
            if source_symptomatic_onset and sink_symptomatic_onset:
                serial_intervals += [sink_symptomatic_onset[0]
                                     [0] - source_symptomatic_onset[0][0]]

        return serial_intervals
        """

    def plotparam(self, myax, param: str, alpha=0.1):
        if param == 'R':
            infected_nums = self.get_infections_per_infected()
            myax.hist(x=infected_nums, bins=list(range(10)),
                      color='#0504aa', alpha=alpha, rwidth=1.0, align='left')
            myax.set_xlabel('new infections per infected')
        elif param == 'serial':
            serial_intervals = self.get_serial_intervals()
            myax.hist(x=serial_intervals, bins=list(range(-10, 10, 1)),
                      color='#0504aa', alpha=alpha, rwidth=1.00, align='mid')
            myax.set_xlabel('serial interval lengh (only severe cases) [days]')
        else:
            assert(False)

    def tick_endofworkingday(self):
        """
        A day is over and the results are saved
        """
        #print(f'End of day {self.t}.')
        # append the day
        self.states += self.get_state_vec()

    def tick_nextday(self):
        """
        Morning of the next day.
        """
        # next day
        self.t += 1
        #print(f'Morning of day {self.t}')

    def tick_startworkingday(self):
        """
        Starts a new working day.
        """
        #print(f'Start working day {self.t}')
        for i, agent in self.agents.items():
            # get agents contacted today
            if agent.quarantined:
                #print(f'quarantined agent: {agent.id}')
                # no (internal) contacts in quarantine
                # can still be infected externally in quarantine
                today_contacts = {}
            else:
                # is working today
                workforce = list(
                    filter(lambda p: self.agents[p].working, config.person_ids))
                self.hospital.define_working(self.t, workforce)
                today_contacts = self.hospital.get_contacts(self.t, i)

            # log contacts
            self.log_contacts(t=self.t, agent_id=i, contacts=today_contacts)

            # schedule a day
            agent.tick()

            # update who gets infected
            if not agent.infected:
                if (i,self.t) in self.scheduled_infections:
                    # prescheduled infection
                    self.log_external_infection(t=self.t, id1=i)
                    self.agents[i].infect(by_id=None, time=self.t)
                else:
                    u = random.random()
                    if u < config.pext:
                        # get externally infected
                        self.log_external_infection(t=self.t, id1=i)
                        agent.infect(None, self.t)
                    else:
                        # get internally infected
                        for so in today_contacts.keys():
                            duration = today_contacts[so]
                            if self.agents[so].spreading:
                                u = random.random()
                                if u < config.pinf(duration):
                                    self.log_internal_infection(self.agents[so].time_infected, self.t, so, i)
                                    agent.infect(self.agents[so].id, self.t)
