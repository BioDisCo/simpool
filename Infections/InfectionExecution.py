#!/usr/bin/env python3

import random
from matplotlib import collections as mc

import config

from Infections.Agent import Symptoms, Agent
from Infections.Hospital import SingleWard


class InfectionExecution(object):
    def __init__(self):
        self.infections = {'internal': [],
                           'external': []}
        self.contacted = dict()
        self.hospital = SingleWard()
        self.agents = {i: Agent(i, self) for i in config.person_ids}

        init_states = {}
        for i in config.person_ids:
            init_states[i] = self.agents[i].state
        self.states : list = [init_states]

        self.t : int = 0

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

    def log_contacts(self, t, agent_id, contact_ids):
        if agent_id in self.contacted.keys():
            self.contacted[agent_id] += [(t, contact_ids)]
        else:
            self.contacted[agent_id] = [(t, contact_ids)]

    def get_contacts_attime(self, agent_id, t):
        cs = []
        if agent_id in self.contacted.keys():
            # get those at time t
            graphs = list(filter(lambda c: c[0] == t,
                                 self.contacted[agent_id]))
            # print(graphs)
            assert (len(graphs) <= 1)
            for t, contact_ids in graphs:
                cs += contact_ids
        return cs

    def get_contactgraph_attime(self, t):
        graph = {}
        for i in config.person_ids:
            graph[i] = self.get_contacts_attime(agent_id=i, t=t)
        return graph

    def get_contacts_fromtime(self, myid, t_from):
        cs = []
        if myid in self.contacted.keys():
            # get those after including t_from
            graphs = list(filter(lambda c: c[0] >= t_from,
                                 self.contacted[myid]))
            # print(graphs)
            for _, contact_ids in graphs:
                cs += contact_ids
        return cs

    def get_nb_working(self, t):
        return len([i for i, state in self.states[t].items() if state['working']])

    def get_nb_quarantined(self, t):
        return len([i for i, state in self.states[t].items() if state['quarantined']])

    def get_nb_working_spreading(self, t):
        return len([i for i, state in self.states[t].items() if state['working'] and state['spreading']])

    def get_nb_dead(self, t):
        return len([i for i, state in self.states[t].items() if state['dead']])

    def get_nb_sick_leave(self, t):
        return len([i for i, state in self.states[t].items() if state['sick_leave']])

    def get_nb_infected(self, t):
        return len([i for i, state in self.states[t].items() if state['infected']])

    def get_nb_working_immune(self, t):
        return len([i for i, state in self.states[t].items() if state['working'] and state['infected']])

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

    def tick(self):
        self.t += 1
        new_states = {}
        for i in config.person_ids:
            new_states[i] = self.agents[i].state
        self.states += [new_states]

        for i, agent in self.agents.items():
            # get agents contacted today
            if agent.quarantined:
                # no (internal) contacts in quarantine
                # can still be infected externally in quarantine
                today_contacts = []
            else:
                # is working today
                workforce = list(
                    filter(lambda p: self.agents[p].working, config.person_ids))
                self.hospital.define_working(self.t, workforce)
                today_contacts = self.hospital.get_contacts(self.t, i)

            # log contacts
            self.log_contacts(t=self.t, agent_id=i, contact_ids=today_contacts)

            # schedule a day
            agent.tick()

            # update who got infected
            if not agent.infected:
                # get externally infected
                u = random.random()
                if u < config.pext:
                    self.log_external_infection(t=self.t, id1=i)
                    agent.infect(None, self.t)

                # get internally infected
                for so in today_contacts:
                    if self.agents[so].spreading:
                        u = random.random()
                        if u < config.pint:
                            self.log_internal_infection(self.agents[so].time_infected, self.t, so, i)
                            agent.infect(self.agents[so].id, self.t)
