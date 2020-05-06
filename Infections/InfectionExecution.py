#!/usr/bin/env python3

import random
from matplotlib import collections as mc

import config

from Infections.Agent import State
from Infections.Hospital import SingleWard


class InfectionExecution(object):
    def __init__(self, N):
        self.infections = {'internal': [],
                           'external': []}
        self.symptomatic = []
        self.contacted = dict()
        self.hospital = SingleWard()

    def add_ext(self, t, id1):
        self.infections['external'] += [(t, id1)]

    def add_inf(self, t1, t2, id1, id2):
        self.infections['internal'] += [(t1, t2, id1, id2)]

    def add_symptomatic(self, t, id1):
        assert ((t, id1, True) not in self.symptomatic)
        self.symptomatic += [(t, id1, True)]

    def remove_symptomatic(self, t, id1):
        assert ((t, id1, False) not in self.symptomatic)
        self.symptomatic += [(t, id1, False)]

    def get_symptomatic(self, t):
        sym_leq_t = set([sym[1]
                         for sym in self.symptomatic if sym[0] <= t and sym[2]])
        healed_leq_t = set(
            [sym[1] for sym in self.symptomatic if sym[0] <= t and not sym[2]])
        return list(sym_leq_t - healed_leq_t)

    def add_contacted(self, t, agent, contact_ids):
        # print(f'add_contacted({t}, id {agent.id}, {contact_ids})')
        i = agent.id
        if i in self.contacted.keys():
            self.contacted[i] += [(t, contact_ids)]
        else:
            self.contacted[i] = [(t, contact_ids)]

    def get_contacts_attime(self, myid, t):
        cs = []
        if myid in self.contacted.keys():
            # get those after at time t
            graphs = list(filter(lambda c: c[0] == t,
                                 self.contacted[myid]))
            # print(graphs)
            assert (len(graphs) <= 1)
            for t, contact_ids in graphs:
                cs += contact_ids
        return cs

    def get_contactgraph_attime(self, t):
        graph = {}
        for i in range(config.N):
            graph[i] = self.get_contacts_attime(myid=i, t=t)
        return graph

    def get_contacts_fromtime(self, myid, t_from):
        cs = []
        if myid in self.contacted.keys():
            # get those after including t_from
            graphs = list(filter(lambda c: c[0] >= t_from,
                                 self.contacted[myid]))
            # print(graphs)
            for t, contact_ids in graphs:
                cs += contact_ids
        return cs

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
        # from onset of symptoms to onset of symptoms along causal chains
        # at the moment only for severe cases - not mild symptomatic cases
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

    def plotparam(self, myax, param, alpha=0.1):
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

    def next_state(self, t, myid, state):
        s = state[myid]
        # today contacted
        if s.quarantined:
            # no (internal) contacts in quarantine
            # can still be infected externally in quarantine
            today_contacts = []
        else:
            # today working
            workforce = list(
                filter(lambda p: state[p].works(), config.person_ids))
            self.hospital.define_working(t, workforce)
            today_contacts = self.hospital.get_contacts(t, myid)
            # today_contacts = random.sample(workforce, min(
            #     len(workforce), config.contacts_perday))
        # log contacts
        self.add_contacted(t=t, agent=s, contact_ids=today_contacts)

        # schedule a day
        s.tick(t)

        # update who got infected
        if s.state is State.UNINFECTED:
            # get externally infected
            u = random.random()
            if u < config.pext:
                self.add_ext(t=t, id1=myid)
                s.infect(None, t)

            # get internally infected
            for so in today_contacts:
                if state[so].state is State.INFECTED_SPREADING:
                    u = random.random()
                    if u < config.pint:
                        self.add_inf(state[so].time_infected, t, so, myid)
                        s.infect(state[so].id, t)
