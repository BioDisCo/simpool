#!/usr/bin/env python3

from enum import Enum, auto
import random
import networkx as nx

import config

from Infections.Agent import Agent


class GraphType(Enum):
    AGENT_SAMPLES = auto()
    ERDOS_RENYI = auto() 
    REGULAR = auto()


class Hospital:

    def __init__(self):
        pass

    def get_contacts(self, agent, time):
        return []


class SingleWard(Hospital):

    def __init__(self, contacts_per_agent=4):
        self.graphs = []
        self.working = []
        self.contacts_per_agent = contacts_per_agent

    def define_working(self, time, working):
        """
        Define the set of working personnel for a given time.
        """
        current_max_time = len(self.working)

        for _ in range(current_max_time, time):
            self.working += [[]]
        self.working[time-1] = working

    def generate_graph(self, time: int, gtype=GraphType.AGENT_SAMPLES):
        if gtype is GraphType.AGENT_SAMPLES:
            pop = self.working[time-1]
            graph = nx.Graph()
            for i in self.working[time-1]:
                graph.add_node(i)

                random.shuffle(pop)
                contacts = pop[0:min(4,len(pop))]

                # don't include self-loops
                if i in contacts and len(pop) > 4:
                    contacts.remove(i)
                    contacts += [pop[4]]

                for j in contacts:
                    graph.add_edge(i, j)
                
            return graph
        elif gtype is GraphType.ERDOS_RENYI:
            raise NotImplementedError()
        elif gtype is GraphType.REGULAR:
            #Returns a random d-regular graph on the set of agent indices.
            n = len(self.working[time-1])
            d = min(self.contacts_per_agent, n-1)

            if n == 0:
                return nx.complete_graph(0)
            #print(f'd = {d}, n = {n}')
            if d >= n:
                return nx.complete_graph(n)

            working_graph = nx.random_regular_graph(d, n)

            graph = nx.Graph()
            for i in config.person_ids:
                graph.add_node(i)
            for j in range(n):
                for k in working_graph.neighbors(j):
                    graph.add_edge(self.working[time-1][j], self.working[time-1][k])
            return graph
        else:
            raise ValueError('Graph type not supported.')

    def generate_graphs(self, time, gtype=GraphType.AGENT_SAMPLES):
        """
        Generates and stores contact graphs up to the specified time, as
        necessary.
        """
        current_max_time = len(self.graphs)

        if len(self.working) < current_max_time:
            raise ValueError('Not enough working sets specified')

        for _ in range(current_max_time, time):
            graph = self.generate_graph(time, gtype)
            self.graphs += [graph]

    def get_contacts(self, time, agent):
        """
        Returns the list of contacts of the specified agent at the specified
        time.
        """
        self.generate_graphs(time)

        if not self.graphs[time-1].has_node(agent):
            return []

        return list(self.graphs[time-1].neighbors(agent))
