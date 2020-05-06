#!/usr/bin/env python3

import random
import networkx as nx

import config

from Infections.Agent import Agent


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

        for _ in range(current_max_time, time+1):
            self.working += [[]]
        self.working[time-1] = working

    def generate_graph(self, time):
        """
        Returns a random d-regular graph on the set of agent indices.
        """
        n = len(self.working[time-1])
        d = min(self.contacts_per_agent, n-1)
        working_graph = nx.random_regular_graph(d, n)

        graph = nx.Graph()
        for i in config.person_ids:
            graph.add_node(i)
        for j in range(n):
            for k in working_graph.neighbors(j):
                graph.add_edge(self.working[time-1][j], self.working[time-1][k])
        return graph

    def generate_graphs(self, time):
        """
        Generates and stores contact graphs up to the specified time, as
        necessary.
        """
        current_max_time = len(self.graphs)

        if len(self.working) < current_max_time:
            raise ValueError('Not enough working sets specified')

        for _ in range(current_max_time, time+1):
            graph = self.generate_graph(time)
            self.graphs += [graph]

    def get_contacts(self, time, agent):
        """
        Returns the list of contacts of the specified agent at the specified
        time.
        """
        self.generate_graphs(time)
        return list(self.graphs[time-1].neighbors(agent))
