#!/usr/bin/env python3

from enum import Enum, auto
import random
import networkx as nx

import config

from Infections.Agent import Agent, Category


class GraphType(Enum):
    AGENT_SAMPLES = auto()
    ERDOS_RENYI = auto() 
    REGULAR = auto()
    LYON = auto()


class Hospital:

    def __init__(self):
        pass

    def get_contacts(self, agent, time):
        return []


class SingleWard(Hospital):

    def __init__(self, infex, gtype=GraphType.LYON, gparam={'contacts_per_agent': 4, 'min_per_contact': 15}):
        self.graphs = []
        self.working = []
        self.infex = infex
        self.gtype = gtype
        self.gparam = gparam

        if gtype is GraphType.LYON:
            self.kernel = dict()
            for agent in self.infex.agents.values():
                self.generate_kernel(agent)
        else:
            self.contacts_per_agent = self.gparam['contacts_per_agent']
            self.min_per_contact = self.gparam['min_per_contact']

    def generate_kernel(self, agent):
        self.kernel[agent] = set()
        if agent.category in [Category.NURSE, Category.DOCTOR]: 
            for cat in Category:
                ker_size = config.kerneldist[agent.category][cat]()
                pop = [a for a in self.infex.agents.values() if a.category is cat]
                random.shuffle(pop)
                self.kernel[agent] |= set(pop[0:min(ker_size,len(pop))])

    def generate_contacts(self, agent):
        ret = set()

        if agent.category in [Category.NURSE, Category.DOCTOR]:
            for cat in Category:
                contact_size = config.contactsdist[agent.category][cat]()
                #contact_size = len(self.infex.agents)
                if contact_size < len(self.kernel[agent]):
                    pop = [a for a in self.kernel[agent]]
                    random.shuffle(pop)
                    ret |= set(pop[0:contact_size])
                else:
                    ret |= self.kernel[agent]
                    pop = [a for a in self.infex.agents.values() if a.category is cat and a not in self.kernel[agent]]
                    random.shuffle(pop)
                    nb_additional = min(len(pop), contact_size - len(self.kernel[agent]))
                    ret |= set(pop[0:nb_additional])
                
        return ret

    def define_working(self, time, working):
        """
        Define the set of working personnel for a given time.
        """
        current_max_time = len(self.working)

        for _ in range(current_max_time, time):
            self.working += [[]]
        self.working[time-1] = working

    def generate_graph(self, time: int):
        if self.gtype is GraphType.AGENT_SAMPLES:
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
                    graph.add_edge(i, j, weight=self.min_per_contact)
                
            return graph
        elif self.gtype is GraphType.ERDOS_RENYI:
            raise NotImplementedError()
        elif self.gtype is GraphType.REGULAR:
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
                    graph.add_edge(self.working[time-1][j], self.working[time-1][k], weight=self.min_per_contact)
            return graph
        elif self.gtype is GraphType.LYON:
            pop = self.working[time-1]
            graph = nx.Graph()
            for i in self.working[time-1]:
                graph.add_node(i)
                agent = self.infex.agents[i]

                contacts = self.generate_contacts(agent)

                for agent2 in contacts:
                    duration = config.contact_length_dist()
                    graph.add_edge(i, agent2.id, weight=duration)
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
            graph = self.generate_graph(time)
            self.graphs += [graph]

    def get_contacts(self, time: int, agent: int):
        """
        Returns the list of contacts of the specified agent at the specified
        time.
        """
        self.generate_graphs(time)

        if not self.graphs[time-1].has_node(agent):
            return {}

        ret = dict()
        for other in self.graphs[time-1].neighbors(agent):
            ret[other] = self.graphs[time-1][agent][other]['weight']

        return ret