import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import csv
import collections
import math
from pprint import pprint
import random

from statistics import *

#col = {'MED': 1, 'ADM': 20, 'NUR': 40, 'PAT': 60}
col = {'MN': 1, 'AP': 60}
col = {'MED': 1, 'ADM': 20, 'NUR': 40, 'PAT': 60}
in_group = {'MED': 'MN', 'ADM': 'AP', 'NUR': 'MN', 'PAT': 'AP'}
in_group = {'MED': 'MED', 'ADM': 'ADM', 'NUR': 'NUR', 'PAT': 'PAT'}
cmap = matplotlib.cm.get_cmap('Spectral')
CONTACT_THRESHOLD = 5*60/20 # 15*60/20  # 15min in terms of 20s
MAX_DEGREE = 50 # check


def plot_degree_hist(G):
    fig, ax = plt.subplots()

    node_degree_sequence = [(n,d) for n, d in G.degree()]
    x = range(MAX_DEGREE+1)
    y_old = [0 for i in x]
    for t in col.keys():
        # go over all types t
        # degree sequence for this type
        degree_sequence = sorted([nd[1] for nd in node_degree_sequence if agents[nd[0]] == t], reverse=True)
        degreeCount = collections.Counter(degree_sequence)
        y = [0 for i in x]
        for d,mycount in degreeCount.items():
            y[d] = mycount
        plt.bar(x, y, width=0.80, bottom=y_old, color=cmap(col[t]), label=t)
        y_old = [ y_old[i]+y[i] for i in range(len(y_old)) ]

    plt.legend()
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in x])
    ax.set_xticklabels(x)

def plot_graph(G, agents):
    plt.figure()
    node_color = [ col[agents[a]] for a in agents.keys() ]
    nx.draw_networkx(G, with_labels=True, node_size=200, node_color=node_color, cmap=cmap)
    plot_degree_hist(G)

def read_graphs_from_csv(filename='detailed_list_of_contacts_Hospital.dat'):


    agents = dict()
    interactions = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t', quotechar='#')
        for row in reader:
            time_sec = int(row[0])
            agent1 = int(row[1])
            agent2 = int(row[2])
            agent1_type = row[3]
            agent2_type = row[4]

            # log types
            agents[agent1] = in_group[agent1_type]
            agents[agent2] = in_group[agent2_type]
            interactions += [ {'t': time_sec, 'a1': agent1, 'a2': agent2} ]


    # one graph per day
    graphs = []
    current_graph = nx.Graph()
    current_day_start = 0
    day_sec = 24*60*60
    weights = {}
    for a in agents.keys():
            current_graph.add_node(a)
    for interact in interactions:
        # check if still same day
        if interact['t'] < current_day_start + day_sec:
            # yes
            # add interactions to weights
            min_agent = min(interact['a1'], interact['a2'])
            max_agent = max(interact['a1'], interact['a2'])
            if (min_agent,max_agent) in weights.keys():
                weights[(min_agent,max_agent)] += 1
            else:
                weights[(min_agent,max_agent)] = 1
        else:
            # no
            # build graph
            for a1_a2 in weights.keys():
                if weights[a1_a2] >= CONTACT_THRESHOLD:
                    current_graph.add_edge(a1_a2[0], a1_a2[1], weight= weights[a1_a2])
            graphs += [ current_graph ]
            # new graph
            current_graph = nx.Graph()
            weights = {}
            for a in agents.keys():
                current_graph.add_node(a)
            # new time
            current_day_start += day_sec
            # add interaction
            min_agent = min(interact['a1'], interact['a2'])
            max_agent = max(interact['a1'], interact['a2'])
            if (min_agent,max_agent) in weights.keys():
                weights[(min_agent,max_agent)] += 1
            else:
                weights[(min_agent,max_agent)] = 1
    # finally build last graph
    # build graph
    for a1_a2 in weights.keys():
        if weights[a1_a2] >= CONTACT_THRESHOLD:
            current_graph.add_edge(a1_a2[0], a1_a2[1], weight= weights[a1_a2])
    graphs += [ current_graph ]

    return graphs, agents

def frequent_contacts(myid, graphs, agents, threshold=3, toCat=None):
    if toCat:
        toCats = [toCat]
    else:
        toCats = col.keys()

    ret = []
    for perid in graphs[0].nodes:
        cnt = sum([1 if perid in G.neighbors(myid) and agents[perid] in toCats else 0 for G in graphs])
        if cnt >= threshold:
            ret += [perid]
    return ret

#def contact_weights(myid, graphs):
#    ret = []
#    for perid in graphs[0].nodes:
#        weight = G[e[0]][e[1]]['weight']

def change_relativeto_kernel(myid, graphs, threshold=3):
    ret = []
    freq_contacts = frequent_contacts(myid, graphs, agents, threshold)
    for G in graphs:
        try:
            rel_G_minus_Freq = len(set(G.neighbors(myid)) - set(freq_contacts)) / len(freq_contacts)
            rel_Freq_minus_G = len(set(freq_contacts) - set(G.neighbors(myid))) / len(freq_contacts)
        except ZeroDivisionError:
            rel_G_minus_Freq = 'na'
            rel_Freq_minus_G = 'na'
        ret += [{'+': rel_G_minus_Freq, '-': rel_Freq_minus_G}]
    return ret

def average_nb_contacts(myid, graphs):
    return sum([len(list(G.neighbors(myid))) for G in graphs])/len(graphs)

def union_nb_contacts(graphs, agents, fromCat=None, toCat=None):
    if fromCat:
        fromCats = [fromCat]
    else:
        fromCats = col.keys()

    if toCat:
        toCats = [toCat]
    else:
        toCats = col.keys()

    return [len([otherid for otherid in G.neighbors(myid) if agents[otherid] in toCats]) for G in graphs for myid in graphs[0].nodes if agents[myid] in fromCats]

def union_min_contacts(graphs):
    ret = []
    for G in graphs:
        for e in G.edges:
            i = e[0]
            j = e[1]
            weight = G[i][j]['weight']
            mins = weight / (60/20)
            ret += [mins]
    return ret

def average_min_contacts(myid, graphs):
    return sum([G[myid][otherid]['weight']/3 for G in graphs for otherid in G.neighbors(myid)])/len(graphs)

def nb_contacts_matrix(graphs, agents):
    ret = {cat2: {cat1: 0 for cat1 in col.keys()} for cat2 in col.keys()}
    for G in graphs:
        for e in G.edges:
            cat1 = agents[e[0]]
            cat2 = agents[e[1]]
            ret[cat1][cat2] += 1
            ret[cat2][cat1] += 1

    for cat1 in col.keys():
        nb_cat1 = sum([1 if agents[p] == cat1 else 0 for p in agents.keys()])
        for cat2 in col.keys():
            ret[cat1][cat2] /= (nb_cat1 * len(graphs))

    return ret

def cumul_contacts_matrix(graphs, agents):
    ret = {cat2: {cat1: 0 for cat1 in col.keys()} for cat2 in col.keys()}
    for G in graphs:
        for e in G.edges:
            cat1 = agents[e[0]]
            cat2 = agents[e[1]]
            weight = G[e[0]][e[1]]['weight']
            ret[cat1][cat2] += weight
            ret[cat2][cat1] += weight

    for cat1 in col.keys():
        nb_cat1 = sum([1 if agents[p] == cat1 else 0 for p in agents.keys()])
        for cat2 in col.keys():
            ret[cat1][cat2] /= (nb_cat1 * len(graphs))
            ret[cat1][cat2] /= (60/20) # to minutes

    return ret


def best_params(data, plot=False, binsize=1, discrete=True):
    if discrete:
        best, p, param = get_best_distribution_discrete(data, binsize)
        if plot:
            _, bins, _ = plt.hist(data)
            hist_sum = len(data)
            f_exp = bin_pmf(lambda k: best.pmf(k, **param), bins, n=hist_sum)
            plt.plot(bins[:-1], f_exp)
            plt.show()

    else:
        best, p, param = get_best_distribution(data)
        if plot:
            _, bins, _ = plt.hist(data)
            hist_sum = len(data)
            f_exp = bin_pmf(lambda k: best.pdf(k, *param), bins, n=hist_sum)
            plt.plot(bins[:-1], f_exp)
            plt.show()

    return param

#def nbinom_params

if __name__ == '__main__':

    graphs, agents = read_graphs_from_csv()
    # print
    #for G in graphs:
        #plot_graph(G, agents)
    #plt.show()

    #person = 1157
    #neighbors = []
    #kernel = set(graphs[1].nodes)
    #for G in graphs[1:]:
    #    kernel = kernel.intersection(set(G.neighbors(person)))
        #neighbors += [set(G.neighbors(person))]

        #print(sorted(list(neighbors)))
    
    #print( [(myid, agents[myid]) for myid in frequent_contacts(person, graphs[1:])] )
    #print( average_nb_contacts(person, graphs[1:]) )
    #print( len(frequent_contacts(person, graphs[1:], threshold=3)) )

    #print('blab')

    #graphs = graphs[1:]
    #print(len(graphs[0].nodes))
    #for myid in graphs[0].nodes:
    #    if agents[myid] == 'ADM':
    #        print((myid, agents[myid], average_nb_contacts(myid, graphs), len(frequent_contacts(myid, graphs))))



    # plot
    # fig, ax = plt.subplots()

    # k = 3
    # #node_degree_sequence = [(n,d) for n, d in G.degree()]

    # x = range((MAX_DEGREE+1)//k)

    # y = [0 for i in x]
    # for myid in agents.keys():
    #     if agents[myid] == 'NUR':
    #         #cnt = math.trunc(average_nb_contacts(myid, graphs))
    #         cnt = len(frequent_contacts(myid, graphs))
    #         y[cnt//k] += 1
    # plt.bar(x, y, width=0.80)

    # #plt.legend()
    # plt.title("Average Degree Histogram")
    # plt.ylabel("Count")
    # plt.xlabel("Average Degree")
    # ax.set_xticks([d + 0.4 for d in x])
    # ax.set_xticklabels([k*(xi+1) for xi in x])

    # plt.show()

    #random.shuffle(graphs)
    
    # print('-nb contacts-')
    # pprint(nb_contacts_matrix(graphs, agents))
    # print('-cumul contacts-')
    # pprint(cumul_contacts_matrix(graphs, agents))
    
    print('# -- staff --')
    for cat in set(agents.values()):
        print(f'{cat}: {sum([1 if agents[i] == cat else 0 for i in agents.keys()])}')
    print('# -- end staff --')

    print('-changes-')
    for myid in agents.keys():
        print(f'{myid} [{agents[myid]}] kernelsize = {len(frequent_contacts(myid, graphs, agents))}')
        pprint(change_relativeto_kernel(myid, graphs))



    
    # kernel model:
    for cat1 in col.keys():
        print(f'Category.{cat1}: ')
        for cat2 in col.keys():
            data_full = [len(frequent_contacts(myid, graphs, agents, toCat=cat2)) for myid in agents.keys() if agents[myid] == cat1]
            data = [d for d in data_full if d != 0]
            data = data_full
            #print(f'probability of >0: {len(data)/len(data_full)}')
            if len(data) > 0:
                #print(best_params(data, plot=False))
                print(f'   Category.{cat2}: lambda: nbinom.rvs(**{best_params(data, plot=False)}),')

    # contacts model:
    print('# -- contacts --')
    for cat1 in col.keys():
        print(f'Category.{cat1}: ')
        for cat2 in col.keys():
            # fitting
            data_full = union_nb_contacts(graphs, agents, fromCat=cat1, toCat=cat2)
            data = [d for d in data_full if d != 0]
            if len(data) > 0:
                print(f'   Category.{cat2}: lambda: nbinom.rvs(**{best_params(data, plot=False)}),')
    print('# -- end contacts --')


    # fitting
    print('extra')
    data_full = union_nb_contacts(graphs, agents, fromCat='NUR', toCat='PAT')
    #data_full = [len(frequent_contacts(myid, graphs, agents, toCat='PAT')) for myid in agents.keys() if agents[myid] == 'NUR']
    data = [d for d in data_full if d != 0]
    #data = data_full
    print(f'probability of >0: {len(data)/len(data_full)}')
    plt.figure()
    print(best_params(data, plot=False))

    data = union_min_contacts(graphs)
    print(get_best_distribution(data))
    plt.hist(data, bins= 30)

    plt.figure()
    best_params(data, plot=True, binsize=1, discrete=False)

    plt.show()