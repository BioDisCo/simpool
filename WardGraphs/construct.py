import networkx as nx
import matplotlib.pyplot as plt
import csv
import collections

col = {'MED': 1, 'ADM': 20, 'NUR': 40, 'PAT': 60}
CONTACT_THRESHOLD = 15*60/20  # 15min in terms of 20s

def plot_degree_hist(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

def plot_graph(G):
    plt.figure()
    node_color = [ col[agents[a]] for a in agents.keys() ]
    nx.draw_networkx(G, with_labels=True, node_size=200, node_color=node_color, cmap=plt.cm.Blues)
    plot_degree_hist(G)

agents = dict()
interactions = []
with open('detailed_list_of_contacts_Hospital.dat', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quotechar='#')
    for row in reader:
        time_sec = int(row[0])
        agent1 = int(row[1])
        agent2 = int(row[2])
        agent1_type = row[3]
        agent2_type = row[4]

        # log types
        agents[agent1] = agent1_type
        agents[agent2] = agent2_type
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

# print
for G in graphs:
    plot_graph(G)
plt.show()
