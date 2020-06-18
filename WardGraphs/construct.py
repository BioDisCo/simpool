import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import csv
import collections

col = {'MED': 1, 'ADM': 20, 'NUR': 40, 'PAT': 60}
cmap = matplotlib.cm.get_cmap('Spectral')
CONTACT_THRESHOLD = 15*60/20  # 15min in terms of 20s
MAX_DEGREE = 50 # check

agents = dict()
interactions = []

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

def plot_graph(G):
    plt.figure()
    node_color = [ col[agents[a]] for a in agents.keys() ]
    nx.draw_networkx(G, with_labels=True, node_size=200, node_color=node_color, cmap=cmap)
    plot_degree_hist(G)

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
