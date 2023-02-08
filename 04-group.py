NUM_USERS = 45000
GROUP_SIZE = 10200

print('Building Graph...')
import networkx as nx
G = nx.Graph()
G.add_nodes_from(range(NUM_USERS))

cfsn = open('./data/cfsn.txt', 'r')
lines  = cfsn.readlines()
for line in lines:
    edge = line.strip().split(",")
    G.add_edge(int(edge[0]), int(edge[1]))

print('Creating Groups...')
cc = nx.connected_components(G)
cc = [c for c in cc if len(c) > 1]
cc.sort(key=len)

groups = []
group = set([])
for c in cc:
    group = group.union(c)
    if (len(group) >= GROUP_SIZE):
        groups.append(group)
        group = set([])

print('Created ' + str(len(groups)) + ' groups of size', [len(group) for group in groups])
file = open('./data/groups.txt','w')
file.write("\n".join([",".join(map(str, group)) for group in groups]))
file.close()
