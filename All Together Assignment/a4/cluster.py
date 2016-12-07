from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
from TwitterAPI import TwitterAPI
from collect import get_twitter, add_all_friends, get_users, read_screen_names

def count_friends(users):
    friend_list = []
    add_all_friends(get_twitter(), users)
    for user in users:
        friend_list.extend(user['friends'])
    return Counter(friend_list)
    pass

def create_graph(users, friend_counts):
    graph = nx.Graph()
    for friend in friend_counts:
        if friend_counts[friend] > 1:
            for user in users:
                if friend in user['friends']:
                    graph.add_edge(user['id'],friend)
    return graph
    pass

def draw_network(graph, users, filename):
    team = {}
    #print("\nin draw_network, users = ", users[0]['id'])
    for user in users:
        team[user['id']] = user['screen_name']
    plt.figure(figsize=(25,25))
    nx.draw_networkx(graph, nx.spring_layout(graph), labels = team, font_size=18, font_color='blue', font_weight="bold", node_color='black', edge_color='red', with_labels=True)
    plt.axis('off')
    plt.savefig(filename, format = "PNG")
    pass

def girvan_partitioning(G, minsize=20, maxsize=35):
    
    if G.order() == 1:
            return [G.nodes()]

    def suit_edge(G0):
        # eb is dict of (edge, score) pairs, where higher is better
        # Return the edge with the highest score.
        eb = nx.edge_betweenness_centrality(G)
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)

    # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(G)]
    edges_removed=0
    edge_to_remove = suit_edge(G)
    while len(components) == 1:
        for n in edge_to_remove: 
            if len(components) == 1:
                G.remove_edge(*n[0])
                edges_removed+=1
                components = [c for c in nx.connected_component_subgraphs(G)]
    #print ('removed %d edges' %(edges_removed))

    comp = [c for c in components]
    comp_size=[i.order() for i in comp]
    #print ('Component sizes='+str(comp_size))
    result=[]
    for c in comp:
            if c.order() >maxsize:
                result.extend(girvan_partitioning(c,minsize,maxsize))
            elif (c.order() in range(minsize,maxsize+1)):
                #print ('stopping for %d' %(c.order()))
                result.append(c.nodes())

    return result

def main():
    twitter = get_twitter()
    screen_names = read_screen_names('teams.txt')
    users = get_users(twitter, screen_names)
    friend_counts = count_friends(users)
    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')    
    results = []
    g = graph.copy()
    results = girvan_partitioning(g, 10, 15)
    print('Length of cluster 1 = ', len(results[0]))
    print('Length of cluster 2 = ', len(results[1]))
    print('The nodes in clusters are :')
    print(results[0])
    print(results[1])
    
if __name__ == '__main__':
    main()
