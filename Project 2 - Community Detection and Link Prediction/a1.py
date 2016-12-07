"""
Online Social Network Analysis - Project 2
In this project, I have implemented community detection and link prediction algorithms using Facebook "like" data.

The file 'edges.txt.gz' indicates like relationships between facebook users. 
This was collected using snowball sampling: beginning with the user "Bill Gates", I crawled all the people he "likes", 
then, for each newly discovered user, I crawled all the people they liked.

Here the resulting graph is clustered into communities, as well as friends are recommended for Bill Gates. :)

The output of this analysis is provided in Log.txt.
"""

from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx
import urllib.request

def example_graph():
    """
    Creates the example graph from class. Used for testing.
    """
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g


def bfs(graph, root, max_depth):
    """
    This function performs a breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
    E.g., if max_depth=2, only paths of length 2 or less will be considered.
    This means that nodes greather than max_depth distance from the root will not
    appear in the result.

    The following two classes are used for this method's implementation:
      https://docs.python.org/3.5/library/collections.html#collections.defaultdict
      https://docs.python.org/3.5/library/collections.html#collections.deque

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    """
    node2num_paths = {}
    node2num_paths[root] = 1
    
    node2distances = {}
    node2distances[root] = 0
    
    node2parents = {}
    
    nodes_done = deque()
    nodes_done.append(root)
    
    nodes_to_do = deque()
    
    visited_node = {root : True}
    for n in graph:
        if(n != root):
            visited_node[n] = False
    
    for i in graph:
        if(len(nx.shortest_path(graph, root, i))-1 <= max_depth):
            node2distances[i] = len(nx.shortest_path(graph, root, i))-1
    
    for m in node2distances:
        if m!=root:
            node2num_paths[m] = len(list(nx.all_shortest_paths(graph,root,m)))
    
    while(max_depth > 0):
        for i in list(nodes_done):
            if(visited_node.get(i) == True):
                nodes_done.popleft()
                for j in graph.neighbors(i):
                    if(j != root):
                        nodes_to_do.append(j)                  
                        if j not in node2parents:
                            node2parents[j] = [i]
                        elif len(node2parents[j]) < node2num_paths[j]:
                            node2parents[j].append(i)

                        if(visited_node[j] == False):
                            visited_node[j] = True
        for k in list(nodes_to_do):
            if(k not in nodes_done):
                nodes_done.append(k)
        nodes_to_do.clear()
            
        max_depth -= 1
    
    for m in node2parents:
        node2num_paths[m] = len(node2parents[m])
        
    return node2distances, node2num_paths, node2parents
    pass


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    This function computes the final step of the Girvan-Newman algorithm:
        The third and final step is to calculate for each edge e the sum
        over all nodes Y of the fraction of shortest paths from the root
        X to Y that go through e. This calculation involves computing this
        sum for both nodes and edges, from the bottom. Each node other
        than the root is given a credit of 1, representing the shortest
        path to that node. This credit may be divided among nodes and
        edges above, since there could be several different shortest paths
        to the node.

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge. 
      Any edges excluded from the results in bfs are also exluded here.
    """    
    ncredit = 1
    ecredit = 0
    nodecredit = {}
    result = {}
    num_parents = []

    for n in node2distances.keys():
        if(n != root):
            nodecredit[n] = 1
        else:
            nodecredit[n] = 0

    sorted_node2distances = sorted(node2distances.items(), key = lambda x:x[1], reverse=True)
    for node_distance in sorted_node2distances:
        if(node_distance[0] != root):
            nodecredit[node_distance[0]] = (nodecredit[node_distance[0]])/(node2num_paths[node_distance[0]])
            num_parents = node2parents[node_distance[0]]
            for parent in num_parents:
                nodecredit[parent] = nodecredit[parent] + nodecredit[node_distance[0]]
                all_edges_in_graph = sorted((node_distance[0], parent))
                result[tuple(all_edges_in_graph)] = nodecredit[node_distance[0]]
                
    return result

    pass


def approximate_betweenness(graph, max_depth):
    """
    Computes the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    The bfs and bottom_up functions defined above are used here for each node
    in the graph, and sum together the results. The final result is divided by 2 at the
    end to get the final betweenness.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A dict mapping edges to betweenness. Each key is a tuple of two strings
      representing an edge.
    """
    betweenness = {}
    for node in graph:
        node2distances, node2num_paths, node2parents = bfs(graph, node, max_depth)
        result = bottom_up(node, node2distances, node2num_paths, node2parents)
        for r in result:
            if r not in betweenness.keys():
                betweenness[r] = result[r]/2
            else:
                betweenness[r] += result[r]/2

    return betweenness

    pass


def partition_girvan_newman(graph, max_depth):
    """
    Using the approximate_betweenness from above function the graph is partitioned here.
    The edges are recursively removed until more than one component is created, then
    those components are returned.
    That is, compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.
    
    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A list of networkx Graph objects, one per partition.
    """
    duplicate_graph = graph.copy()
    betweenness = approximate_betweenness(graph, max_depth)
    betweenness_sort = sorted(sorted(betweenness.items(), key = lambda x:(x[0],x[1]), reverse=False), key = lambda y:y[1], reverse=True)
    
    num_components = nx.number_connected_components(duplicate_graph)
   
    count = 0

    while(num_components <= 1):
        if graph.has_edge(betweenness_sort[count][0][0],betweenness_sort[count][0][1]):
            duplicate_graph.remove_edge(betweenness_sort[count][0][0],betweenness_sort[count][0][1])
            count = count + 1
            num_components = nx.number_connected_components(duplicate_graph)

    return list(nx.connected_component_subgraphs(duplicate_graph))

    pass


def get_subgraph(graph, min_degree):
    """Returns a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    This is used in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.
    """
    sg = nx.Graph()
    deg = graph.degree()
    sg.add_nodes_from([k for k in deg if deg[k] >= min_degree])
   
    for i in range(0, len(sg.nodes())):
        for j in range(0, len(sg.nodes())):
            
            if(graph.has_edge(sg.nodes()[i], sg.nodes()[j])and i!=j):
                sg.add_edge(sg.nodes()[i], sg.nodes()[j])
    return sg

    pass


def volume(nodes, graph):
    """
    Computes the volume for a list of nodes, which
    is the number of edges in `graph` with at least one end in
    nodes' list.
    Params:
      nodes...a list of strings for the nodes to compute the volume of.
      graph...a networkx graph
    """
    deg = graph.edges(nodes)
    s = len(deg)
    return s
    pass


def cut(S, T, graph):
    """
    Computes the cut-set of the cut (S,T), which is
    the set of edges that have one endpoint in S and
    the other in T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An int representing the cut-set.
    """    
    number_of_cuts = 0
    for e in graph.edges():
        for s in S:
            for t in T:
                if((s == e[0] and t == e[1]) or (s == e[1] and t == e[0])):
                    number_of_cuts += 1
    
    return number_of_cuts
    pass


def norm_cut(S, T, graph):
    """
    The normalized cut value is computed for the cut S/T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An float representing the normalized cut value
    """
    normalized_cut_value = 0.0
    cut_value = cut(S, T, graph)
    vol_s = volume(S, graph)
    vol_t = volume(T, graph)
    normalized_cut_value = (cut_value/vol_s) + (cut_value/vol_t)
    return normalized_cut_value
    pass


def score_max_depths(graph, max_depths):
    """
    In order to assess the quality of the approximate partitioning method
    developed above, this method is run it with different values for max_depth
    and see how it affects the norm_cut score of the resulting partitions.
    Smaller norm_cut scores correspond to better partitions.

    Params:
      graph........a networkx Graph
      max_depths...a list of ints for the max_depth values to be passed
                   to calls to partition_girvan_newman

    Returns:
      A list of (int, float) tuples representing the max_depth and the
      norm_cut value obtained by the partitions returned by
      partition_girvan_newman.
    """
    result_list = []
    for i in max_depths:
        components = partition_girvan_newman(graph, i)
        result_list.append((i,norm_cut(components[0].nodes(), components[1].nodes(),graph)))
    
    return result_list
    
    pass


def make_training_graph(graph, test_node, n):
    """
    To make a training graph, we need to remove n edges from the graph.
    Here a test node is assumed for which the edges are removed. 
    i.e Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.
    
    Params:
      graph.......a networkx Graph
      test_node...a string representing one node in the graph whose
                  edges will be removed.
      n...........the number of edges to remove.

    Returns:
      A *new* networkx Graph with n edges removed.
    """
    n1 = 0
    train_graph = graph.copy()
    edges = train_graph.edges()
    neigh = train_graph.neighbors(test_node)
    sorted_neigh = sorted(neigh)
    while((n1 < n) and (n1 < len(sorted_neigh))):
        train_graph.remove_edge(test_node,sorted_neigh[n1])
        n1 += 1

    return train_graph
    pass


def jaccard(graph, node, k):
    """
    This function computes the k highest scoring edges to add to this node based on
    the Jaccard similarity measure.

    Params:
      graph....a networkx graph
      node.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.

    Returns:
      A list of tuples in descending order of score representing the
      recommended new edges. Ties are broken by
      alphabetical order of the terminal node in the edge.
    """
    jaccard_all = []
    pre_final_jaccard = []
    final_jaccard = []
    for i in graph:
        preds = nx.jaccard_coefficient(graph, [(node, i)])
        for u,v,w in preds:
            jaccard_all.append(([(u, v)], w))
    sorted_jaccard_all = sorted(jaccard_all)
     
    for n in range(len(sorted_jaccard_all)):
        if(((graph.has_edge(node, sorted_jaccard_all[n][0][0][1])) == False) and (node != sorted_jaccard_all[n][0][0][1])):
                pre_final_jaccard.append(((sorted_jaccard_all[n][0][0][0], sorted_jaccard_all[n][0][0][1]), sorted_jaccard_all[n][1]))
    pre_final_jaccard = sorted(sorted(pre_final_jaccard, key = lambda x:(x[0],x[1]), reverse=False), key = lambda y:y[1], reverse=True)
    count = 0
    while(count < k):
        final_jaccard.append(pre_final_jaccard[count])
        count = count+1
    return final_jaccard
    
    pass


def path_score(graph, root, k, beta):
    """
    Computes a new link prediction scoring function based on the shortest
    paths between two nodes, as defined above.

    Params:
      graph....a networkx graph
      root.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
      beta.....the beta parameter in the equation above.

    Returns:
      A list of tuples in descending order of score. Ties are broken by
      alphabetical order of the terminal node in the edge.
    """
    score = []
    node2distances, node2num_paths, node2parents = bfs(graph, root, 5)
    for n in graph.nodes():
        if(n != root and graph.has_edge(root, n) == False):
            #score.append((beta**node2distances[n])*(node2num_paths[n]))
            score.append(((root, n), ((beta**node2distances[n])*(node2num_paths[n]))))
    score = sorted(sorted(score, key = lambda x:(x[0],x[1]), reverse=False), key = lambda y:y[1], reverse=True)
    return score[:k]
    pass


def evaluate(predicted_edges, graph):
    """
    Returns the fraction of the predicted edges that exist in the graph.

    Args:
      predicted_edges...a list of edges (tuples) that are predicted to
                        exist in this graph
      graph.............a networkx Graph

    Returns:
      The fraction of edges in predicted_edges that exist in the graph.
    """
    edges = graph.edges()
    numerator = 0
    for e in predicted_edges:
        if(e in edges or  e[::-1] in edges):
            numerator += 1
    result = (numerator*1./len(predicted_edges))
    return result
    pass


def download_data():
    """
    Download the Facebook data already stored in edges.txt.gz file
    """
    urllib.request.urlretrieve('http://cs.iit.edu/~culotta/cs579/a1/edges.txt.gz', 'edges.txt.gz')


def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():
    download_data()
    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))
    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))
    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())

    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))


    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
          evaluate([x[0] for x in path_scores], subgraph))


if __name__ == '__main__':
    main()
