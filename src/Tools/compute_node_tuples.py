import itertools

def compute_node_tuples(nodes_list):
    nodes_tuples = set(itertools.permutations(nodes_list, 2))
    return nodes_tuples