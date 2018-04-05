import networkx as nx
import pickle
from compute_node_tuples import *
import numpy as np

class Graph_metrics(object):

    def __init__(self,G,nodes_index,nodes_tuples,inverse_nodes_index):
        self.G = G
        self.nodes_index = nodes_index
        self.nodes_tuples = nodes_tuples
        self.inverse_nodes_index = inverse_nodes_index

    #Create a 1-D array with all possible edges and assign 1 if they already exist , 0 otherwise
    def record_existing_edges(self):
        index = 0
        existing_edges = np.zeros((len(self.nodes_tuples),1),dtype=np.int32)
        print('existing edges array shape: ',existing_edges.shape)
        sorted_node_tuples = sorted(self.nodes_tuples)
        for first_node,second_node in sorted_node_tuples:
            first_node_name = self.nodes_index[first_node]
            second_node_name = self.nodes_index[second_node]
            if self.G.has_edge(first_node_name, second_node_name):
                existing_edges[index] = 1
            else:
                existing_edges[index] = 0
            index += 1
        np.save('existing_edges_Y_train',existing_edges)

    #Compute the in degree dictionary
    def compute_in_degree(self):
        in_degree_dictionary = {}
        dict_out = open('in_degree_dictionary.pkl','wb')
        for index,node in self.nodes_index.items():
            in_degree_dictionary[index] = self.G.in_degree(node)
        pickle.dump(in_degree_dictionary,dict_out)
        dict_out.close()
        print('Calculation of nodes in_degree completed successfully!')

    #Compute the out degree dictionary
    def compute_out_degree(self):
        out_degree_dictionary = {}
        dict_out = open('out_degree_dictionary.pkl','wb')
        for index,node in self.nodes_index.items():
            out_degree_dictionary[index] = self.G.out_degree(node)
        pickle.dump(out_degree_dictionary,dict_out)
        dict_out.close()
        print('Calculation of nodes out_degree completed successfully!')

    #Compute the degree centrality
    def degree_centrality(self):
        degree_centrality = {}
        centrality = nx.degree_centrality(self.G)
        dict_out = open('degree_centrality.pkl', 'wb')
        for index, node in self.nodes_index.items():
            degree_centrality[index] = centrality[node]
        pickle.dump(degree_centrality, dict_out)
        dict_out.close()
        print('Calculation of nodes degree centrality completed successfully!')

    #Compute the common out neighbours
    def common_out_neighbors(self):
        common_out_neighbors={}
        for first_node,second_node in self.nodes_tuples:
            first_node_name = self.nodes_index[first_node]
            second_node_name = self.nodes_index[second_node]
            common_out_neighbors[(first_node,second_node)]=len(set(self.G.successors(first_node_name)).intersection(self.G.successors(second_node_name)))
        dict_out = open('common_out_neighbors.pkl', 'wb')
        pickle.dump(common_out_neighbors, dict_out)
        dict_out.close()

    #Compute the common in neighbours
    def common_in_neighbors(self):
        common_in_neighbors = {}
        for first_node, second_node in self.nodes_tuples:
            first_node_name = self.nodes_index[first_node]
            second_node_name = self.nodes_index[second_node]
            common_in_neighbors[(first_node, second_node)] = len(set(self.G.predecessors(first_node_name)).intersection(self.G.predecessors(second_node_name)))
        dict_out = open('common_in_neighbors.pkl', 'wb')
        pickle.dump(common_in_neighbors, dict_out)
        dict_out.close()

    #Compute max K core of the node from sklearn
    def main_core(self):
        node_cores = {}
        nx_cores =nx.core_number(self.G)
        for node in nx_cores:
            index = self.inverse_nodes_index[node]
            node_cores[index]=nx_cores[node]
        dict_out = open('node_cores.pkl', 'wb')
        pickle.dump(node_cores, dict_out)
        dict_out.close()

    #Compute Pagerank
    def pagerank(self):
        nodes_pagerank = {}
        nx_pagerank = nx.pagerank(self.G)
        for node in nx_pagerank:
            index = self.inverse_nodes_index[node]
            nodes_pagerank[index] = nx_pagerank[node]
        dict_out = open('node_pagerank.pkl', 'wb')
        pickle.dump(nodes_pagerank, dict_out)
        dict_out.close()


if __name__ == '__main__':
    #Create a directed Graph
    G = nx.read_edgelist('edgelist.txt', delimiter='\t', create_using=nx.DiGraph())
    #Load Dictionaries
    nodes_index = pickle.load((open('D:\Pythonas\Chaos\\nodes_index.pkl', 'rb')))
    inverse_nodes_index = pickle.load((open('D:\Pythonas\Chaos\inverse_nodes_index.pkl', 'rb')))
    #Method Call compute_node_tuples
    nodes_tuples = compute_node_tuples(list(nodes_index.keys()))
    #Create a Graph_metrics instance
    graph = Graph_metrics(G,nodes_index,nodes_tuples,inverse_nodes_index)
    graph.record_existing_edges()
    graph.compute_in_degree()
    graph.compute_out_degree()
    graph.degree_centrality()
    graph.common_out_neighbors()
    graph.common_in_neighbors()
    graph.main_core()
    graph.pagerank()