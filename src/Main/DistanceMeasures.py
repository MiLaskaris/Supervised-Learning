from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from compute_node_tuples import *
import pickle
import unicodedata


class DistanceMeasures(object):
    def __init__(self, tf_idf_matrix,nodes_index, nodes_content,inverse_nodes_index):
        self.tf_idf_matrix = tf_idf_matrix
        self.nodes_index = nodes_index
        self.nodes_content = nodes_content
        self.inverse_nodes_index = inverse_nodes_index

    #Method for cosineSimilarity (sklearn)
    def compute_cosineSimilarity(self):
        similarities = cosine_similarity(self.tf_idf_matrix)
        cosine_similarity_dict= {}
        # Write Dictionary
        sorted_node_tuples = sorted(compute_node_tuples(list(self.nodes_index.keys())))
        for first_node,second_node in sorted_node_tuples:
            cosine_similarity_dict[(first_node,second_node)] = similarities[first_node][second_node]
        dict_out = open ('cosine_similarity_dict.pkl', 'wb')
        pickle.dump(cosine_similarity_dict, dict_out)
        dict_out.close()

    #Method for Jaccard (custom)
    def compute_Jaccard(self):
        jaccard_distance_dict = {}
        nodes_sets = {}
        index = 0
        sorted_node_tuples = sorted(compute_node_tuples(list(self.nodes_index.keys())))
        print('Creating nodes\' text_sets...')
        for node in self.nodes_content:
            node_text = self.nodes_content[node]
            node_text = set(node_text.split(' '))
            node_index = self.inverse_nodes_index[node]
            nodes_sets[node_index]=node_text
        print('Number of node tuples: ', len(sorted_node_tuples))
        for first_node, second_node in sorted_node_tuples:
            print('Current tuple: ',index+1 )
            if (first_node,second_node) not in jaccard_distance_dict:
                first_node_text = nodes_sets[first_node]
                second_node_text = nodes_sets[second_node]
                texts_union = first_node_text | second_node_text
                texts_intersection = first_node_text & second_node_text
                jaccard_distance = 1-(len(texts_intersection)/len(texts_union))
                jaccard_distance_dict[(first_node,second_node)] = jaccard_distance
                jaccard_distance_dict[(second_node,first_node)]= jaccard_distance
            index += 1
        dict_out = open('jaccard_distance_dict.pkl', 'wb')
        pickle.dump(jaccard_distance_dict, dict_out)
        dict_out.close()






