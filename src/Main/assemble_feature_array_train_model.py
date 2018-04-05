import numpy as np
import pickle
from compute_node_tuples import *
from normalize_array_L2_norm import *
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
import codecs
from sklearn.preprocessing import robust_scale

nodes_index_path = 'D:\\Pythonas\\Chaos\\nodes_index.pkl'
nodes_index = pickle.load(open(nodes_index_path,'rb'))
nodes_tuples = sorted(compute_node_tuples(list(nodes_index.keys())))

# change the columns of the X array if you want to insert or remove any other features
number_of_features = 13
features_array_X = np.zeros((len(nodes_tuples),number_of_features),dtype=np.float32)

# load features to memory in order to assembly the train array
print("Loading dictionaries...")
cosine_similarity_dict = pickle.load(open('D:\Pythonas\Chaos\cosine_similarity_dict.pkl','rb'))
jaccard_distance_dict = pickle.load(open('D:\Pythonas\Chaos\jaccard_distance_dict.pkl','rb'))
in_degree_dictionary = pickle.load(open('D:\Pythonas\Chaos\in_degree_dictionary.pkl','rb'))
out_degree_dictionary = pickle.load(open('D:\Pythonas\Chaos\out_degree_dictionary.pkl','rb'))
degree_centrality_dictionary = pickle.load(open('D:\Pythonas\Chaos\degree_centrality.pkl','rb'))
common_out_neighbors_dictionary = pickle.load(open('D:\Pythonas\Chaos\common_out_neighbors.pkl','rb'))
common_in_neighbors_dictionary = pickle.load(open('D:\Pythonas\Chaos\common_in_neighbors.pkl','rb'))
nodes_core = pickle.load(open('D:\\Pythonas\\Chaos\\node_cores.pkl','rb'))
nodes_pagerank = pickle.load(open('D:\\Pythonas\\Chaos\\node_pagerank.pkl','rb'))

# load Y train array
print("Loading Y train array...")
Y_train = np.load('existing_edges_Y_train.npy')
Y_train = Y_train.ravel()

print("Assembling X train array...")
index = 0
for first_node,second_node in nodes_tuples:
    features_array_X[index, 0] = cosine_similarity_dict[(first_node, second_node)]
    features_array_X[index, 1] = jaccard_distance_dict[(first_node,second_node)]
    features_array_X[index, 2] = in_degree_dictionary[first_node]
    features_array_X[index, 3] = out_degree_dictionary[first_node]
    features_array_X[index, 4] = in_degree_dictionary[second_node]
    features_array_X[index, 5] = out_degree_dictionary[second_node]
    features_array_X[index, 6] = degree_centrality_dictionary[first_node]
    features_array_X[index, 7] = degree_centrality_dictionary[second_node]
    features_array_X[index, 8] = common_in_neighbors_dictionary[(first_node,second_node)]
    features_array_X[index, 9] = common_out_neighbors_dictionary[(first_node,second_node)]
    features_array_X[index, 10] = nodes_core[first_node] + nodes_core[second_node]
    features_array_X[index, 11] = nodes_pagerank[first_node]
    features_array_X[index, 12] = nodes_pagerank[second_node]
    index += 1


features_array_X =robust_scale(features_array_X,axis=0,with_centering=False, with_scaling=True,copy=True)

# Normalize X array per feature
#features_array_X = normalize_array(features_array_X,0)
# Normalize X array per training instance
features_array_X = normalize_array(features_array_X,1)

print("Starting training...")
rand_forest = RandomForestClassifier(n_estimators= 25,min_samples_leaf = 50)
rand_forest.fit(features_array_X,Y_train)
print("It's prediction time...")
predictions = rand_forest.predict_proba(features_array_X)

#print("Starting training...")
#log_regression = LogisticRegression()
#log_regression.fit(features_array_X,Y_train)
#print("It's prediction time...")
#predictions = log_regression.predict_proba(features_array_X)

# List of the possibilities of each edge to exist
positive_prob_list = [predictions[_,1] for _ in range(predictions.shape[0])]
# List to sort the possibilities
indices_of_top_edges = [i[0] for i in sorted(enumerate(positive_prob_list), key=lambda x:x[1],reverse=True)]

print("Find non existing edges...")
# choose 453 largest values
non_existing_edges = []
for i in indices_of_top_edges:
    if Y_train[i] == 0:
        edge = nodes_tuples[i]
        non_existing_edges.append(edge)
        if len(non_existing_edges) == 453:
            break

with codecs.open('predicted_edges.txt','w',encoding='utf-8') as f:
    for i,j in non_existing_edges:
        first_node_name = nodes_index[i]
        second_node_name = nodes_index[j]
        f.write(first_node_name+'\t'+second_node_name+'\n')


