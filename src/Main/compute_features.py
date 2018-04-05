import pickle
from DistanceMeasures import *

#Load the tf_idf_matrix and other dictionaries
matrix_in = open('D:\Pythonas\Chaos\\tf_idf_matrix.pkl','rb')
tf_idf_matrix = pickle.load(matrix_in)
matrix_in.close()
nodes_index = pickle.load(open('D:\Pythonas\Chaos\\nodes_index.pkl','rb'))
nodes_content = pickle.load(open('D:\Pythonas\Chaos\\nodes_content.pkl','rb'))
inverse_nodes_index = pickle.load(open('D:\Pythonas\Chaos\\inverse_nodes_index.pkl','rb'))
#Create a DistanceMeasures instance
features = DistanceMeasures(tf_idf_matrix,nodes_index,nodes_content,inverse_nodes_index)
#Compute cosine similarity
features.compute_cosineSimilarity()
#Compute Jaccard distance
features.compute_Jaccard()