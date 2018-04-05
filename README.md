# Supervised-Learning

The goal of this project was to use machine learning algorithms and especially, supervised learning techniques
in order to predict the links between the nodes of a graph. 
We were given a dataset of nodes which represented the websites of the Greek web (i.e., name of web page and content).  
The program is written in python and it aims at pre-processing and predicting the links. 

We computed a number of features such as Cosine Similarity, Jaccard distance, In degree, 
Out degree, Degree centrality, Common in neighbors, K-core and Pagerank of the nodes (web pages).

The sequence of execution of the files is: 

1.preprocess.py
2.tfidfVectorizer.py
3.compute_features.py 
4.read_Graph.py
5.assemble_feature_array_train_model.py
