import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

from create_nodes_vocabulary import *
from preprocess import load_stopwords


def create_TF_IDF_matrix():
    #load dictionaries
    nodes_content = pickle.load((open('D:\\Pythonas\\Chaos\\nodes_content.pkl','rb')))
    nodes_index = pickle.load((open('D:\\Pythonas\\Chaos\\nodes_index.pkl','rb')))
    #create a list out of nodes_content dictionary values
    nodes_texts = list(nodes_content.values())
    #load stopwords from preprocess
    stopwords = load_stopwords()
    #sklearn method call
    vectorizer = TfidfVectorizer(stop_words=stopwords,decode_error='ignore')
    #fit transform to create the sparse matrix
    tf_idf_matrix = vectorizer.fit_transform(nodes_texts)
    print(tf_idf_matrix.shape)
    matrix_out = open('D:\\Pythonas\\Chaos\\tf_idf_matrix.pkl','wb')
    pickle.dump(tf_idf_matrix,matrix_out)
    matrix_out.close()




create_TF_IDF_matrix()