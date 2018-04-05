# -*- coding: utf-8 -*-
import glob
import os
import zipfile
import re
import pickle
import collections
from inverse_dictionary import *
import unicodedata

#method for unzipping
def unzip_documents(file_path):
    node_txt = {}
    with zipfile.ZipFile(file_path,'r') as zip_ref:
        for file in zip_ref.namelist():
            if not os.path.isdir(file):
                with zip_ref.open(file) as f:
                    text=f.read()
                    node_txt[text.strip()]=os.path.basename(file)
    return node_txt

#method for stopwords
def load_stopwords():
    with open('stopwords.txt',encoding="utf-8") as stop:
        stopwords= [x.strip() for x in stop.readlines()]
    return stopwords

#method for preprocessing data
def preprocess_data(text):
    #removes links
    no_links = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    # remove digits
    no_digits = re.sub('\d+', ' ',no_links)
    # no words with length under 3 characters
    no_small_words = re.sub(r'\b\w{1,3}\b', '', no_digits)
    # no words with length above 10 characters
    no_big_words = re.sub(r'\b\w{10,1000}\b', '', no_small_words)
    # remove extra white spaces
    no_extra_spaces = re.sub('\s\s+', ' ', no_big_words)
    # remove tonality
    no_tonality = no_extra_spaces.replace('ό','ο').replace('έ','ε').replace('ί','ι').replace('ή','η').replace('ά','α').replace('ύ','υ').replace('ώ','ω').replace('ϊ','ι')
    #remove character '\xa0'
    no_special_character = unicodedata.normalize('NFKD',no_tonality)
    #removes punctuation, extra whitespaces,etc.
    clean_text = re.sub('[.,?;*!%^&_@+():-\[\]{}]','',no_special_character).replace('"', '').replace('/', '').replace('\\', '').replace("'", '').replace('|', '').replace('--', '').replace('–', ' ').replace('«', '').replace('»', '').replace('-','').replace('…','').replace('€','').replace('√','').replace('”','').replace('©','')

    return clean_text

#method for concatenation
def concatenate_texts(node_txt):
    text =''
    for key in node_txt:
        text += key.decode('utf-8')+' '
    return preprocess_data(text)

if __name__ == '__main__':

    #Assign name and path for the dictionaries
    path_to_hosts = 'D:\Pythonas\Chaos\Dataset\hosts'
    nodes_index_path = 'D:\Pythonas\Chaos\\nodes_index.pkl'
    nodes_content_path = 'D:\Pythonas\Chaos\\nodes_content.pkl'
    filenames = os.listdir(path_to_hosts)
    all_nodes = collections.OrderedDict()
    nodes_index = {} # for node indexing --index is the key (e.g. 0,1,2,3,...) and node name is the value (e.g. 10deco.gr, 5a3.gr, etc.)
    index = 0
    #print (filenames)
    nodes_content_file = open(nodes_content_path,'wb')
    nodes_index_file = open(nodes_index_path,'wb')
    for zippedFile in filenames:
        node = os.path.splitext(zippedFile)[0]
        node_txt = unzip_documents(path_to_hosts+'\\'+zippedFile)
        text = concatenate_texts(node_txt)
        all_nodes[node] = text
        nodes_index[index]= node
        index += 1
    pickle.dump(all_nodes,nodes_content_file)
    pickle.dump(nodes_index,nodes_index_file)
    nodes_content_file.close()
    nodes_index_file.close()

    #create an inverse dictionary of nodes, where node is the key and index is the value
    create_inverse_dictionary('D:\\Pythonas\\Chaos\\', nodes_index_path,'inverse_nodes_index')
