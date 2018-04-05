import pickle
import numpy as np

def create_nodes_vocabulary(words_list,all_nodes,nodes_index):
    boolean_array = np.zeros((len(all_nodes),len(words_list)),dtype=np.int32)
    print('Total words: ',len(words_list))
    for pos,word in enumerate(words_list):
        print('Current word: ',pos+1)
        for n_index,node in enumerate(all_nodes):
            node_text = all_nodes[node].split(' ')
            if word in node_text:
                boolean_array[n_index,pos] = 1
    return boolean_array





