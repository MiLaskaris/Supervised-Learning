import pickle

def create_inverse_dictionary(inversed_path,initial_dict_file,filename):

    inversed_dict_path = inversed_path+'%s.pkl' % filename
    initial_dict = pickle.load(open(initial_dict_file,'rb'))
    inversed_dict = {}

    for key, value in initial_dict.items():
        inversed_dict[value]=key

    dict_out = open(inversed_dict_path,'wb')
    pickle.dump(inversed_dict,dict_out)
    dict_out.close()
