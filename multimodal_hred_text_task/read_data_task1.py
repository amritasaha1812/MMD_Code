import sys
import numpy as np
import cPickle as pkl
import os
#from params import *
from prepare_data_for_hred import PrepareData
start_symbol_index = 0
end_symbol_index = 1
unk_symbol_index = 2
pad_symbol_index = 3
import os
'''
def get_params(dir):
    param={}
    dir= str(dir)
    param['train_dir_loc']=dir+"/train/"
    param['valid_dir_loc']=dir+"/valid/"
    param['test_dir_loc']=dir+"/test/"
    param['dump_dir_loc']=dir+"/dump/"
    param['test_output_dir']=dir+"/test_output/"
    param['vocab_file']=dir+"/vocab.pkl"
    param['train_data_file']=dir+"/dump/train_data_file.pkl"
    param['valid_data_file']=dir+"/dump/valid_data_file.pkl"
    param['test_data_file']=dir+"/dump/test_data_file.pkl"
    param['vocab_file']=dir+"/vocab.pkl"
    param['vocab_stats_file']=dir+"/vocab_stats.pkl"
    param['model_path']=dir+"/model"
    param['terminal_op']=dir+"/terminal_output.txt"
    param['logs_path']=dir+"/log"
    param['text_embedding_size'] = 512
    param['image_rep_size'] = 4096
    param['image_embedding_size'] = 512
    param['activation'] = None #tf.tanh
    param['output_activation'] = None #tf.nn.softmax
    param['cell_size']=512
    param['cell_type']=None #rnn_cell.GRUCell
    param['batch_size']=64
    param['vocab_freq_cutoff']=2
    param['learning_rate']=0.0004
    param['patience']=200
    param['early_stop']=100
    param['max_epochs']=1000000
    param['max_len']=20
    param['max_images']=5
    param['max_utter']=5
    param['print_train_freq']=1000
    param['show_grad_freq']=20000
    param['valid_freq']=10000
    param['max_gradient_norm']=0.1
    param['train_loss_incremenet_tolerance']=0.01
    return param
'''
#BELOW LINE IS JUST FOR THE SAMPLE DATA.. THIS WOULD BE REPLACED WITH AN ANNOY INDEX BASED LOADING
#sample_image_index = pkl.load(open('true_data_25k/sample_image_index.pkl'))

#BELOW LINES ARE FOR THE ACTUAL ANNOY INDEX BASED LOADING.. COMMENT OUT THE ABOVE LINES AND UNCOMMENT THE FOLLOWING
from annoy import AnnoyIndex
import cPickle as pkl
annoyIndex = AnnoyIndex(4096, metric='euclidean')
annoyIndex.load('../multimodal_dialogue/image_annoy_index/annoy.ann')
annoyPkl = pkl.load(open('../multimodal_dialogue/image_annoy_index/FileNameMapToIndex.pkl'))

def get_dialog_dict(param, is_test = False):
    train_dir_loc = param['train_dir_loc']
    valid_dir_loc = param['valid_dir_loc']
    test_dir_loc = param['test_dir_loc']
    dump_dir_loc = param['dump_dir_loc']	
    vocab_file = param['vocab_file']
    vocab_stats_file = param['vocab_stats_file']
    vocab_freq_cutoff = param['vocab_freq_cutoff']
    train_data_file = param['train_data_file']
    valid_data_file = param['valid_data_file']
    test_data_file = param['test_data_file']
    max_utter = param['max_utter']	
    max_len = param['max_len']
    max_images = param['max_images']
    if 'test_state' in param:
	test_state = param['test_state']
    else:
	test_state = None
    preparedata = PrepareData(max_utter, max_len, max_images, start_symbol_index, end_symbol_index, unk_symbol_index, pad_symbol_index, "text", cutoff=vocab_freq_cutoff)
    if os.path.isfile(vocab_file):
	print 'found existing vocab file in '+str(vocab_file)+', ... reading from there'
    if not is_test:
	    preparedata.prepare_data(train_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "train"), train_data_file)
	    preparedata.prepare_data(valid_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "valid"), valid_data_file)
    if test_state is not None:
	    preparedata.prepare_data(test_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc+"/test_data_file_state/", "test_"+test_state), test_data_file, False, True, test_state)
    else:
	    preparedata.prepare_data(test_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "test"), test_data_file, False, True, test_state)

def get_weights(padded_target, batch_size, max_len, actual_seq_len):
    remaining_seq_len = max_len - actual_seq_len
    weights = [[1.]*actual_seq_len_i+[0.]*remaining_seq_len_i for actual_seq_len_i, remaining_seq_len_i in zip(actual_seq_len, remaining_seq_len)]
    weights = np.asarray(weights)   
    #print 'weights shape ', weights.shape
    return weights

def get_utter_seq_len(dialogue_text_dict, dialogue_image_dict, dialogue_target, max_len, max_images, image_rep_size, max_utter, batch_size):
    padded_utters_id = None
    padded_image_rep = None
    padded_target=[]
    decode_seq_len=[]
    dummy_image=[0.]*image_rep_size
    #USE THE BELOW TWO COMMENTED LINES IF START AND END SYMBOL HAS NOT BEEN APPENDED YET
    #padded_utters_id = np.asarray([[[start_symbol_index]+xij[:max_len]+[end_symbol_index] if len(xij)>max_len else [start_symbol_index]+xij+[end_symbol_index]+[pad_symbol_index]*(max_len-len(xij)) for xij in dialogue_i] for dialogue_i in dialogue_text_dict])
    #padded_image_rep = np.asarray([[xij[:max_images] if len(xij)>max_images else xij+dummy_image*(max_images-len(xij)) for xij in dialogue_i] for dialogue_i in dialogue_image_dict])
    #USE THE BELOW TWO COMMENTED LINES IF START AND END SYMBOL HAS BEEN ALREADY APPENDED 
    #padded_utters_id = np.asarray([[xij[:(max_len-1)]+[end_symbol_index] if len(xij)>max_len else xij+[pad_symbol_index]*(max_len-len(xij)) for xij in dialogue_i] for dialogue_i in dialogue_text_dict])
    #padded_image_rep = np.asarray([[xij[:max_images] if len(xij)>max_images else xij+[dummy_image]*(max_images-len(xij)) for xij in dialogue_i] for dialogue_i in dialogue_image_dict])
    padded_utters_id = np.asarray([[xij for xij in dialogue_i] for dialogue_i in dialogue_text_dict])
    padded_image_rep = np.asarray([[xij for xij in dialogue_i] for dialogue_i in dialogue_image_dict])	
    #padded_utters_id is of dimension (batch_size, max_utter+1, max_len)
    #padded_image_rep is of dimension (batch_size, max_utter+1, max_images, image_rep_size)

    #USE THE BELOW COMMENTED LINE IF START AND END SYMBOL HAS NOT BEEN APPENDED YET
    #padded_target = np.asarray([[start_symbol_index]+xij[:max_len]+[end_symbol_index] if len(xi)>max_len else [start_symbol_index]+xi+[end_symbol_index]+[pad_symbol_index]*(max_len-len(xi)) for xi in dialogue)])
    #USE THE BELOW COMMENTED LINE IF START AND END SYMBOL HAS BEEN ALREADY APPENDED
    #padded_target = np.asarray([xi[:(max_len-1)]+[end_symbol_index] if len(xi)>max_len else xi+[pad_symbol_index]*(max_len-len(xi)) for xi in dialogue_target])
    padded_target = np.asarray([xi  for xi in dialogue_target])
    #padded_target is of dimension (batch_size, max_len)
    pad_to_target = np.reshape(np.asarray([pad_symbol_index]*batch_size), (batch_size, 1))
    #pad_to_target is of dimension (batch_size, 1)
    #print 'padded_target[:,:-1].shape ', padded_target[:,:-1].shape
    #print 'pad_to_target.shape ',pad_to_target.shape
    #print 'padded_target[:,:-1].shape ',padded_target[:,:-1].shape
    padded_decoder_input = np.concatenate((pad_to_target, padded_target[:,:-1]), axis=1)
    #padded_decoder_input is of dimension (batch_size, max_len)
    #padded_utters_id is of dimension (batch_size, max_utter, max_len)
    #print 'padded_target ', padded_target
    #print 'np.where(padded_target==end_symbol_index) ', np.where(padded_target==end_symbol_index)		
    decoder_seq_len = [-1]*batch_size
    row, col  = np.where(padded_target==end_symbol_index)
    for row_i, col_i in zip(row, col):
	decoder_seq_len[row_i] = col_i
    if -1 in decoder_seq_len:
	raise Exception('cannot find end symbol in training dialogue')
    decoder_seq_len = np.asarray(decoder_seq_len)
    decoder_seq_len = decoder_seq_len + 1 #???????? check if decoder_seq_len=decoder_seq_len+1 is required or not (YES CHECKED WILL BE REQUIRED)
    #decoder_seq_len is of dimension batch_size
    #print 'padded_utters_id.shape ',padded_utters_id.shape
    #print 'padded_image_rep.shape ',padded_image_rep.shape
    #print 'padded_target.shape ',padded_target.shape
    #print 'padded_decoder_input.shape ',padded_decoder_input.shape
    #print 'decoder_seq_len.shape ',decoder_seq_len.shape 		
    return padded_utters_id, padded_image_rep, padded_target, padded_decoder_input, decoder_seq_len
    
def get_batch_data(max_len, max_images, image_rep_size, max_utter, batch_size, data_dict):
    #get batch_text_dict, batch_image_dict, batch_target_dict from data_dict
    #data_dict is a batch_size sized list of zips(batch_text_dict, batch_image_dict, batch_target)
    data_dict = np.asarray(data_dict)
    #converting data dict from a multidimensional list to a numpy matrix in order to carry out the operations below
    batch_text_dict = data_dict[:,0]
    #batch_text_dict is a multidimensional list integers (word ids) of dimension batch_size * max_utter * max_len
    
    batch_image_dict = data_dict[:,1]
    #batch_image_dict is a multidimensional list of strings of dimension batch_size * max_utter * max_images
    
    batch_target = data_dict[:,2]
    #batch_target is a list of list of words ids of dimension batch_size * max_len
    if len(data_dict)%batch_size!=0:
        batch_text_dict, batch_image_dict, batch_target = check_padding(batch_text_dict, batch_image_dict, batch_target, max_len, max_images, max_utter, batch_size)
    	 
    batch_image_dict = [[[get_image_representation(entry_ijk, image_rep_size) for entry_ijk in data_dict_ij] for data_dict_ij in data_dict_i] for data_dict_i in batch_image_dict]
    #batch_image_dict is now transformed to a multidimensional list of image_representations of dimension batch_size * max_utter * max_images * image_rep_size
    
    padded_utters,padded_image_rep, padded_target, padded_decoder_input, decoder_seq_len=get_utter_seq_len(batch_text_dict, batch_image_dict, batch_target, max_len, max_images, image_rep_size, max_utter, batch_size)
    #padded_utters is of dim (batch_size, max_utter,  max_len)
    #padded_image_rep is of dim (batch_size, max_utter, max_images, image_rep_size)
    #padded_target is of dim (batch_size, max_len)
    #padded_decoder_input is of dim (batch_size, max_len)

    padded_weights=get_weights(padded_target, batch_size, max_len, decoder_seq_len)
    #padded_weights is of dim (batch_size, max_len)

    padded_utters, padded_image_rep, padded_target, padded_decoder_input, padded_weights = transpose_utterances(padded_utters, padded_image_rep, padded_target, padded_decoder_input, padded_weights)
    #after transposing, padded_utters is of dim (max_utter, max_len, batch_size)
    #after transposing, padded_image_rep is of dim (max_utter, max_images, batch_size, image_rep_size)
    #after transposing, padded_target is of dim (max_len, batch_size)
    #after transposing, padded_decoder_input is of dim (max_len, batch_size)
    #after transposing padded_weights is of dim (max_len, batch_size)
    #print ' padded_weights ', padded_weights[:,0]
    #print ' padded_weights ', padded_weights[:,1]
    #print 'decoder_seq_len ', decoder_seq_len		
    #print 'padded weights shape ', padded_weights.shape
    return padded_utters, padded_image_rep, padded_target, padded_decoder_input, padded_weights

def get_image_representation(image_filename, image_rep_size):
    image_filename = image_filename.strip()	
    if image_filename=="":
	return [0.]*image_rep_size
    #FOR ANNOY BASED INDEX
    try:	
	return annoyIndex.get_item_vector(annoyPkl[image_filename]) 
    except:
	return [0.]*image_rep_size			

    #FOR SAMPLE INDEX
    #return sample_image_index[image_filename]

def transpose_utterances(padded_utters, padded_image_rep, padded_target, padded_decoder_input, padded_weights):
    padded_transposed_utters = padded_utters.transpose((1,2,0))
    padded_transposed_image_rep = padded_image_rep.transpose((1,2,0,3))
    padded_transposed_target = padded_target.transpose((1,0))
    padded_transposed_decoder_input = padded_decoder_input.transpose((1,0))
    padded_transposed_weights = padded_weights.transpose((1,0))
    return padded_transposed_utters, padded_transposed_image_rep, padded_transposed_target, padded_transposed_decoder_input, padded_transposed_weights

def batch_padding_text(data_mat, max_len, max_utter, pad_size):
    empty_data = [start_symbol_index, end_symbol_index]+[pad_symbol_index]*(max_len-2)
    empty_data = [empty_data]*max_utter
    empty_data_mat = [empty_data]*pad_size
    data_mat=data_mat.tolist()	
    data_mat.extend(empty_data_mat)
    return data_mat

def batch_padding_image(data_mat, max_images, max_utter, pad_size):
    empty_data = ['']*max_images
    empty_data = [empty_data]*max_utter
    empty_data_mat = [empty_data]*pad_size
    data_mat = data_mat.tolist()
    data_mat.extend(empty_data_mat)
    return data_mat

def batch_padding_target_text(data_mat, max_len, pad_size):
    empty_data = [start_symbol_index, end_symbol_index]+[pad_symbol_index]*(max_len-2)
    empty_data = [empty_data]*pad_size
    data_mat=data_mat.tolist()
    data_mat.extend(empty_data)
    return data_mat

def check_padding(batch_text_dict, batch_image_dict, batch_target, max_len, max_images, max_utter, batch_size): 
    pad_size = batch_size - len(batch_target)%batch_size
    batch_text_dict = batch_padding_text(batch_text_dict, max_len, max_utter, pad_size)
    batch_image_dict = batch_padding_image(batch_image_dict, max_images, max_utter, pad_size)
    batch_target = batch_padding_target_text(batch_target, max_len, pad_size)
    return batch_text_dict, batch_image_dict, batch_target
        
def load_valid_test_target(data_dict):
    return np.asarray(data_dict)[:,2]


if __name__=="__main__":
    param = get_params(sys.argv[1])
    train_dir_loc = param['train_dir_loc']
    valid_dir_loc = param['valid_dir_loc']
    test_dir_loc = param['test_dir_loc'].replace('test','test_smallest')
    dump_dir_loc = param['dump_dir_loc']
    vocab_file = param['vocab_file']
    vocab_stats_file = param['vocab_stats_file']
    vocab_freq_cutoff = param['vocab_freq_cutoff']
    train_data_file = param['train_data_file']
    valid_data_file = param['valid_data_file']
    test_data_file = param['test_data_file'].replace('test','test_smallest')
    #print test_data_file
    #sys.exit(1)
    max_utter = param['max_utter']
    max_len = param['max_len']
    max_images = param['max_images']
    preparedata = PrepareData(max_utter, max_len, max_images, start_symbol_index, end_symbol_index, unk_symbol_index, pad_symbol_index, "text", cutoff=vocab_freq_cutoff)
    if os.path.isfile(vocab_file):
        print 'found existing vocab file in '+str(vocab_file)+', ... reading from there'  
    preparedata.prepare_data(test_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "test_smallest"), test_data_file)

