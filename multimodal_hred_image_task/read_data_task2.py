import itertools
import sys
import numpy as np
import cPickle as pkl
import os
from prepare_data_for_hred import PrepareData
start_symbol_index = 0
end_symbol_index = 1
unk_symbol_index = 2
pad_symbol_index = 3
from annoy import AnnoyIndex

annoyIndex = None
def load_image_representation(image_annoy_dir):
    annoyInde = AnnoyIndex(4096, metric='euclidean')
    annoyIndex.load(image_annoy_dir+'/annoy.ann')
    annoyPkl = pkl.load(open(image_annoy_dir+'/ImageUrlToIndex.pkl'))

def get_dialog_dict(param, is_test=False):
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
    max_negs = param['max_negs']
    if 'test_state' in param:
        test_state = param['test_state']
    else:
        test_state = None	
    preparedata = PrepareData(max_utter, max_len, max_images, max_negs, start_symbol_index, end_symbol_index, unk_symbol_index, pad_symbol_index, "image", cutoff=vocab_freq_cutoff)
    if os.path.isfile(vocab_file):
	print 'found existing vocab file in '+str(vocab_file)+', ... reading from there'
    if not is_test:	
	    preparedata.prepare_data(train_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "train"), train_data_file, True, False, None)
	    preparedata.prepare_data(valid_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "valid"), valid_data_file, False, False, None)
    if test_state is not None:
	    preparedata.prepare_data(test_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc+"/test_data_file_state/", "test_"+test_state), test_data_file, False, True, test_state)
    else:
	    preparedata.prepare_data(test_dir_loc, vocab_file, vocab_stats_file, os.path.join(dump_dir_loc, "test"), test_data_file, False, True, test_state)


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
    #padded_target = np.asarray([xi  for xi in dialogue_target])
    #padded_target is of dimension (batch_size, max_len)
    #pad_to_target = np.reshape(np.asarray([pad_symbol_index]*batch_size), (batch_size, 1))
    #pad_to_target is of dimension (batch_size, 1)
    #print 'padded_target[:,:-1].shape ', padded_target[:,:-1].shape
    #print 'pad_to_target.shape ',pad_to_target.shape
    #print 'padded_target[:,:-1].shape ',padded_target[:,:-1].shape
    #padded_decoder_input = np.concatenate((pad_to_target, padded_target[:,:-1]), axis=1)
    #padded_decoder_input is of dimension (batch_size, max_len)
    #padded_utters_id is of dimension (batch_size, max_utter, max_len)
    #print 'padded_target ', padded_target
    #print 'np.where(padded_target==end_symbol_index) ', np.where(padded_target==end_symbol_index)		
    #decoder_seq_len = [-1]*batch_size
    #row, col  = np.where(padded_target==end_symbol_index)
    #for row_i, col_i in zip(row, col):
    #	decoder_seq_len[row_i] = col_i
    #if -1 in decoder_seq_len:
    #	raise Exception('cannot find end symbol in training dialogue')
    #decoder_seq_len = np.asarray(decoder_seq_len)
    #decoder_seq_len = decoder_seq_len + 1 #???????? check if decoder_seq_len=decoder_seq_len+1 is required or not (YES CHECKED WILL BE REQUIRED)
    #decoder_seq_len is of dimension batch_size
    #print 'padded_utters_id.shape ',padded_utters_id.shape
    #print 'padded_image_rep.shape ',padded_image_rep.shape
    #print 'padded_target.shape ',padded_target.shape
    #print 'padded_decoder_input.shape ',padded_decoder_input.shape
    #print 'decoder_seq_len.shape ',decoder_seq_len.shape 		
    #return padded_utters_id, padded_image_rep, padded_target, padded_decoder_input, decoder_seq_len
    
def get_batch_data(max_len, max_images, image_rep_size, max_utter, max_negs, batch_size, data_dict):
    #get batch_text_dict, batch_image_dict, batch_target_dict from data_dict
    #data_dict is a batch_size sized list of zips(batch_text_dict, batch_image_dict, batch_target)
    data_dict = np.asarray(data_dict)
    #converting data dict from a multidimensional list to a numpy matrix in order to carry out the operations below
    #batch_index_dict is a list of integers of dimension batch_size (will be used later to match the positive and negative data)
	
    batch_text_dict = data_dict[:,0]
    #batch_text_dict is a multidimensional list integers (word ids) of dimension batch_size * max_utter * max_len
    
    batch_image_dict = data_dict[:,1]
    #batch_image_dict is a multidimensional list of strings of dimension batch_size * max_utter * max_images
    
    batch_target_pos = data_dict[:,2]
    batch_target_pos = list(itertools.chain(*batch_target_pos.tolist()))	

    batch_target_negs = data_dict[:,3]
    lens = [len(x) for x in batch_target_negs]
    batch_mask_negs = data_dict[:,4]
    #batch_target is a list of strings of dimension batch_size
    if len(data_dict)%batch_size!=0:
        batch_text_dict, batch_image_dict, batch_target_pos, batch_target_negs, batch_mask_negs = check_padding(batch_text_dict, batch_image_dict, batch_target_pos, batch_target_negs, batch_mask_negs, max_len, max_images, max_utter, max_negs, batch_size)
    	 
    batch_image_dict = [[[get_image_representation(entry_ijk, image_rep_size) for entry_ijk in data_dict_ij] for data_dict_ij in data_dict_i] for data_dict_i in batch_image_dict]
    #batch_image_dict is now transformed to a multidimensional list of image_representations of dimension batch_size * max_utter * max_images * image_rep_size
    batch_target_pos = [get_image_representation(entry_i, image_rep_size) for entry_i in batch_target_pos]

    batch_target_negs = [[get_image_representation(entry_ij, image_rep_size) for entry_ij in batch_target_neg] for batch_target_neg in batch_target_negs]
    
    padded_utters = np.asarray([[xij for xij in dialogue_i] for dialogue_i in batch_text_dict])
    padded_image_rep = np.asarray([[xij for xij in dialogue_i] for dialogue_i in batch_image_dict])
    padded_target_negs = np.asarray([[xij for xij in dialogue_i] for dialogue_i in batch_target_negs])
    padded_mask_negs = np.asarray([[xij for xij in dialogue_i] for dialogue_i in batch_mask_negs])
    #padded_utters,padded_image_rep, padded_target, padded_decoder_input, decoder_seq_len=get_utter_seq_len(batch_text_dict, batch_image_dict, batch_target, max_len, max_images, image_rep_size, max_utter, batch_size)
    #padded_utters is of dim (batch_size, max_utter,  max_len)
    #padded_image_rep is of dim (batch_size, max_utter, max_images, image_rep_size)
    #padded_target_pos is of dim (batch_size)
    #padded_target_negs is of dim (batch_size, max_negs, image_rep_size)
    #padded_decoder_input is of dim (batch_size, max_len)

    #padded_weights=get_weights(padded_target, batch_size, max_len, decoder_seq_len)
    #padded_weights is of dim (batch_size, max_len)

    padded_utters, padded_image_rep, padded_target_negs, padded_mask_negs = transpose_utterances(padded_utters, padded_image_rep, padded_target_negs, padded_mask_negs)
    #after transposing, padded_utters is of dim (max_utter, max_len, batch_size)
    #after transposing, padded_image_rep is of dim (max_utter, max_images, batch_size, image_rep_size)
    #after transposing, batch_target_pos is of dim (batch_size)
    #after transposing, batch_target_negs is of dim (max_negs, batch_size, image_rep_size)
    #after transposing, padded_decoder_input is of dim (max_len, batch_size)
    #after transposing padded_weights is of dim (max_len, batch_size)
    #print ' padded_weights ', padded_weights[:,0]
    #print ' padded_weights ', padded_weights[:,1]
    #print 'decoder_seq_len ', decoder_seq_len		
    #print 'padded weights shape ', padded_weights.shape
    return padded_utters, padded_image_rep, batch_target_pos, padded_target_negs, padded_mask_negs

def get_image_representation(image_filename, image_rep_size):
    image_filename = image_filename.strip() 
    #if image_filename=="" or image_filename not in sample_image_index:
    #	return [0.]*image_rep_size
    #FOR ANNOY BASED INDEX
    try:	
    	return annoyIndex.get_item_vector(annoyPkl[image_filename]) 
    except:
    	return [0.]*image_rep_size			

    #FOR SAMPLE INDEX
    #return sample_image_index[image_filename]

def transpose_utterances(padded_utters, padded_image_rep, padded_target_negs, padded_mask_negs):
    try: 	
	padded_transposed_utters = padded_utters.transpose((1,2,0))
    except:
	raise Exception (' error transposing padded_utters ', padded_utters.shape)
    try:	
    	padded_transposed_image_rep = padded_image_rep.transpose((1,2,0,3))
    except:
        raise Exception (' error transposing padded_image_rep ',padded_image_rep.shape)
    try:
    	padded_transposed_target_negs = padded_target_negs.transpose((1,0,2))
    except:
	raise Exception (' error transposing padded_target_negs ', padded_target_negs.shape)
    try:
    	padded_transposed_mask_negs = padded_mask_negs.transpose((1,0))
    except:
        raise Exception (' error transposing padded_mask_negs ', padded_mask_negs.shape)	
    return padded_transposed_utters, padded_transposed_image_rep, padded_transposed_target_negs, padded_transposed_mask_negs

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


def batch_padding_target_pos(data_mat, pad_size):
    empty_data = ['']*pad_size
    data_mat.extend(empty_data)
    return data_mat

def batch_padding_target_negs(data_mat, max_negs, pad_size):
    empty_data = ['']*max_negs
    empty_data = [empty_data]*pad_size
    data_mat = data_mat.tolist()
    data_mat.extend(empty_data)
    return data_mat

def batch_padding_mask_negs(data_mat, max_negs, pad_size):
    empty_data = [0.]*max_negs
    empty_data = [empty_data]*pad_size
    data_mat = data_mat.tolist()
    data_mat.extend(empty_data)
    return data_mat

def check_padding(batch_text_dict, batch_image_dict, batch_target_pos_dict, batch_target_negs_dict, batch_mask_negs_dict, max_len, max_images, max_utter, max_negs, batch_size): 
    pad_size = batch_size - len(batch_text_dict)%batch_size
    batch_text_dict = batch_padding_text(batch_text_dict, max_len, max_utter, pad_size)
    batch_image_dict = batch_padding_image(batch_image_dict, max_images, max_utter, pad_size)
    batch_target_pos_dict = batch_padding_target_pos(batch_target_pos_dict, pad_size)
    batch_target_negs_dict = batch_padding_target_negs(batch_target_negs_dict, max_negs, pad_size)
    batch_mask_negs_dict = batch_padding_mask_negs(batch_mask_negs_dict, max_negs, pad_size)
    return batch_text_dict, batch_image_dict, batch_target_pos_dict, batch_target_negs_dict, batch_mask_negs_dict

