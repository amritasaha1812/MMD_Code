import math
import sys
import os
sys.path.append(os.getcwd())
import math
import pickle as pkl
import random
import os.path
import numpy as np
from params_test import * 
import nltk
from read_data_task2 import *
from hierarchy_model import *
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
import os

def feeding_dict(model, inputs_text, inputs_image, target_image_pos, target_image_negs, target_image_weights):
        feed_dict = {}
	for encoder_text_input, input_text in zip(model.encoder_text_inputs, inputs_text):
	    for encoder_text_input_i, input_text_i in zip(encoder_text_input, input_text):
		feed_dict[encoder_text_input_i] = input_text_i
	for encoder_img_input, input_image in zip(model.encoder_img_inputs, inputs_image):
	    for encoder_img_input_i, input_image_i in zip(encoder_img_input, input_image):
		feed_dict[encoder_img_input_i] = input_image_i
	feed_dict[model.target_img_pos] = target_image_pos 
        for target_img_neg, target_image_neg in zip(model.target_img_negs, target_image_negs):
		feed_dict[target_img_neg] = target_image_neg
	for image_weight, target_image_weight in zip(model.image_weights, target_image_weights):
		feed_dict[image_weight] = target_image_weight
        return feed_dict
    
def check_dir(param):
    '''Checks whether model and logs directtory exists, if not then creates both directories for saving best model and logs.
    Args:
        param:parameter dictionary.'''
    if not os.path.exists(param['logs_path']):
        os.makedirs(param['logs_path'])    
    if not os.path.exists(param['model_path']):
        os.makedirs(param['model_path'])

def get_test_loss(sess, model, batch_dict, param, loss, cosine_sim_pos, cosine_sim_negs):
    test_batch_text, test_batch_image, batch_image_target_pos, batch_image_target_negs, batch_mask_negs = get_batch_data(param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['max_negs'], param['batch_size'], batch_dict)
    feed_dict = feeding_dict(model, test_batch_text, test_batch_image, batch_image_target_pos, batch_image_target_negs, batch_mask_negs)
    batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs = sess.run([loss, cosine_sim_pos, cosine_sim_negs], feed_dict=feed_dict)
    return batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs


def perform_test(sess, model, param, loss, cosine_sim_pos, cosine_sim_negs):
    test_data = pkl.load(open(param['test_data_file'],'rb'))
    print "Test dialogues loaded"
    n_batches = int(math.ceil(float(len(test_data))/float(param['batch_size'])))
    sum_cosine_sim_pos = 0.0
    sum_cosine_sim_neg = 0.0	
    sum_test_loss = 0.0
    sum_mm_loss = 0.0
    sum_recall_1 = 0.0
    sum_total_1 = 0.0
    sum_recall_2 = 0.0
    sum_total_2 = 0.0
    sum_recall_3 = 0.0
    sum_total_3 = 0.0
    sum_recall_4 = 0.0
    sum_total_4 = 0.0
    count_test = 0
    for i in range(n_batches):
        batch_dict = test_data[i*param['batch_size']:(i+1)*param['batch_size']]
        batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs = get_test_loss(sess, model, batch_dict, param, loss, cosine_sim_pos, cosine_sim_negs)
	batch_test_loss, count_batch, recall_1, total_1, recall_2, total_2, recall_3, total_3, recall_4, total_4 = get_loss(param, batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs)
	batch_cosine_sim_negs = np.asarray(batch_cosine_sim_negs)
	avg_batch_cosine_sim_negs = np.average(batch_cosine_sim_negs, axis=0)			
	sum_cosine_sim_pos = sum_cosine_sim_pos + np.sum(batch_cosine_sim_pos)
	sum_cosine_sim_neg = sum_cosine_sim_neg + np.sum(avg_batch_cosine_sim_negs)
	sum_test_loss = sum_test_loss + batch_test_loss
	sum_mm_loss = sum_mm_loss + batch_mm_loss
	count_test = count_test + count_batch
	sum_recall_1 = sum_recall_1 + recall_1
        sum_recall_2 = sum_recall_2 + recall_2
        sum_recall_3 = sum_recall_3 + recall_3
        sum_recall_4 = sum_recall_4 + recall_4
        sum_total_1 = sum_total_1 + total_1
	sum_total_2 = sum_total_2 + total_2
        sum_total_3 = sum_total_3 + total_3
        sum_total_4 = sum_total_4 + total_4
	#if count_batch>0 and sum_test_loss>0:
	#	print ('for batch %d, batch test loss = %.6f' %(i, sum_test_loss/float(count_batch)))
	try: 
		if total_1>0 and total_2>0 and total_3>0 and total_4>0:
			print ('recall@1=%.6f recall@2=%.6f recall@3=%.6f recall@4=%.6f' %(recall_1/total_1, recall_2/total_2, recall_3/total_3, recall_4/total_4))	
	except:
		continue	
	#break
    avg_test_loss = sum_test_loss/float(count_test)	
    avg_mm_loss = sum_mm_loss/float(len(test_data))
    avg_cosine_sim_pos = sum_cosine_sim_pos/float(len(test_data))
    avg_cosine_sim_neg = sum_cosine_sim_neg/float(len(test_data))
    print ('average test loss is = %.6f/%.6f = %.6f   average MM loss =%.6f (cosine-dist-pos=%.6f cosine-dist-neg=%.6f) ' %(sum_test_loss, count_test, avg_test_loss, avg_mm_loss, avg_cosine_sim_pos, avg_cosine_sim_neg))
    avg_recall_1 = sum_recall_1 / sum_total_1
    avg_recall_2 = sum_recall_2 / sum_total_2
    avg_recall_3 = sum_recall_3 / sum_total_3
    avg_recall_4 = sum_recall_4 / sum_total_4
    print ('Recall@1 = %.6f/%.6f = %.6f' %(sum_recall_1, sum_total_1, avg_recall_1))
    print ('Recall@2 = %.6f/%.6f = %.6f' %(sum_recall_2, sum_total_2, avg_recall_2))
    print ('Recall@3 = %.6f/%.6f = %.6f' %(sum_recall_3, sum_total_3, avg_recall_3))
    print ('Recall@4 = %.6f/%.6f = %.6f' %(sum_recall_4, sum_total_4, avg_recall_4))
    sys.stdout.flush() 

def get_loss(param, cosine_sim_pos, cosine_sim_negs, mask_negs):
    cosine_diff = [(x-cosine_sim_pos).tolist() for x in cosine_sim_negs]
    cosine_diff = np.asarray(cosine_diff)
    cosine_diff = np.transpose(cosine_diff)
    cosine_sim_pos = np.transpose(cosine_sim_pos)
    cosine_sim_negs = np.transpose(cosine_sim_negs)
    for i in range(param['batch_size']):
    	print cosine_sim_pos[i], cosine_sim_negs[i]
    mask_negs = np.transpose(mask_negs)
    if cosine_diff.shape !=  mask_negs.shape:
	raise Exception('cosine_diff.shape !=  mask_negs.shape', cosine_diff.shape , mask_negs.shape)		
    if cosine_diff.shape[0]!=param['batch_size'] or cosine_diff.shape[1]!=param['max_negs']:
	raise Exception('cosine diff.shape != ',param['batch_size'],param['max_negs'], cosine_diff.shape)
    #count_incorrect = sum([1 for x in (cosine_diff*mask_negs).flatten() if x>0])
    count_incorrect = sum([1 for x in cosine_diff.flatten() if x>0])	
    count = mask_negs.sum()
    cosine_diff = cosine_diff*mask_negs
    #np.save(param['dir']+'/cosine_diff.npy',cosine_diff)
    #np.save(param['dir']+'/mask_negs.npy',mask_negs)
    count_incorrect_list = np.where(cosine_diff>0)[0]
    count_incorrect_list = [(count_incorrect_list==i).sum() for i in range(param['batch_size'])]
    count_incorrect_list = np.asarray(count_incorrect_list)
    total = np.sum(mask_negs, axis=1)
    recall_1 = np.logical_and(total>0,count_incorrect_list==0).sum()
    recall_2 = np.logical_and(total>1,count_incorrect_list<=1).sum() 
    recall_3 = np.logical_and(total>2,count_incorrect_list<=2).sum()
    recall_4 = np.logical_and(total>3,count_incorrect_list<=3).sum()
    total_1 = (total>0).sum()
    total_2 = (total>1).sum()
    total_3 = (total>2).sum()
    total_4 = (total>3).sum()
    #if count_incorrect > count:
    #	raise Exception('count_incorrect > count')		
    return float(count_incorrect), float(count), float(recall_1), float(total_1), float(recall_2), float(total_2), float(recall_3), float(total_3), float(recall_4), float(total_4)
    	
def get_sum(x):
    return np.sum(np.asarray(x))     

def run_test(param):
    vocab = pkl.load(open(param['vocab_file'],"rb"))
    vocab_size = len(vocab)
    param['decoder_words'] = vocab_size
    print 'valid target sentence list loaded'
    print 'writing terminal output to file'
    f_out = open(param['terminal_op'],'w')
    sys.stdout=f_out
    check_dir(param)
    load_image_representation(param['image_annoy_dir'])
    with tf.Graph().as_default():
        model = Hierarchical_seq_model('image', param['text_embedding_size'], param['image_embedding_size'], param['image_rep_size'], param['cell_size'], param['cell_type'], param['batch_size'], param['learning_rate'], param['max_len'], param['max_utter'], param['max_images'], param['max_negs'], param['patience'], param['decoder_words'], param['max_gradient_norm'], param['activation'], param['output_activation'])   
        model.create_placeholder()
        output_representation = model.inference()
        loss, cosine_sim_pos, cosine_sim_negs = model.loss_task_image(output_representation)
        sys.stdout.flush()
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()
	old_model_file = None
	if len(os.listdir(param['model_path']))>0:
            old_model_file = None
            try:
                  checkpoint_file_lines = open(param['model_path']+'/checkpoint').readlines()
                  for line in checkpoint_file_lines:
                      if line.startswith('model_checkpoint_path:'):
                           old_model_file = os.path.join(param['model_path'],line.split('"')[1])
            except:
                  old_model_file = None
        else:
            old_model_file = None
        if old_model_file is not None:
            print "best model exists.. restoring from that point"
            saver.restore(sess, old_model_file)
        else:
            raise Exception('cannot find model .. exiting')
            sess.run(init)
        print 'Evaluating on test data'
	perform_test(sess, model, param, loss, cosine_sim_pos, cosine_sim_negs)
    f_out.close()
    
    

def main():
    if len(sys.argv)==3:
    	param = get_params(sys.argv[1], sys.argv[2], None)
    else:
	param = get_params(sys.argv[1], sys.argv[2], sys.argv[3])
    print param
    if os.path.exists(param['train_data_file']) and os.path.exists(param['valid_data_file']) and os.path.exists(param['test_data_file']):
        print 'dictionary already exists'
        sys.stdout.flush()
    else:
        get_dialog_dict(param, True)
        print 'dictionary formed'
        sys.stdout.flush()
    run_test(param)
    
if __name__=="__main__":
    main()            



                    


    

































