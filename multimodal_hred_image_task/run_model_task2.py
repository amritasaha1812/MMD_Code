import math
import sys
import os
sys.path.append(os.getcwd())
import math
import pickle as pkl
import random
import os.path
import params
import numpy as np
from params import * 
import nltk
from read_data_task2 import *
from hierarchy_model import *

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

def get_test_op(sess, model, batch_dict, param, output_representation, loss, cosine_sim_pos, cosine_sim_negs):
    test_batch_text, test_batch_image, batch_image_target_pos, batch_image_target_negs, batch_mask_negs = get_batch_data(param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['max_negs'], param['batch_size'], batch_dict)
    feed_dict = feeding_dict(model, test_batch_text, test_batch_image, batch_image_target_pos, batch_image_target_negs, batch_mask_negs)
    batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs = sess.run([loss, cosine_sim_pos, cosine_sim_negs], feed_dict=feed_dict)
    return batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs


def write_to_file(pred_op, true_op):
    pred_file = ''
    true_file = ''
    with open(true_file, 'w') as f_true:
        for true_sentence in true_op:
            f_true.write(true_sentence.strip()+'\n')

    with open(pred_file, 'w') as f_pred:
        for pred_sentence in pred_op:
            f_pred.write(pred_sentence.strip()+'\n')
    print 'Test (true and predicted) output written to corresponding files'
    
def perform_test(sess, model, saver, model_file, get_pred_sentence, param, loss, cosine_sim_pos, cosine_sim_negs):
    print 'reading model from  modelfile'
    saver.restore(sess, model_file)
    test_data = pkl.load(open(param['test_data_file']),'rb')
    print "Test dialogues loaded"
    n_batches = math.ceil(float(len(test_data))/float(param['batch_size']))
    sum_cosine_sim_pos = 0.0
    sum_cosine_sim_neg = 0.0	
    sum_test_loss = 0.0
    sum_mm_loss = 0.0
    count_test = 0
    for i in range(n_batches):
        batch_dict = test_data[i*param['batch_size']:(i+1)*param['batch_size']]
        batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs = get_test_op(sess, model, batch_dict, param, loss, cosine_sim_pos, cosine_sim_negs)
	batch_test_loss, count_batch = get_loss(batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs)
	batch_cosine_sim_negs = np.asarray(batch_cosine_sim_negs)
	avg_batch_cosine_sim_negs = np.average(batch_cosine_sim_negs, axis=0)			
	sum_cosine_sim_pos = sum_cosine_sim_pos + np.sum(batch_cosine_sim_pos)
	sum_cosine_sim_neg = sum_cosine_sim_neg + np.sum(avg_batch_cosine_sim_negs)
	sum_test_loss = sum_test_loss + batch_test_loss
	sum_mm_loss = sum_mm_loss + batch_mm_loss
	count_test = count_test + count_batch
    avg_test_loss = sum_test_loss/float(count_test)	
    avg_mm_loss = sum_mm_loss/float(len(test_data))
    avg_cosine_sim_pos = sum_cosine_sim_pos/float(len(test_data))
    avg_cosine_sim_neg = sum_cosine_sim_neg/float(len(test_data))
    print ('average test loss is =%.6f   average MM loss =%.6f (cosine-dist-pos=%.6f cosine-dist-neg=%.6f) ' %(avg_test_loss, avg_mm_loss, avg_cosine_sim_pos, avg_cosine_sim_neg))
    sys.stdout.flush() 

def get_loss(cosine_sim_pos, cosine_sim_negs, mask_negs):
    cosine_diff = [(x-cosine_sim_pos).tolist() for x in cosine_sim_negs]
    cosine_diff = np.asarray(cosine_diff)
    if cosine_diff.shape !=  mask_negs.shape:
	raise Exception('cosine_diff.shape !=  mask_negs.shape', cosine_diff.shape , mask_negs.shape)		
    count_incorrect = sum([1 for x in (cosine_diff*mask_negs).flatten() if x>0])
    count = mask_negs.sum()
    if count_incorrect > count:
	raise Exception('count_incorrect > count')		
    return float(count_incorrect), float(count)
    	
def run_training(param):

    def get_train_loss(model, batch_dict, step):
        train_batch_text, train_batch_image, batch_image_target_pos, batch_image_target_negs, batch_mask_negs = get_batch_data(param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['max_negs'], param['batch_size'], batch_dict)
	feed_dict = feeding_dict(model, train_batch_text, train_batch_image, batch_image_target_pos, batch_image_target_negs, batch_mask_negs)
	batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs, _ = sess.run([loss, cosine_sim_pos, cosine_sim_negs, train_op], feed_dict=feed_dict)	
	'''
	if step % param['show_grad_freq'] == 0:
		grad_vals = sess.run(gradients, feed_dict=feed_dict)
		var_to_grad = {}
		for grad_val, var in zip(grad_vals, gradients):
			if type(grad_val).__module__ == np.__name__:
				var_to_grad[var.name] = grad_val
				sys.stdout.flush()
				print 'var.name ', var.name, 'shape(grad) ',grad_val.shape, 'mean(grad) ',np.mean(grad_val)
				sys.stdout.flush()	
	'''
	#sys.stdout.flush()      
	#print 'batch_cosine_sim_pos' ,batch_cosine_sim_pos
	#print 'batch_cosine_sim_negs ', batch_cosine_sim_negs
	#sys.stdout.flush()
	return batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs

    def get_valid_loss(model, batch_dict):
	valid_batch_text, valid_batch_image, batch_image_target_pos, batch_image_target_negs, batch_mask_negs = get_batch_data(param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['max_negs'], param['batch_size'], batch_dict)
        feed_dict = feeding_dict(model, valid_batch_text, valid_batch_image, batch_image_target_pos, batch_image_target_negs, batch_mask_negs)
        batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs = sess.run([loss, cosine_sim_pos, cosine_sim_negs], feed_dict=feed_dict)
        return batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs

    def get_sum(x):
        return np.sum(np.asarray(x))     

    def evaluate(model, epoch, step, valid_data):
        print 'Validation started'
        sys.stdout.flush()
        sum_valid_loss = 0.0
        sum_mm_loss = 0.0 	
	sum_cosine_sim_pos = 0.0
	sum_cosine_sim_neg = 0.0	
        count_valid = 0
        n_batches = int(math.ceil(float(len(valid_data))/float(param['batch_size']))) 
        for i in range(n_batches):
	    batch_dict = valid_data[i*param['batch_size']:(i+1)*param['batch_size']]
            batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs = get_valid_loss(model, batch_dict)
	    batch_valid_loss, count_batch = get_loss(batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs)
            batch_cosine_sim_negs = np.asarray(batch_cosine_sim_negs)
            avg_batch_cosine_sim_negs = np.average(batch_cosine_sim_negs, axis=0)
            sum_cosine_sim_pos = sum_cosine_sim_pos + np.sum(batch_cosine_sim_pos)
            sum_cosine_sim_neg = sum_cosine_sim_neg + np.sum(avg_batch_cosine_sim_negs)
            sum_valid_loss = sum_valid_loss + batch_valid_loss
            sum_mm_loss = sum_mm_loss + batch_mm_loss
            count_valid = count_valid + count_batch
        avg_mm_loss = sum_mm_loss/float(len(valid_data))
        avg_cosine_sim_pos = sum_cosine_sim_pos/float(len(valid_data))
        avg_cosine_sim_neg = sum_cosine_sim_neg/float(len(valid_data))
	avg_valid_loss = (float(sum_valid_loss)/float(count_valid))
        return avg_valid_loss, avg_mm_loss, avg_cosine_sim_pos, avg_cosine_sim_neg

    train_data = pkl.load(open(param['train_data_file']))
    print 'Train dialogue dataset loaded'
    sys.stdout.flush()
    valid_data = pkl.load(open(param['valid_data_file']))
    print 'Valid dialogue dataset loaded'
    sys.stdout.flush()
    vocab = pkl.load(open(param['vocab_file'],"rb"))
    vocab_size = len(vocab)
    param['decoder_words'] = vocab_size
    print 'valid target sentence list loaded'
    print 'writing terminal output to file'
    f_out = open(param['terminal_op'],'w')
    sys.stdout=f_out
    check_dir(param)
    load_image_representation(param['image_annoy_dir'])
    n_batches = int(math.ceil(float(len(train_data))/float(param['batch_size'])))
    print 'number of batches ', n_batches, 'len train_data', len(train_data), ' batch size' , param['batch_size']
    model_file = os.path.join(param['model_path'],"best_model")
    with tf.Graph().as_default():
        model = Hierarchical_seq_model('image', param['text_embedding_size'], param['image_embedding_size'], param['image_rep_size'], param['cell_size'], param['cell_type'], param['batch_size'], param['learning_rate'], param['max_len'], param['max_utter'], param['max_images'], param['max_negs'], param['patience'], param['decoder_words'], param['max_gradient_norm'], param['activation'], param['output_activation'])   
        model.create_placeholder()
        output_representation = model.inference()
        loss, cosine_sim_pos, cosine_sim_negs = model.loss_task_image(output_representation)
        train_op, gradients = model.train(loss)
        print "model created"
        sys.stdout.flush()
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        if os.path.isfile(model_file):
            print "best model exists.. restoring from that point"
            saver.restore(sess, model_file)
        else:
            print "initializing fresh variables"
            sess.run(init)
        best_valid_loss = float("inf")
        best_valid_epoch=0
        all_var = tf.all_variables()
        print 'printing all' , len(all_var),' TF variables:'
        for var in all_var:
            print var.name, var.get_shape()
        print 'training started'
        sys.stdout.flush()
	last_avg_mm_loss = None
	overall_step_count = 0
        for epoch in range(param['max_epochs']):
	    r = random.random()
            random.shuffle(train_data, lambda : r)
	    sum_cosine_sim_pos = 0.0
            sum_cosine_sim_neg = 0.0
            count_train = 0
	    sum_train_loss = 0.0
	    sum_mm_loss = 0.0		
            for i in range(n_batches):
		overall_step_count = overall_step_count + 1
	        train_batch_dict=train_data[i*param['batch_size']:(i+1)*param['batch_size']]
                batch_mm_loss, batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs = get_train_loss(model, train_batch_dict, overall_step_count)
		batch_train_loss, count_batch = get_loss(batch_cosine_sim_pos, batch_cosine_sim_negs, batch_mask_negs)
                batch_cosine_sim_negs = np.asarray(batch_cosine_sim_negs)
                avg_batch_cosine_sim_negs = np.average(batch_cosine_sim_negs, axis=0)
		sum_batch_cosine_sim_pos = np.sum(batch_cosine_sim_pos)
		sum_batch_cosine_sim_neg = np.sum(avg_batch_cosine_sim_negs)
                sum_cosine_sim_pos = sum_cosine_sim_pos + sum_batch_cosine_sim_pos
          	sum_cosine_sim_neg = sum_cosine_sim_neg + sum_batch_cosine_sim_neg
                sum_train_loss = sum_train_loss + batch_train_loss
                sum_mm_loss = sum_mm_loss + batch_mm_loss
                count_train = count_train + count_batch
		if overall_step_count%param['print_train_freq']==0:
	                print('Epoch  %d Step %d average train loss is =%.6f   average MM loss =%.6f (cosine-dist-pos=%.6f cosine-dist-neg=%.6f) over %d instances' %(epoch, i, batch_train_loss/float(count_batch), batch_mm_loss/float(len(train_batch_dict)), sum_batch_cosine_sim_pos/float(len(train_batch_dict)), sum_batch_cosine_sim_neg/float(len(train_batch_dict)), len(train_batch_dict)))
                	sys.stdout.flush()
                if overall_step_count>0 and overall_step_count%param['valid_freq']==0:
                    avg_valid_loss, avg_mm_loss, avg_cosine_sim_pos, avg_cosine_sim_neg = evaluate(model, epoch, i, valid_data)
                    print ('Epoch %d Step %d average valid loss is =%.6f   average MM loss =%.6f (cosine-dist-pos=%.6f cosine-dist-neg=%.6f) ' %(epoch, i, avg_valid_loss, avg_mm_loss, avg_cosine_sim_pos, avg_cosine_sim_neg))
                    sys.stdout.flush()
                    if best_valid_loss>avg_valid_loss:
                        saver.save(sess, model_file)
                        best_valid_loss=avg_valid_loss
                    else:
                        continue
	    avg_mm_loss = sum_mm_loss/float(len(train_data))
	    avg_train_loss = sum_train_loss/float(count_train)
	    avg_cosine_sim_pos = sum_cosine_sim_pos/float(len(train_data))
	    avg_cosine_sim_neg = sum_cosine_sim_neg/float(len(train_data))
            print ('Epoch %d of training is completed ... average train loss is =%.6f   average MM loss =%.6f (cosine-dist-pos=%.6f cosine-dist-neg=%.6f) ' %(epoch, avg_train_loss, avg_mm_loss, avg_cosine_sim_pos, avg_cosine_sim_neg))
	    sys.stdout.flush()	
	    if last_avg_mm_loss is not None and avg_mm_loss > last_avg_mm_loss:
		diff = avg_mm_loss - last_avg_mm_loss
		if diff>param['train_loss_incremenet_tolerance']:	
			print 'WARNING: training MM loss (%.6f) has increased by %.6f since last epoch, has exceed tolerance of %f ' %(avg_mm_loss, diff, param['train_loss_incremenet_tolerance'])
		else:
			print 'WARNING: training MM loss (%.6f) has increased by %.6f since last epoch, but still within tolerance of %f ' %(avg_mm_loss, diff, param['train_loss_incremenet_tolerance'])	
	    last_avg_mm_loss = avg_mm_loss    		
            sys.stdout.flush()
        print 'Training over'
        print 'Evaluating on test data'
    f_out.close()
    
    

def main():
    data_dir = ''
    param = get_params(data_dir, sys.argv[1])
    print param
    if os.path.exists(param['train_data_file']) and os.path.exists(param['valid_data_file']) and os.path.exists(param['test_data_file']):
        print 'dictionary already exists'
        sys.stdout.flush()
    else:
        get_dialog_dict(param)
        print 'dictionary formed'
        sys.stdout.flush()
    run_training(param)
    
if __name__=="__main__":
    main()            



                    


    

































