import math
import sys
import os
sys.path.append(os.getcwd())
import math
import pickle as pkl
import random
import os.path
import params_test
import numpy as np
from params_test import *
import nltk
from read_data_task1 import *
from hierarchy_model_text import *

def feeding_dict(model, inputs_text, inputs_image, target_text, decoder_text_inputs, text_weights, feed_prev):
        feed_dict = {}
	for encoder_text_input, input_text in zip(model.encoder_text_inputs, inputs_text):
	    for encoder_text_input_i, input_text_i in zip(encoder_text_input, input_text):
		feed_dict[encoder_text_input_i] = input_text_i
	for encoder_img_input, input_image in zip(model.encoder_img_inputs, inputs_image):
	    for encoder_img_input_i, input_image_i in zip(encoder_img_input, input_image):
		feed_dict[encoder_img_input_i] = input_image_i
        for model_target_text_i, target_text_i in zip(model.target_text, target_text):
            feed_dict[model_target_text_i] = target_text_i
        for model_decoder_text_input, decoder_text_input in zip(model.decoder_text_inputs, decoder_text_inputs):
            feed_dict[model_decoder_text_input] = decoder_text_input
        for model_text_weight, text_weight in zip(model.text_weights, text_weights):
            feed_dict[model_text_weight] = text_weight
        feed_dict[model.feed_previous] = feed_prev    
        return feed_dict
    
def check_dir(param):
    '''Checks whether model and logs directtory exists, if not then creates both directories for saving best model and logs.
    Args:
        param:parameter dictionary.'''
    if not os.path.exists(param['logs_path']):
        os.makedirs(param['logs_path'])    
    if not os.path.exists(param['model_path']):
        os.makedirs(param['model_path'])

def get_test_loss(sess, model, batch_dict, param, logits, losses):
    test_batch_text, test_batch_image, batch_text_target, batch_decoder_input, batch_text_weight = get_batch_data(param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['batch_size'], batch_dict)
    feed_dict = feeding_dict(model, test_batch_text, test_batch_image, batch_text_target, batch_decoder_input, batch_text_weight, True)
    loss, dec_op = sess.run([losses, logits], feed_dict=feed_dict)
    return dec_op, loss
    
def write_to_file(pred_op, true_op, param):
    pred_file = param['test_output_dir']+'/pred.txt'
    true_file = param['test_output_dir']+'/true.txt'	
    with open(true_file, 'w') as f_true:
        for true_sentence in true_op:
	    #print true_sentence
            f_true.write(true_sentence.strip()+'\n')

    with open(pred_file, 'w') as f_pred:
        for pred_sentence in pred_op:
	    #print pred_sentence
            f_pred.write(pred_sentence.strip()+'\n')
    print 'Test (true and predicted) output written to corresponding files'
    
def perform_test(sess, model, param, logits, losses, vocab):
    test_data = pkl.load(open(param['test_data_file']))
    print "Test dialogues loaded"
    predicted_sentence = []
    test_loss = 0
    n_batches = int(math.ceil(float(len(test_data))/float(param['batch_size'])))
    test_text_targets = load_valid_test_target(test_data)
    true_sentence = []
    for i in range(n_batches):
	if i%10==0:
		print 'finished ',i, 'batches '
        batch_dict = test_data[i*param['batch_size']:(i+1)*param['batch_size']]
	batch_target_word_ids = test_text_targets[i*param['batch_size']:(i+1)*param['batch_size']]
	batch_target_sentences = map_id_to_word(batch_target_word_ids, vocab)
        test_op, batch_loss = get_test_loss(sess, model, batch_dict, param, logits, losses)
	sum_batch_loss = get_sum_batch_loss(batch_loss)
        test_loss = sum_batch_loss + test_loss
	batch_predicted_sentence, prob_predicted_words, prob_true_words = get_predicted_sentence(test_op, batch_target_word_ids, vocab)
        predicted_sentence.extend(batch_predicted_sentence)
	print_pred_true_op(batch_predicted_sentence, prob_predicted_words, prob_true_words, batch_target_sentences, i, batch_loss)
	true_sentence.extend(batch_target_sentences)
    test_predicted_sentence = predicted_sentence[0:len(test_text_targets)]
    test_true_sentence = true_sentence[0:len(test_text_targets)]
    write_to_file(test_predicted_sentence, test_true_sentence, param)
    print ('average test loss is =%.6f' %(float(test_loss)/float(len(test_data))))
    sys.stdout.flush() 


def get_sum_batch_loss(batch_loss):
        return np.sum(np.asarray(batch_loss))

        
def print_pred_true_op(pred_op, prob_pred, prob_true, true_op, step, batch_loss):
        for i in range(0,len(true_op),1):
	    print true_op[i], '\t', pred_op[i]
	    sys.stdout.flush()
	    '''
            print "true sentence in step "+str(step)+" is:"
            sys.stdout.flush()
            print true_op[i]
            print "\n"
            print "predicted sentence in step "+str(step)+" is:"
            sys.stdout.flush()
            print pred_op[i]
            print "\n"
	    print "prob of predicted words in step "+str(step)+" is:"
            sys.stdout.flush()
            print prob_pred[i]
            print "\n"
	    print "prob of true words in step "+str(step)+" is:"
            sys.stdout.flush()
            print prob_true[i]
	    print "\n"
            sys.stdout.flush()
            print "loss for the pair of true and predicted sentences", str(batch_loss[i])
            print "\n"
	    '''

def map_id_to_word(word_indices, vocab):
        sentence_list = []
        for sent in word_indices:
            word_list = []
            for word_index in sent:
		if word_index==start_symbol_index:
			continue
		if word_index==pad_symbol_index:
			continue	
		if word_index==end_symbol_index:
			break
                word = vocab[word_index]
                word_list.append(word)
            sentence_list.append(" ".join(word_list))
        return sentence_list

def get_predicted_sentence(valid_op, true_op, vocab):
        max_probs_index = []
	max_probs = []
	#
	#
        if true_op is not None:
		true_op = true_op.tolist()
		true_op = np.asarray(true_op).T.tolist()
		true_op_prob = []
	i=0
	for op in valid_op:
	    #print 'op. shape ', op.shape	
	    sys.stdout.flush()
            max_index = np.argmax(op, axis=1)
	    max_prob  = np.max(op, axis=1)
	    max_probs.append(max_prob)
            max_probs_index.append(max_index)
	    if true_op is not None:	
	            true_op_prob.append([v_ij[t_ij] for v_ij,t_ij in zip(op, true_op[i])])
		    i=i+1
        max_probs_index = np.transpose(max_probs_index)
	max_probs = np.transpose(max_probs)
	if true_op is not None:
		true_op_prob = np.asarray(true_op_prob)
		true_op_prob = np.transpose(true_op_prob)
		if true_op_prob.shape[0]!=max_probs.shape[0] and true_op_prob.shape[1]!=max_probs.shape[1]:
	                raise Exception('some problem shape of true_op_prob' , true_op_prob.shape)
	#max_probs is of shape batch_size, max_len
        pred_sentence_list = map_id_to_word(max_probs_index, vocab)
        return pred_sentence_list, max_probs, true_op_prob

def run_test(param):
    vocab = pkl.load(open(param['vocab_file'],"rb"))
    vocab_size = len(vocab)
    param['decoder_words'] = vocab_size
    print 'writing terminal output to file'
    f_out = open(param['terminal_op'],'w')
    sys.stdout=f_out
    check_dir(param)
    load_image_representation(param['image_annoy_dir'])
    with tf.Graph().as_default():
        model = Hierarchical_seq_model_text('text', param['text_embedding_size'], param['image_embedding_size'], param['image_rep_size'], param['cell_size'], param['cell_type'], param['batch_size'], param['learning_rate'], param['max_len'], param['max_utter'], param['max_images'], param['patience'], param['decoder_words'], param['max_gradient_norm'], param['activation'], param['output_activation'])   
        model.create_placeholder()
        logits = model.inference()
        losses = model.loss_task_text(logits)
        #train_op, gradients = model.train(losses)
        print "model created"
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
	    raise Exception('there is no model.. exiting')	
        print 'Evaluating on test data'
	perform_test(sess, model, param, logits, losses, vocab)
    f_out.close()
    
    

def main():
    data_dir=sys.argv[1]
    if len(sys.argv)==3:
	param = get_params(data_dir, sys.argv[2])
    else:
    	param = get_params(data_dir, sys.argv[2], sys.argv[3])
    if os.path.exists(param['train_data_file']) and os.path.exists(param['valid_data_file']) and os.path.exists(param['test_data_file']):
        print 'dictionary already exists'
        sys.stdout.flush()
    else:
        get_dialog_dict(param, is_test = True)
        print 'dictionary formed'
        sys.stdout.flush()
    run_test(param)
    
if __name__=="__main__":
    main()            



                    


    

































