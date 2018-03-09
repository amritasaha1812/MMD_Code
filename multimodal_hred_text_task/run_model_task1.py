import math
import sys
import os
sys.path.append(os.getcwd())
import math
import pickle as pkl
import random
import os.path
import numpy as np
from params import *
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

def get_test_op(sess, model, batch_dict, param, logits, losses):
    test_batch_text, test_batch_image, batch_text_target, batch_decoder_input, batch_text_weight = get_batch_data(param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['batch_size'], batch_dict)
    feed_dict = feeding_dict(model, test_batch_text, test_batch_image, batch_text_targets, batch_decoder_input, batch_text_weight, True)
    dec_op, loss = sess.run([logits, losses], feed_dict=feed_dict)
    sum_batch_loss = get_sum_batch_loss(loss) 
    return dec_op, sum_batch_loss
    
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
    
def perform_test(sess, model, saver, model_file, get_pred_sentence, param, logits, losses, vocab):
    print 'reading model from  modelfile'
    saver.restore(sess, model_file)
    test_data = pkl.load(open(param['test_data_file']),'rb')
    print "Test dialogues loaded"
    predicted_sentence = []
    test_loss = 0
    n_batches = len(test_data)/param['batch_size']
    test_text_targets = read_data_task1.load_valid_test_target(param['test_data_file'])
    for i in range(n_batches):
        batch_dict = test_data[i*param['batch_size']:(i+1)*param['batch_size']]
        test_op, sum_batch_loss = get_test_op(sess, model, batch_dict, param, logits, losses)
        test_loss = sum_batch_loss + test_loss
        predicted_sentence.append(get_predicted_sentence(test_op, None, vocab)[0])
    test_predicted_sentence = predicted_sentence[0:len(test_text_targets)]
    write_to_file(test_predicted_sentence, test_text_targets)
    print ('average test loss is =%.6f' %(float(test_loss)/float(len(test_data))))
    sys.stdout.flush() 

def run_training(param):

    def get_train_loss(model, batch_dict, step):
        train_batch_text, train_batch_image, batch_text_target, batch_decoder_input, batch_text_weight = get_batch_data(param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['batch_size'], batch_dict)
        if epoch<0:
            feed_dict = feeding_dict(model, train_batch_text, train_batch_image, batch_text_target, batch_decoder_input, batch_text_weight, False)
        else:
            feed_dict = feeding_dict(model, train_batch_text, train_batch_image, batch_text_target, batch_decoder_input, batch_text_weight, True)
        loss, dec_op, _ = sess.run([losses, logits, train_op], feed_dict=feed_dict)
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
        return loss, dec_op

    def get_valid_loss(model, batch_dict):
        valid_batch_text, valid_batch_image, batch_text_target, batch_decoder_input, batch_text_weight = get_batch_data(param['max_len'], param['max_images'], param['image_rep_size'], param['max_utter'], param['batch_size'], batch_dict)
        feed_dict = feeding_dict(model, valid_batch_text, valid_batch_image, batch_text_target, batch_decoder_input, batch_text_weight, True)
        loss, dec_op = sess.run([losses, logits], feed_dict)
        return loss, dec_op

    def get_sum_batch_loss(batch_loss):
        return np.sum(np.asarray(batch_loss))     

    def perform_training(model, batch_dict, step):
        batch_train_loss, dec_op = get_train_loss(model, batch_dict, step)
        sum_batch_loss = get_sum_batch_loss(batch_train_loss)
        return sum_batch_loss

    def perform_evaluation(model, batch_dict, batch_target_word_ids, batch_text_targets, epoch, step, vocab):
        batch_valid_loss, valid_op = get_valid_loss(model, batch_dict)
        sum_batch_loss = get_sum_batch_loss(batch_valid_loss)    
        batch_predicted_sentence, prob_predicted_words, prob_true_words = get_predicted_sentence(valid_op, batch_target_word_ids, vocab)
	sys.stdout.flush()
	print 'shape of prob_true_words ', prob_true_words.shape
	sys.stdout.flush()
        print_pred_true_op(batch_predicted_sentence, prob_predicted_words, prob_true_words, batch_text_targets, step, epoch, batch_valid_loss)
        return sum_batch_loss

    def evaluate(model, epoch, step, valid_data, valid_text_targets, vocab):
        print 'Validation started'
        sys.stdout.flush()
        valid_loss = 0
        batch_predicted_sentence=[] 
        n_batches = int(math.ceil(float(len(valid_data))/float(param['batch_size']))) 
        for i in range(n_batches):
            batch_dict = valid_data[i*param['batch_size']:(i+1)*param['batch_size']]
            batch_target_word_ids = valid_text_targets[i*param['batch_size']:(i+1)*param['batch_size']]
	    batch_target_sentences = map_id_to_word(batch_target_word_ids, vocab)
            sum_batch_loss = perform_evaluation(model, batch_dict, batch_target_word_ids, batch_target_sentences, epoch, step, vocab)
            valid_loss = valid_loss + sum_batch_loss
        return float(valid_loss)/float(len(valid_data))
        
    def print_pred_true_op(pred_op, prob_pred, prob_true, true_op, step, epoch, batch_valid_loss):
        for i in range(0,len(true_op),100):
            print "true sentence in step "+str(step)+" of epoch "+str(epoch)+" is:"
            sys.stdout.flush()
            print true_op[i]
            print "\n"
            print "predicted sentence in step "+str(step)+" of epoch "+str(epoch)+" is:"
            sys.stdout.flush()
            print pred_op[i]
            print "\n"
	    print "prob of predicted words in step "+str(step)+" of epoch "+str(epoch)+" is:"
            sys.stdout.flush()
            print prob_pred[i]
            print "\n"
	    print "prob of true words in step "+str(step)+" of epoch "+str(epoch)+" is:"
            sys.stdout.flush()
            print prob_true[i]
	    print "\n"
	    #print "crossent  in step "+str(step)+" of epoch "+str(epoch)+" is:"
	    #sys.stdout.flush()
	    #print sum([math.log(x+1e-12) for x in prob_true[i]])		
            #print "\n"
            sys.stdout.flush()
            print "loss for the pair of true and predicted sentences", str(batch_valid_loss[i])
            print "\n"

    def map_id_to_word(word_indices, vocab):
        sentence_list = []
        for sent in word_indices:
            word_list = []
            for word_index in sent:
                word = vocab[word_index]
                word_list.append(word)
            sentence_list.append(" ".join(word_list))
        return sentence_list

    def get_predicted_sentence(valid_op, true_op, vocab):
        max_probs_index = []
	max_probs = []
        #len(valid_op) is max_len
        #true_op is of dimension batch_size * max_len
        if true_op is not None:
		true_op = true_op.tolist()
		true_op = np.asarray(true_op).T.tolist()
		true_op_prob = []
	i=0
	for op in valid_op:
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

    train_data = pkl.load(open(param['train_data_file']))
    print 'Train dialogue dataset loaded'
    sys.stdout.flush()
    valid_data = pkl.load(open(param['valid_data_file']))
    print 'Valid dialogue dataset loaded'
    sys.stdout.flush()
    vocab = pkl.load(open(param['vocab_file'],"rb"))
    vocab_size = len(vocab)
    param['decoder_words'] = vocab_size
    valid_text_targets = load_valid_test_target(valid_data)
    print 'valid target sentence list loaded'
    print 'writing terminal output to file'
    f_out = open(param['terminal_op'],'w')
    sys.stdout=f_out
    check_dir(param)
    n_batches = int(math.ceil(float(len(train_data))/float(param['batch_size'])))
    print 'number of batches ', n_batches, 'len train data ', len(train_data), 'batch size' , param['batch_size']
    model_file = os.path.join(param['model_path'],"best_model")
    with tf.Graph().as_default():
        model = Hierarchical_seq_model_text('text', param['text_embedding_size'], param['image_embedding_size'], param['image_rep_size'], param['cell_size'], param['cell_type'], param['batch_size'], param['learning_rate'], param['max_len'], param['max_utter'], param['max_images'], param['patience'], param['decoder_words'], param['max_gradient_norm'], param['activation'], param['output_activation'])   
        model.create_placeholder()
        logits = model.inference()
        losses = model.loss_task_text(logits)
        train_op, gradients = model.train(losses)
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
	last_overall_avg_train_loss = None
	overall_step_count = 0
        for epoch in range(param['max_epochs']):
            random.shuffle(train_data)
            train_loss=0
            for i in range(n_batches):
		overall_step_count = overall_step_count + 1
                train_batch_dict=train_data[i*param['batch_size']:(i+1)*param['batch_size']]
                sum_batch_loss = perform_training(model, train_batch_dict, overall_step_count)
		avg_batch_loss = sum_batch_loss / float(param['batch_size'])
		if overall_step_count%param['print_train_freq']==0:
	                print('Epoch  %d Step %d train loss (avg over batch) =%.6f' %(epoch, i, avg_batch_loss))
        	        sys.stdout.flush()
                train_loss = train_loss + sum_batch_loss
                avg_train_loss = float(train_loss)/float(i+1)            
                if overall_step_count>0 and overall_step_count%param['valid_freq']==0:
                    overall_avg_valid_loss = evaluate(model, epoch, i, valid_data, valid_text_targets, vocab)
                    print ('Epoch %d Step %d ... overall avg valid loss= %.6f' %(epoch, i, overall_avg_valid_loss))
                    sys.stdout.flush()
                    if best_valid_loss>overall_avg_valid_loss:
                        saver.save(sess, model_file)
                        best_valid_loss=overall_avg_valid_loss
                    else:
                        continue
	    overall_avg_train_loss = train_loss/float(len(train_data))	
            print 'epoch ',epoch,' of training is completed ... overall avg. train loss ', overall_avg_train_loss	
	    if last_overall_avg_train_loss is not None and overall_avg_train_loss > last_overall_avg_train_loss:
		diff = overall_avg_train_loss - last_overall_avg_train_loss
		if diff>param['train_loss_incremenet_tolerance']:	
			print 'WARNING: training loss (%.6f) has increased by %.6f since last epoch, has exceed tolerance of %f ' %(overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance'])
		else:
			print 'WARNING: training loss (%.6f) has increased by %.6f since last epoch, but still within tolerance of %f ' %(overall_avg_train_loss, diff, param['train_loss_incremenet_tolerance'])	
	    last_overall_avg_train_loss = overall_avg_train_loss    		
            sys.stdout.flush()
        print 'Training over'
        print 'Evaluating on test data'
    f_out.close()
    
    

def main():
    data_dir = sys.argv[1]
    param = get_params(data_dir, sys.argv[2])
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



                    


    

































