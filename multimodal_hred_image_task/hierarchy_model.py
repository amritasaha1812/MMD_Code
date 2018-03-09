import numpy as np
import math
import sys
import os
from cStringIO import StringIO
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
from tensorflow.python.ops import control_flow_ops
import seq2seq
from seq2seq import *

sys.path.append(os.getcwd())
from seq2seq import *
class Hierarchical_seq_model():
    def __init__(self, task_type, text_embedding_size, image_embedding_size, image_rep_size, cell_size, cell_type, batch_size, learning_rate, max_len, max_utter, max_images, max_negs, patience, decoder_words, max_gradient_norm, activation, output_activation):
	'''Parameter initialization '''
	self.task_type=task_type
        self.text_embedding_size=text_embedding_size
        self.image_embedding_size=image_embedding_size
        self.image_rep_size=image_rep_size
        self.cell_size=cell_size
        self.cell_type=cell_type
        self.batch_size=batch_size
        self.learning_rate=tf.Variable(float(learning_rate),trainable=False)
        self.max_len=max_len
        self.max_utter=max_utter
        self.max_images=max_images
	self.max_negs = max_negs
        self.patience=patience
        self.decoder_words=decoder_words
        self.max_gradient_norm=max_gradient_norm

        self.encoder_img_inputs = None
        self.encoder_text_inputs = None
        self.decoder_text_inputs = None
        #self.concat_dec_inputs = None
        self.target_text = None
        self.target_img_pos = None
	self.target_img_negs = None
        self.text_weights = None
        self.feed_previous = None
	self.image_weights = None        
        self.activation = activation
	self.output_activation = output_activation

        self.enc_scope_text=None#scope for the sentence level encoder for text

        self.W_enc_img=None#List of encoder cells (corresponds to a list of images; list can be maximum of length max_images)
        self.b_enc_img=None
        self.enc_scope_img=None#scope for the image encoder

        self.enc_cells_utter=None#encoder cells (utterenace context level), at this level it is multimodal
        self.enc_scope_utter=None#scope for the utterance level encoder

        if self.task_type=="text":
      		self.dec_cells_text=None#List for different decoder cells(different languages) for now only one.
        	self.dec_scope_text=None#scope for the sentence decoder

        if self.task_type=="image":
        	self.W_proj_utter=None
        	self.b_proj_utter=None
           	self.proj_scope_utter=None	
        	self.W_enc_tgt_img=None#encoder cell (corresponding to the target image)
        	self.b_enc_tgt_img=None
       		self.tgt_scope_img=None#scope for the target image encoder

    	def create_cell_scopes():
    		self.enc_cells_text = rnn_cell.EmbeddingWrapper(self.cell_type(self.cell_size), self.decoder_words, self.text_embedding_size)
	    	self.enc_scope_text = "encoder_text"
    		max_val = np.sqrt(6. / (self.image_rep_size + self.image_embedding_size))
	    	self.W_enc_img = tf.Variable(tf.random_uniform([self.image_rep_size, self.image_embedding_size], -1.*max_val, max_val), name="W_enc_img")
	    	self.b_enc_img = tf.Variable(tf.constant(0., shape=[self.image_embedding_size]), name="b_enc_img")
    		self.enc_scope_img = "encoder_img"
	    	self.enc_cells_utter = [self.cell_type(self.cell_size), self.cell_type(self.cell_size)]
    		self.enc_scope_utter = "encoder_utter"
	    	if self.task_type=="text":
    			self.dec_cells_text = self.cell_type(self.cell_size)
    			self.dec_scope_text = "decoder_text"
	    	if self.task_type=="image":	
    			self.tgt_scope_img = "target_encoder_img"
			max_val = np.sqrt(6. / (self.image_rep_size + self.image_embedding_size))
    			self.W_enc_tgt_img = tf.Variable(tf.random_uniform([self.image_rep_size, self.image_embedding_size], -1.*max_val, max_val), name="W_enc_tgt_img")
	    		self.b_enc_tgt_img = tf.Variable(tf.constant(0., shape=[self.image_embedding_size]), name="b_enc_tgt_img")
                        
			max_val = np.sqrt(6. / (self.cell_size + self.image_embedding_size))
    			self.proj_scope_utter="proj_utter"
    			self.W_proj_utter = tf.Variable(tf.random_uniform([self.cell_size, self.image_embedding_size], -1.*max_val, max_val), name="W_proj_utter")
	    		self.b_proj_utter = tf.Variable(tf.constant(0., shape=[self.image_embedding_size]), name="b_proj_utter")


    	create_cell_scopes()
   
    def create_placeholder(self):
        #self.encoder_img_inputs is a max_utter sized list of a max_images sized list of tensors of dimension batch_size * image_rep_size
    	self.encoder_img_inputs = [[tf.placeholder(tf.float32,[None, self.image_rep_size], name="encoder_img_inputs") for j in range(self.max_images)] for i in range(self.max_utter)]  # list of tensor placeholders; altogether of dimension  (max_utter * max_images * image_rep_size * batch_size)
    	#self.encoder_text_inputs is a max_utter sized list of a max_len sized list of tensors of dimension batch_size
        self.encoder_text_inputs = [[tf.placeholder(tf.int32,[None], name="encoder_text_inputs") for i in range(self.max_len)] for j in range(self.max_utter)] # list of list of tensor placeholders; altogether of dimension batch_size * max_utter * max_len
    	if self.task_type=="text":
           #self.decoder_text_inputs is a max_len sized list of tensors of dimension batch_size
           self.decoder_text_inputs = [tf.placeholder(tf.int32,[None], name="decoder_text_inputs") for i in range(self.max_len)] # list of tensor placeholders of dimension batch_size * max_len
    	   #self.concat_dec_inputs is a max_len sized list of tensors of dimension batch_size * (cell_size+text_embedding_size)
           #self.concat_dec_inputs = [tf.placeholder(tf.int32, [None, self.cell_size+self.text_embedding_size], name="concat_dec_inputs") for i in range(self.max_len)]
           #self.text_weights is a max_len sized list of tensors of dimension batch_size
    	   self.text_weights = [tf.placeholder(tf.float32, [None], name="text_weights") for i in range(self.max_len)]
           self.feed_previous = tf.placeholder(tf.bool, name='feed_previous')
           #self.target_text is a max_len sized list of tensors of dimension batch_size
           self.target_text = [tf.placeholder(tf.int32,[None], name="target_text") for i in range(self.max_len)]
        elif self.task_type=="image":
           #self.target_img is of dimension  batch_size * image_rep_size
    	   self.target_img_pos = tf.placeholder(tf.float32,[None, self.image_rep_size], name="target_img_pos") ####### THIS PART IS NOT COMPLETE
	   self.target_img_negs = [tf.placeholder(tf.float32, [None, self.image_rep_size], name="target_img_negs") for i in range(self.max_negs)]
	   self.image_weights = [tf.placeholder(tf.float32, [None], name="image_weights") for i in range(self.max_negs)]

    def hierarchical_encoder(self):
    	#enc_text_states = tf.concat(0, self.encoder_text_inputs) ## check
    	#enc_img_states = tf.concat(0, self.encoder_img_inputs) ## check
        #self.encoder_text_inputs is of dimension (max_utter, max_len, batch_size)
        #self.encoder_img_inputs is of dimension (max_utter, max_images, batch_size, img_rep_size)
    	n_steps = self.max_len
    	enc_text_states = self.sentence_encoder(self.encoder_text_inputs)
        #enc_text_states is of dimension (max_utter, batch_size, cell_size)
    	enc_img_states = self.image_encoder(self.encoder_img_inputs)
        #enc_img_states is of dimension (max_utter, batch_size, (max_len*image_embedding_size))
    	enc_concat_text_img_states = self.concat_text_image(enc_text_states, enc_img_states)
        #enc_concat_text_img_states is of dimension (max_utter, batch_size, (cell_size+max_images*image_embedding_size))
    	enc_utter_states = self.utterance_encoder(enc_concat_text_img_states)
        #enc_utter_states is of dimension (cell_size, batch_size)
        return enc_utter_states
    		
    def image_encoder(self, enc_img_inputs):
	#enc_img_inputs would be of dimension (max_utter * max_images * batch_size * img_rep_size)   ## check 		
	#W is of dimension (image_rep_size, image_embedding_size)
	#b is of dimension (1, image_embedding_size)
	enc_img_states = []
	with tf.variable_scope(self.enc_scope_img) as scope:
		for i, enc_img_input in enumerate(enc_img_inputs):
                	#enc_img_input is of dimension (max_images * batch_size * image_rep_size)
                	enc_img_states_i = []
                	for j, inp in enumerate(enc_img_input):
                    		#inp is of dimension (batch_size * image_rep_size)
				if i>0 or j>0:
				    	scope.reuse_variables()
			     	enc_img_state = tf.matmul(inp, self.W_enc_img) + self.b_enc_img
				if self.activation is not None:
					enc_img_state = self.activation(enc_img_state)	
                   		#enc_img_state is of dimension (batch_size * image_embedding_size)	
				enc_img_states_i.append(enc_img_state)
               	 		#enc_img_states_i  is of dimension (max_images * batch_size * image_embedding_size)    
                	enc_img_states.append(enc_img_states_i)    
			#enc_img_states is of dimension (max_utter * max_images * batch_size * image_embedding_size)
	concat_enc_img_states = []		
	for i in range(0, len(enc_img_states)):
        	#enc_img_states[i] is of dimension (max_images * batch_size * image_embedding_size)
            	concat_enc_img_states.append(tf.concat(1, enc_img_states[i]))
		#concat_enc_img_states the max_utter length list of tensors batch_size * (max_images * images_embedding_size)	
		#concat_enc_img_states is of dimension (max_utter * batch_size * (max_images*image_embedding_size))	
	return concat_enc_img_states				

    def concat_text_image(self, enc_text_states, enc_img_states):
    	#enc_text_states is of dimension (max_utter, batch_size, cell_size)
    	#enc_img_states is of dimension (max_utter, batch_size, max_images*image_embedding_size)
    	concat_text_image = []
    	for i in range(len(enc_text_states)):
    		concat_text_image.append(tf.concat(1, [enc_text_states[i], enc_img_states[i]]))
        #concat_text_image is of dimension (max_utter, batch_size, cell_size + max_images*image_embedding_size) 
    	return concat_text_image	

    def sentence_encoder(self, enc_inputs):
        # for the sentence level encoder: enc_inputs is of dimension (max_utter, max_len, batch_size)
        utterance_states = []
        with tf.variable_scope(self.enc_scope_text) as scope:
            #init_state = self.enc_cells_text.zero_state(self.batch_size, tf.float32)
            for i in range(0, len(enc_inputs)):
                if i>0:
                    scope.reuse_variables()
                #enc_inputs[i] is a max_len sized list of tensor of dimension (batch_size) ################# CHECK IF INDEXING OVER TF VARIABLE IS WORKING  
                _, states = rnn.rnn(self.enc_cells_text, enc_inputs[i], scope=scope, dtype=tf.float32)
                #rnn.rnn takes a max_len sized list of tensors of dimension (batch_size * self.text_embedding_size) (after passing through the embedding wrapper)
                #states is of dimension (batch_size, cell_size)
                utterance_states.append(states)
        #utterance_states is of dimension (max_utter, batch_size, cell_size)        
        return utterance_states
    
    def utterance_encoder(self, enc_inputs):
        # for the utterance level encoder: enc_inputs is of dimension (max_utter, batch_size, cell_size+max_images*image_embedding_size)
        utterance_states =  None
        with tf.variable_scope(self.enc_scope_utter) as scope:
            #init_state = self.enc_cells_utter.zero_state(self.batch_size, tf.float32)
            #enc_inputs is of dimension (max_utter, batch_size, cell_size+max_images*image_embedding_size)
            #_, states = rnn.rnn(self.enc_cells_utter, enc_inputs, scope=scope, dtype=tf.float32)
	    _, states, _ = rnn.bidirectional_rnn(self.enc_cells_utter[0], self.enc_cells_utter[1], enc_inputs, dtype=tf.float32, scope=scope)
            #rnn.rnn takes a max_utter sized list of tensors of dimension (batch_size * cell_size+(max_images*image_embedding_size))
            utterance_states= states
        # utterance_states is of dimension (batch_size, cell_size)    
	#self.tf_print(utterance_states)
        return utterance_states   

    def hierarchical_decoder(self, utterance_output):
    	dec_outputs = self.decoder(self.decoder_text_inputs, utterance_output)
    	return dec_outputs
 
    def decode(self, concatenated_input, loop_fn, dec_cell, init_state, utterance_output, dec_scope):
    	state = init_state
    	outputs = []
    	prev = None
        #concatenated_input is of dimension (max_len * batch_size * (cell_size+text_embedding_size))
    	for i, inp in enumerate(concatenated_input):
        	#inp is of dimension batch_size * (cell_size*text_embedding_size))
    		if loop_fn is not None and prev is not None:
    			with tf.variable_scope("loop_function", reuse=True):
    				inp = loop_fn(prev, i)
    				inp = tf.concat(1,[utterance_output, inp])
    		if i > 0:
    			dec_scope.reuse_variables()
    		output, state = dec_cell(inp, state, scope=dec_scope)
    		#inp is of dimension batch_size * (cell_size*text_embedding_size))
    		#state is of dimension batch_size * cell_size
    		#output is of dimension batch_size * cell_size ???????????
    		outputs.append(output)
    		if loop_fn is not None:
    			prev = output
    	#outputs is a max_len sized list of dimension batch_size * cell_size		
    	return outputs, state

    def decoder(self, decoder_inputs, utterance_output):
    	#decoder_inputs is of dimension max_len * batch_size
     	#utterance_output is of dimension cell_size * batch_size										
    	with tf.variable_scope(self.dec_scope_text) as scope:
    		init_state = self.dec_cells_text.zero_state(self.batch_size, tf.float32)
    		#weights = tf.Variable(tf.truncated_normal([self.cell_size, self.decoder_words], stddev=1.0 / math.sqrt(self.cell_size)), name="weights")
    		#weights = tf.Variable(tf.random_uniform([self.cell_size, self.decoder_words], -0.1, 0.1), name="weights")
		#biases = tf.Variable(tf.zeros([self.decoder_words]), name="biases")
    		max_val = np.sqrt(6. / (self.decoder_words + self.cell_size))
                weights=tf.get_variable("dec_weights",[self.cell_size,self.decoder_words],initializer=tf.random_uniform_initializer(-1.*max_val,max_val))#For projecting decoder output which is of self.batch_size*self.cell_size to self.batch_size*self.vocab_size.
                biases = tf.get_variable("dec_biases",[self.decoder_words],initializer=tf.constant_initializer(0.0))
	
    		def feed_previous_decode(feed_previous_bool):
    			dec_embed, loop_fn = seq2seq.get_decoder_embedding(decoder_inputs, self.decoder_words, self.text_embedding_size, output_projection=(weights, biases), feed_previous=feed_previous_bool)
    			#dec_embed is of dimension max_len * batch_size * self.text_embedding_size
    			#utterance_output is of dimension  batch_size * cell_size
    			concatenated_input = self.get_dec_concat_ip(dec_embed, utterance_output)
    			dec_output, _ = self.decode(concatenated_input, loop_fn, self.dec_cells_text, init_state, utterance_output, scope)
    			#dec_output is a max_len sized list of tensors of dimension batch_size * cell_size
    			return dec_output

    		dec_output = control_flow_ops.cond(self.feed_previous, lambda: feed_previous_decode(True), lambda: feed_previous_decode(False))
    		output_projection = (weights, biases)
    		#weights is a tensor of dimension cell_size * decoder_words
    		#bias is a tensor of dimension decoder_words
    		for i in range(len(dec_output)):
    			dec_output[i]=tf.matmul(dec_output[i], output_projection[0])+output_projection[1]
			if self.output_activation is not None:
				dec_output[i] = self.output_activation(dec_output[i])
    			#before the linear transformation, dec_output[i] is a tensor of dimension batch_size * cell_size
    			#weights is a tensor of dimension cell_size * decoder_words
    			#after the linear transformation, dec_output[i] is a tensor of dimension batch_size * decoder_words
    	#dec_output is a max_len sized list of 2D tensors of dimension batch_size * decoder_words
    	return dec_output
    	
    def get_dec_concat_ip(self, dec_embed, utterance_output):
	#self.tf_print(utterance_output)
	concat_dec_inputs = []
   	for (i, inp) in enumerate(dec_embed):
   		#self.tf_print(inp)
		#inp is of dimension batch_size * self.text_embedding_size 
   		#utterance_output is of dimension  batch_size * cell_size
   		concat_dec_inputs.append(tf.concat(1, [utterance_output, inp]))
   		#self.concat_dec_inputs[i] is of dimension batch_size * (cell_size + self.text_embedding_size)					
   	#self.concat_dec_inputs is of dimension max_len * batch_size * (cell_size + self.text_embedding_size)
   	return concat_dec_inputs

    def tf_print(self, x):
	old_stdout =sys.stdout
        sys.stdout= mystdout = StringIO()
        shape_dec = x.get_shape()
        print(shape_dec)
        sys.stdout= old_stdout
        #shape_dec = tf.convert_to_tensor(shape_dec)
        with tf.Session() as session:
               print(session.run(mystdout.getvalue()))
			
    def inference(self):
   	if self.task_type=="text":
   		utterance_output = self.hierarchical_encoder()
		logits = self.hierarchical_decoder(utterance_output)
   		return logits
   	elif self.task_type=="image":
   		utterance_output = self.hierarchical_encoder()
   		projected_utterance_output = self.project_utter_encoding(utterance_output)
   		return projected_utterance_output        

    def loss_task_text(self, logits):
	
	#for logit in logits:	
	#	self.tf_print(logit)
	#print 'len of text weights ', self.text_weights
	#for txt in self.target_text:
	#	self.tf_print(txt)
   	#logits is a max_len sized list of 2-D tensors of dimension batch_size * decoder_words
   	#self.target_text is a max_len sized list of 1-D tensors of dimension batch_size
   	#self.text_weights is a max_len sized list of 1-D tensors of dimension batch_size
   	losses=seq2seq.sequence_loss_by_example(logits, self.target_text, self.text_weights)
   	#losses is a 1-D tensor of dimension batch_size 
   	return losses
   
    def loss_task_image(self, projected_utterance_output):
        target_pos_image_embedding = self.target_pos_image_encoder(self.target_img_pos) 
	target_neg_image_embeddings= self.target_neg_image_encoder(self.target_img_negs)
   	cosine_sim_pos = self.cosine_similarity(projected_utterance_output, target_pos_image_embedding)
	const = tf.ones([self.batch_size])
	zeros = tf.zeros([self.batch_size])
	cosine_sim_negs = []
	losses = []
	for i in range(len(target_neg_image_embeddings)):
		cosine_sim_neg = self.cosine_similarity(projected_utterance_output, target_neg_image_embeddings[i])
		loss = tf.maximum(zeros, const - cosine_sim_pos +cosine_sim_neg)
		#const is a tensor of dimension batch_size
		#cosine_sim_pos is a tensor of dimension batch_size
		#cosine_sim_neg is a tensor of dimension batch_size
		loss = loss * self.image_weights[i]
		losses.append(loss)	
		cosine_sim_negs.append(cosine_sim_neg)
	#cosine_sim_negs is a max_negs sized list of tensors of batch_size
	#cosine_sim_pos is a tensor of batch_size
	#losses is a tensor of dimention self.batch_size
	#text_weights is a placeholder of dimension self.batch_size
	#losses is a max_negs sized list of tensors of batch_size
   	loss = tf.reduce_sum(tf.add_n(losses))
	
	#loss is a tensor of batch_size
	return loss, cosine_sim_pos, cosine_sim_negs

    def project_utter_encoding(self, utter):
    	#utter is of dimension batch_size * cell_size
    	#self.W_proj_utter is of dimension cell_size * image_embedding_size
    	#self.b_proj_utter is of dimension image_embedding_size
        proj_utter = None
        with tf.variable_scope(self.proj_scope_utter) as scope:
            proj_utter = tf.matmul(utter, self.W_proj_utter) + self.b_proj_utter
	    if self.activation is not None:
		proj_utter = self.activation(proj_utter)		
        #proj_utter is of dimension batch_size * image_embedding_size    
        return proj_utter       

    def target_pos_image_encoder(self, target_img_input_pos):
        target_img_state_pos = None
        #target_img_input is of dimension batch_size * image_rep_size
        #self.W_enc_tgt_img is of dimension image_rep_size * image_embedding_size
        #self.b_enc_tgt_img is of dimension image_embedding_size
        with tf.variable_scope(self.tgt_scope_img) as scope:
            target_img_state_pos = tf.matmul(target_img_input_pos, self.W_enc_tgt_img) + self.b_enc_tgt_img
	    if self.activation is not None:
		target_img_state_pos = self.activation(target_img_state_pos)			
        #target_img_state is of dimension batch_size * image_embedding_size    
        return target_img_state_pos
    
    def target_neg_image_encoder(self, target_img_input_negs):
        target_img_state_negs = []
        #target_img_input is of dimension batch_size * image_rep_size
        #self.W_enc_tgt_img is of dimension image_rep_size * image_embedding_size
        #self.b_enc_tgt_img is of dimension image_embedding_size
        with tf.variable_scope(self.tgt_scope_img) as scope:
	    for i in range(0, len(target_img_input_negs)):
		if i>0:
			scope.reuse_variables()		
	        target_img_state_neg = tf.matmul(target_img_input_negs[i], self.W_enc_tgt_img) + self.b_enc_tgt_img
                if self.activation is not None:
                	target_img_state_neg = self.activation(target_img_state_neg)
		target_img_state_negs.append(target_img_state_neg)	
        #target_img_state is of dimension batch_size * image_embedding_size    
        return target_img_state_negs
        
    def cosine_similarity(self, rep1, rep2):
   	#rep1 is of dimension (batch_size * image_embedding_size)
   	#rep2 is of dimension (batch_size * image_embedding_size)
   	normed_rep1 = tf.nn.l2_normalize(rep1, 1) #here normalization has been done per row
   	######## CHECK THAT NORMALIZATION IS HAPPENING W.R.T the row or the batch??? i.e. every dimension in the image_embedding_size is normalized over the batch_size number of values	
   	normed_rep2 = tf.nn.l2_normalize(rep2, 1) #here normalization has been done per row
   	#normed_rep1 is of dimension batch_size * image_embedding_size
   	#normed_rep2 is of dimension batch_size * image_embedding_size
   	cosine_sim = tf.matmul(normed_rep1, normed_rep2, transpose_b=True)
   	cosine_sim = tf.diag_part(cosine_sim)
   	#cosine_similarity is of dimension batch_size * batch_size
   	return cosine_sim

    def train(self, losses):
        parameters=tf.trainable_variables()
        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
        gradients=tf.gradients(losses,parameters)
	#print tf.get_default_graph().as_graph_def()
        clipped_gradients,norm=tf.clip_by_global_norm(gradients,self.max_gradient_norm)
        global_step=tf.Variable(0,name="global_step",trainable='False')
        #train_op=optimizer.minimize(losses,global_step=global_step)
        train_op=optimizer.apply_gradients(zip(clipped_gradients,parameters),global_step=global_step)
        return train_op, clipped_gradients
