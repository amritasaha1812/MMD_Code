import random
import sys
import os
import json
import copy
import cPickle as pkl
import argparse
import logging
import nltk
from nltk import word_tokenize
from itertools import izip
import collections
from collections import Counter
import gc

class PrepareData():
	def __init__(self, max_utter, max_len, max_images, max_negs, start_symbol_index, end_symbol_index, unk_symbol_index, pad_symbol_index, task_type, cutoff=-1):
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger('prepare_data_for_hred')
		self.max_utter = max_utter
 		self.max_len = max_len
		self.max_images = max_images
		self.max_negs = max_negs
		self.task_type = task_type
		self.unknown_word_id = unk_symbol_index
		self.start_word_id = start_symbol_index
		self.pad_word_id = pad_symbol_index
		self.end_word_id = end_symbol_index
		self.start_word_symbol = '</s>'
		self.end_word_symbol = '</e>'
		self.pad_symbol = '<pad>'
		self.unk_symbol = '<unk>'	
		self.cutoff = cutoff
		self.input = None
		self.output = None
		self.vocab_file = None
		self.vocab_dict = None
		self.word_counter = None
		gc.enable()

	def safe_pickle(self, obj, filename):
		if os.path.isfile(filename):
			self.logger.info("Overwriting %s." % filename)
		else:
			self.logger.info("Saving to %s." % filename)

		with open(filename, 'wb') as f:
			pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

	def prepare_data(self, input, vocab_file, vocab_stats_file, output, dialogue_pkl_file, isTrain, isTest=False, test_state=None):
		if not os.path.isdir(input) or len(os.listdir(input))==0:
			raise Exception("Input file not found")
		if not self.task_type=="text" and not self.task_type=="image":
			raise Exception("task_type has to be either text or image, found "+self.task_type)
		self.vocab_file = vocab_file
		self.vocab_stats_file = vocab_stats_file	
		self.output = output
		self.isTrain = isTrain
		if self.task_type=="text":
			self.dialogue_context_text_task_text_file = self.output+"_context_text_task_text.txt"
			self.dialogue_context_image_task_text_file = self.output+"_context_image_task_text.txt"
			self.dialogue_target_text_task_text_file = self.output+"_target_text_task_text.txt"
		if self.task_type=="image":	
			self.dialogue_context_text_task_image_file = self.output+"_context_text_task_image.txt"
			self.dialogue_context_image_task_image_file = self.output+"_context_image_task_image.txt"
			self.dialogue_target_image_task_image_pos_file = self.output+"_target_image_task_image_pos.txt"	
			self.dialogue_target_image_task_image_negs_file = self.output+"_target_image_task_image_negs.txt"
		if os.path.isfile(vocab_file):
			print 'found pre-existing vocab file ... reusing it'
			create_vocab = False
		else:
			create_vocab = True
		self.read_jsondir(input, isTest, test_state, create_vocab)	
		if create_vocab:
			self.build_vocab()
		else:	
			self.read_vocab()
		if create_vocab or not os.path.exists(dialogue_pkl_file):
			if self.task_type=="text":
                	        self.binarize_text_corpora(self.dialogue_context_text_task_text_file, self.dialogue_context_image_task_text_file, self.dialogue_target_text_task_text_file, self.task_type, dialogue_pkl_file)
	                if self.task_type=="image":
        	                self.binarize_image_corpora(self.dialogue_context_text_task_image_file, self.dialogue_context_image_task_image_file, self.dialogue_target_image_task_image_pos_file, self.dialogue_target_image_task_image_negs_file, dialogue_pkl_file)	
	
	def read_jsondir(self, json_dir, isTest, test_state, create_vocab = False):
		'''
		if self.task_type=="text":
                        if create_vocab and os.path.isfile(self.dialogue_context_text_task_text_file) and os.path.isfile(self.dialogue_context_image_task_text_file) and os.path.isfile(self.dialogue_target_text_task_text_file):
				print 'reading from pre-existing dialogue temp text files, please make sure these files have the full data'
                                return
                elif self.task_type=="image":
                        if create_vocab and os.path.isfile(self.dialogue_context_text_task_image_pos_file) and os.path.isfile(self.dialogue_context_image_task_image_pos_file) and os.path.isfile(self.dialogue_target_image_task_image_pos_file) and os.path.isfile(dialogue_index_task_image_pos_file) and os.path.isfile(self.dialogue_context_text_task_image_neg_file) and os.path.isfile(self.dialogue_context_image_task_image_neg_file) and os.path.isfile(self.dialogue_target_image_task_image_neg_file) and os.path.isfile(dialogue_index_task_image_neg_file):
				print 'reading from pre-existing dialogue temp text files, please make sure these files have the full data'	
                                return
		else:
		'''	
		if self.task_type=="text":
			if os.path.isfile(self.dialogue_context_text_task_text_file):
				os.remove(self.dialogue_context_text_task_text_file)
			if os.path.isfile(self.dialogue_context_image_task_text_file):
				os.remove(self.dialogue_context_image_task_text_file)
			if os.path.isfile(self.dialogue_target_text_task_text_file):
				os.remove(self.dialogue_target_text_task_text_file)
		if self.task_type=="image":
			if os.path.isfile(self.dialogue_context_text_task_image_file):	
				os.remove(self.dialogue_context_text_task_image_file)
			if os.path.isfile(self.dialogue_context_image_task_image_file):
				os.remove(self.dialogue_context_image_task_image_file)
			if os.path.isfile(self.dialogue_target_image_task_image_pos_file):
				os.remove(self.dialogue_target_image_task_image_pos_file)		
			if os.path.isfile(self.dialogue_target_image_task_image_negs_file):
				os.remove(self.dialogue_target_image_task_image_negs_file)	
		if create_vocab:
	                self.word_counter = Counter()
                else:
                        self.word_counter = None
		for file in os.listdir(json_dir):			
			if file.endswith('.json'):
				self.read_jsonfile(os.path.join(json_dir, file), create_vocab, isTest, test_state)


        def pad_or_clip_dialogue(self, dialogue_instance):
                dialogue_instance = self.rollout_dialogue(dialogue_instance)
                if len(dialogue_instance)>(self.max_utter+1):
                        dialogue_instance_copy = dialogue_instance[-(self.max_utter+1):]
                        return dialogue_instance_copy
                elif len(dialogue_instance)<(self.max_utter+1):
                        padded_dialogue_instance = []
                        pad_length = self.max_utter + 1 - len(dialogue_instance)
                        padded_dialogue_instance = [{'images':[], 'nlg':''}]*pad_length
                        padded_dialogue_instance.extend(dialogue_instance)
                        return padded_dialogue_instance
                else:
                        return dialogue_instance

        def rollout_dialogue(self, dialogue_instance):
                rolledout_dialogue = []
                for instance in dialogue_instance:
                        if instance['nlg'] is not None:
                                rolledout_dialogue.append({"nlg":instance['nlg'] , "images":None})
                        if instance['images'] is not None and len(instance['images'])>0:
                                for image in instance['images']:
                                        rolledout_dialogue.append({"images":[image],"nlg":None})
		dialogue_instance = rolledout_dialogue
                return rolledout_dialogue

        def pad_or_clip_utterance(self, utterance):
         	if len(utterance)>(self.max_len-2):
			utterance = utterance[:(self.max_len-2)]
			utterance.append(self.end_word_symbol)
                        utterance.insert(0, self.start_word_symbol)
		elif len(utterance)<(self.max_len-2):
			pad_length = self.max_len - 2 - len(utterance)
			utterance.append(self.end_word_symbol)
                        utterance.insert(0, self.start_word_symbol)
			utterance = utterance+[self.pad_symbol]*pad_length
		else:
			utterance.append(self.end_word_symbol)
                        utterance.insert(0, self.start_word_symbol)		
		return utterance

	def pad_or_clip_images(self, images):
		if len(images)>self.max_images:
			images = images[:self.max_images]
		elif len(images)<self.max_images:
			pad_length = self.max_images - len(images)
			images = images+['']*pad_length
		return images							

	def read_jsonfile(self, json_file, create_vocab, is_test, test_state):
		#print 'json file ', json_file
		try:
			dialogue = json.load(open(json_file))
		except:
			return None
		filter(None, dialogue)
		#dialogue_multimodal is a list of training instances, each of len max_utter, and each ending with a system response. Whenever a dialogue has context less than max_utter, it is accordingly padded
		dialogue_vocab = {}
		dialogue_multimodal = []
		if self.task_type=="text":
			dialogue_context_text_task_text = []
			dialogue_context_image_task_text = []
			dialogue_target_text_task_text = []
		if self.task_type=="image":
			dialogue_context_text_task_image = []
			dialogue_context_image_task_image = []
			dialogue_target_image_task_image_pos = []
			dialogue_target_image_task_image_negs = []
		dialogue_instance_multimodal = []
		last_question_type = None	
		count=0 
		for utterances in dialogue:
		   if utterances is None or len(utterances)==0:
		   	continue	
 		   if not isinstance(utterances, list):
		     utterances = [utterances]
                   for utterance in utterances:
			if utterance is None:
				continue	
			if not isinstance(utterance, dict):
				print 'impossible ', utterance, json_file
				raise Exception('error in reading dialogue json')
				continue 
			images = None
			false_images = None	
			nlg = None
			speaker = utterance['speaker']
			'''
			if 'images' not in utterance['utterance']:
				raise Exception('utterance problematic ', utterance['utterance'])	
			'''
			if 'images' in utterance['utterance']:	
				images = utterance['utterance']['images']
			'''
			if images is not None and 'false images' not in utterance['utterance']:
				raise Exception('utterance problematic ', utterance['utterance'])
			'''
			if 'false images' in utterance['utterance']:
				false_images = utterance['utterance']['false images']
			nlg = utterance['utterance']['nlg']
			if nlg is not None:
				nlg = nlg.strip().encode('utf-8')
			if nlg is None:
				nlg = ""
			nlg = nlg.lower().replace("|","")
			try:
				nlg_words = nltk.word_tokenize(nlg)
			except:
				nlg_words = nlg.split(' ')
			if create_vocab:
				self.word_counter.update(nlg_words)
			dialogue_instance_multimodal.append({'images': images, 'nlg':nlg, 'false images':false_images})
			if speaker=="system"  and (test_state is None or is_test is False or (last_question_type is not None and test_state is not None and is_test is True and test_state in last_question_type)):
				last_utterance = dialogue_instance_multimodal[-1]
				if self.task_type=="text" and (last_utterance['nlg'] is None or last_utterance['nlg']==""):	
					continue
				if self.task_type=="image" and ((last_utterance['images'] is None or len(last_utterance['images'])==0) and (last_utterance['false images'] is None or len(last_utterance['false images'])==0)):
					continue
				if self.task_type=="image":	
					count=count+1
				#try:
				#	print 'for state ', test_state, ' last_q_type',last_question_type, 'is_test',is_test,'  got a last utterance ', dialogue_instance_multimodal[-2]['nlg']
				#except:
				#	print ''
				padded_clipped_dialogue = self.pad_or_clip_dialogue(dialogue_instance_multimodal)
				if len(padded_clipped_dialogue)!=(self.max_utter+1):
					raise Exception('some problem with dialogue instance, len != max_utter+1')
				#dialogue_instance_task_test is a max_utter length list of utterances where the last utterance in the list is the target utterance 
				dialogue_instance_text_context =  [x['nlg'] if x['nlg'] is not None else '' for x in padded_clipped_dialogue[:-1]]
				#dialogue_instance_task_image is a max_utter length list of image-lists where the last entry in the list is a single image instead of a list and it is the target image
				dialogue_instance_image_context = [x['images'] if x['images'] is not None else [] for x in padded_clipped_dialogue[:-1]]
				
				#print 'dialogue_instance_text_context ', dialogue_instance_text_context
				#print ''
				#print 'dialogue_instance_image_context ', dialogue_instance_image_context
				if len(dialogue_instance_text_context)!=self.max_utter:
					raise Exception('len(dialogue_instance_text_context)!=self.max_utter')
				if len(dialogue_instance_image_context)!=self.max_utter:	
					raise Exception('len(dialogue_instance_image_context)!=self.max_utter')
				if self.task_type=="text":
					dialogue_target_text = dialogue_instance_multimodal[-1]['nlg']
					dialogue_instance_context_text_task_text = copy.deepcopy(dialogue_instance_text_context)
					dialogue_instance_context_image_task_text = copy.deepcopy(dialogue_instance_image_context)
					dialogue_context_text_task_text.append(dialogue_instance_context_text_task_text)
					dialogue_context_image_task_text.append(dialogue_instance_context_image_task_text)
					dialogue_target_text_task_text.append(dialogue_target_text)
				if self.task_type=="image":
					dialogue_target_images_pos = dialogue_instance_multimodal[-1]['images']
					dialogue_target_images_negs = []
					if dialogue_instance_multimodal[-1]['false images'] is not None:
						for img in dialogue_instance_multimodal[-1]['false images']:
							if dialogue_target_images_pos and img in dialogue_target_images_pos and len(dialogue_target_image_task_image_pos)>0:
								sampled_img = random.sample(dialogue_target_image_task_image_pos,1)				
								if isinstance(sampled_img, list) or isinstance(sampled_img, set):
									sampled_img = list(sampled_img)[0]
								dialogue_target_images_negs.append(img)
							else:
								dialogue_target_images_negs.append(img)
					if (dialogue_target_images_pos is None or len(dialogue_target_images_pos)==0)  and (dialogue_target_images_negs is None or dialogue_target_images_negs is None):
						continue
					if dialogue_target_images_pos is not None:
						if (dialogue_target_images_negs is None or len(dialogue_target_images_negs)<self.max_negs):
							if dialogue_target_images_negs is None:
								dialogue_target_images_negs = []
							pad_length = self.max_negs - len(dialogue_target_images_negs)
							dialogue_target_images_negs = dialogue_target_images_negs + ['RANDOM']*pad_length
						elif len(dialogue_target_images_negs)>self.max_negs:
							dialogue_target_images_negs = dialogue_target_images_negs[:self.max_negs]	
					if dialogue_target_images_negs is None:
						dialogue_target_images_negs = []
					if dialogue_target_images_pos is None:
						dialogue_target_images_pos = []
					for image in dialogue_target_images_pos:
						dialogue_instance_context_text_task_image = copy.deepcopy(dialogue_instance_text_context)
        	                                dialogue_instance_context_image_task_image = copy.deepcopy(dialogue_instance_image_context)
                	                        dialogue_context_text_task_image.append(dialogue_instance_context_text_task_image)
                        	                dialogue_context_image_task_image.append(dialogue_instance_context_image_task_image)
						dialogue_target_image_task_image_pos.append(image)
						dialogue_target_image_task_image_negs.append(dialogue_target_images_negs)
			if 'question-type' in utterance and test_state is not None:
	                        last_question_type = utterance['question-type']
			elif speaker!="system":
				last_question_type = None	
		if self.task_type=="text":				
			with open(self.dialogue_context_text_task_text_file, 'a') as fp:
				for dialogue_instance in dialogue_context_text_task_text:
					dialogue_instance = '|'.join(dialogue_instance)
					fp.write(dialogue_instance+'\n')		
			with open(self.dialogue_context_image_task_text_file, 'a') as fp:
				for dialogue_instance in dialogue_context_image_task_text:
					image_context = None
					if len(dialogue_instance)!=self.max_utter:
						raise Exception('len(dialogue_instance_image_context)!=self.max_utter')		
					for images in dialogue_instance:	
						if image_context is None:
							image_context  = ",".join(images)
						else:	
							image_context = image_context+"|"+",".join(images)
					#print image_context
					if len(image_context.split("|"))!=self.max_utter:
                                                raise Exception('len(dialogue_instance_image_context)!=self.max_utter')	
	 				fp.write(image_context+'\n')
		 	with open(self.dialogue_target_text_task_text_file, 'a') as fp:
		 		for dialogue_instance in dialogue_target_text_task_text:
	 				fp.write(dialogue_instance+'\n')
		if self.task_type=="image":
		 	with open(self.dialogue_context_text_task_image_file, 'a') as fp:
		 		for dialogue_instance in dialogue_context_text_task_image:
	 				dialogue_instance = '|'.join(dialogue_instance)
	 				fp.write(dialogue_instance+'\n')
		 	with open(self.dialogue_context_image_task_image_file, 'a') as fp:
		 		for dialogue_instance in dialogue_context_image_task_image:
					image_context = None
					if len(dialogue_instance)!=self.max_utter:
                                                raise Exception('len(dialogue_instance_image_context)!=self.max_utter')
                                        for images in dialogue_instance:
                                                if image_context is None:
                                                        image_context  = ",".join(images)
                                                else:
                                                        image_context = image_context+"|"+",".join(images)
					if len(image_context.split("|"))!=self.max_utter:
                                                raise Exception('len(dialogue_instance_image_context)!=self.max_utter')
                                        fp.write(image_context+'\n')
		 	with open(self.dialogue_target_image_task_image_pos_file, 'a') as fp:
		 		for dialogue_instance in dialogue_target_image_task_image_pos:
	 				fp.write(dialogue_instance+'\n')
			with open(self.dialogue_target_image_task_image_negs_file, 'a') as fp:
				for dialogue_instance in dialogue_target_image_task_image_negs:
					fp.write('|'.join(dialogue_instance)+'\n')

	def read_vocab(self):
		assert os.path.isfile(self.vocab_file)
		self.vocab_dict = {word:word_id for word_id, word in pkl.load(open(self.vocab_file, "r")).iteritems()}
		assert self.unk_symbol in self.vocab_dict
		assert self.start_word_symbol in self.vocab_dict
		assert self.end_word_symbol in self.vocab_dict
		assert self.pad_symbol in self.vocab_dict
	 
	def build_vocab(self):
		total_freq = sum(self.word_counter.values())
		self.logger.info("Total word frequency in dictionary %d ", total_freq)

		if self.cutoff != -1:
			self.logger.info("Cutoff %d", self.cutoff)
			vocab_count = [x for x in self.word_counter.most_common() if x[1]>=self.cutoff]
		else:
			vocab_count = [x for x in self.word_counter.most_common() if x[1]>=5] 

		self.vocab_dict = {self.unk_symbol:self.unknown_word_id, self.start_word_symbol:self.start_word_id, self.pad_symbol:self.pad_word_id, self.end_word_symbol:self.end_word_id}
		
		i = 4
		for (word, count) in vocab_count:
			if not word in self.vocab_dict:
				self.vocab_dict[word] = i
				i += 1

		self.logger.info('Vocab size %d' % len(self.vocab_dict))

	def binarize_image_corpora(self, dialogue_file_text, dialogue_file_image, dialogue_target_file_pos, dialogue_target_file_negs, dialogue_pkl_file):
		binarized_corpus = []
                binarized_corpus_text_context = []
                binarized_corpus_image_context = []
                binarized_corpus_target = []    
                unknowns = 0.
                num_terms = 0.
                freqs = collections.defaultdict(lambda: 0)
                df = collections.defaultdict(lambda: 0)
                num_instances = 0
		all_images = open(dialogue_target_file_pos).readlines()
                with open(dialogue_file_text) as textlines, open(dialogue_file_image) as imagelines, open(dialogue_target_file_pos) as targetposlines, open(dialogue_target_file_negs) as targetnegslines:
			#if len(set([len(indexlines), len(textlines), len(imagelines), len(targetlines)]))>1:
			#	raise Exception('different number of of lines in the index/text/image/target files')
                        for text_context, image_context, target_pos, target_negs in izip(textlines, imagelines, targetposlines, targetnegslines):
				text_context = text_context.lower().strip()
                                image_context = image_context.strip()
                                target_pos = target_pos.strip()
				target_negs = target_negs.strip().split('|')
				loc_randoms = [i for i,entry in enumerate(target_negs) if entry=='RANDOM']
				mask_target_negs = [1.]*len(target_negs)
				num_random = len(loc_randoms)
				randoms = random.sample(all_images, num_random)
				if not self.isTrain:
					for i, random_image in zip(loc_randoms, randoms):
						mask_target_negs[i]=0.
						target_negs[i] = random_image
				else:
					for i, random_image in zip(loc_randoms, randoms):
						target_negs[i] = random_image
                                #print 'text_content ', text_context
                                #print 'image_content ', image_context
                                #print 'target ', target
                                #print ''
                                num_instances += 1
				if num_instances%10000==0:
					print 'finished ',num_instances

                                utterances = text_context.split('|')
                                binarized_text_context = []
                                for utterance in utterances:
					try:
	                                        utterance_words = nltk.word_tokenize(utterance)
					except:
						utterance_words = utterance.split(' ')
                                        utterance_words = self.pad_or_clip_utterance(utterance_words)
                                        if self.end_word_symbol not in utterance_words:
                                                print 'utterance ',utterance
                                                print 'utterance words ',utterance_words 
                                                raise Exception('utterance does not have end symbol')
                                        utterance_word_ids = []
                                        for word in utterance_words:
                                                word_id = self.vocab_dict.get(word, self.unknown_word_id)
                                                utterance_word_ids.append(word_id)
                                                unknowns += 1 * (word_id == self.unknown_word_id)
                                                freqs[word_id] += 1
                                        if self.end_word_id not in utterance_word_ids:
                                                print 'utterance word ids ', utterance_word_ids
                                                raise Exception('utterance word ids does not have end word id') 
                                        num_terms += len(utterance_words)

                                        unique_word_indices = set(utterance_word_ids)
                                        for word_id in unique_word_indices:
                                                df[word_id] += 1
                                        binarized_text_context.append(utterance_word_ids)
                                if len(binarized_text_context)!=self.max_utter:
                                        raise Exception('binarized_text_context should be a list of length max_utter, found length ', len(binarized_text_context))
                                binarized_image_context = [self.pad_or_clip_images(x.split(",")) for x in image_context.split('|')]
                                
                                if len(binarized_image_context)!=self.max_utter:
                                        raise Exception('binarized_image_context should be a list of length max_utter, found length ', len(binarized_image_context))

                                binarized_target_pos = target_pos
				binarized_target_negs = target_negs
                                #binarized_corpus_text_context.append(binarized_text_context)
                                #binarized_corpus_image_context.append(binarized_image_context)
                                #binarized_corpus_target.append(binarized_target)
                                binarized_corpus.append([binarized_text_context, binarized_image_context, [binarized_target_pos], binarized_target_negs, mask_target_negs])
                #binarized_corpus = [binarized_corpus_text_context,  binarized_corpus_image_context, binarized_corpus_target]
                self.safe_pickle(binarized_corpus, dialogue_pkl_file)           
                if not os.path.isfile(self.vocab_file):
                        self.safe_pickle([(word, word_id, freqs[word_id], df[word_id]) for word, word_id in self.vocab_dict.items()], self.vocab_stats_file)
                        inverted_vocab_dict = {word_id:word for word, word_id in self.vocab_dict.iteritems()}   
                        self.safe_pickle(inverted_vocab_dict, self.vocab_file)
                        print 'dumped vocab in ', self.vocab_file
                self.logger.info("Number of unknowns %d" % unknowns)
                self.logger.info("Number of terms %d" % num_terms)
                try:
			self.logger.info("Mean document length %f" % float(sum(map(len, binarized_corpus))/len(binarized_corpus)))
		except:
			self.logger.info("")	
                self.logger.info("Writing training %d dialogues (%d left out)" % (len(binarized_corpus), num_instances + 1 - len(binarized_corpus))) 
	
	def binarize_text_corpora(self, dialogue_file_text, dialogue_file_image, dialogue_target_file, dialogue_pkl_file):
		binarized_corpus = []
		binarized_corpus_text_context = []
		binarized_corpus_image_context = []
		binarized_corpus_target = []	
		unknowns = 0.
		num_terms = 0.
		freqs = collections.defaultdict(lambda: 0)
		df = collections.defaultdict(lambda: 0)
		num_instances = 0
		with open(dialogue_file_text) as textlines, open(dialogue_file_image) as imagelines, open(dialogue_target_file) as targetlines:
			for text_context, image_context, target in izip(textlines, imagelines, targetlines):
				text_context = text_context.lower().strip()
				image_context = image_context.strip()
				target = target	
				#print 'text_content ', text_context
				#print 'image_content ', image_context
				#print 'target ', target
				#print ''
				num_instances += 1
				utterances = text_context.split('|')
				binarized_text_context = []
				for utterance in utterances:
					try:
						utterance_words = nltk.word_tokenize(utterance)
					except:
						utterance_words = utterance.split(' ')
					utterance_words = self.pad_or_clip_utterance(utterance_words)
					if self.end_word_symbol not in utterance_words:
						print 'utterance ',utterance
						print 'utterance words ',utterance_words 
						raise Exception('utterance does not have end symbol')
					utterance_word_ids = []
					for word in utterance_words:
						word_id = self.vocab_dict.get(word, self.unknown_word_id)
						utterance_word_ids.append(word_id)
						unknowns += 1 * (word_id == self.unknown_word_id)
						freqs[word_id] += 1
					if self.end_word_id not in utterance_word_ids:
						print 'utterance word ids ', utterance_word_ids
						raise Exception('utterance word ids does not have end word id')	
					num_terms += len(utterance_words)

					unique_word_indices = set(utterance_word_ids)
					for word_id in unique_word_indices:
						df[word_id] += 1
					binarized_text_context.append(utterance_word_ids)
				if len(binarized_text_context)!=self.max_utter:
					raise Exception('binarized_text_context should be a list of length max_utter, found length ', len(binarized_text_context))
				binarized_image_context = [self.pad_or_clip_images(x.split(",")) for x in image_context.split('|')]
				
				if len(binarized_image_context)!=self.max_utter:
					raise Exception('binarized_image_context should be a list of length max_utter, found length ', len(binarized_image_context)) 
				
				binarized_target = None
				if task_type=="text":
					utterance = target
					try:
						utterance_words = nltk.word_tokenize(utterance)
					except:
						utterance_words = utterance.split(' ')	
					utterance_words = self.pad_or_clip_utterance(utterance_words)
					if self.end_word_symbol not in utterance_words:
                                                print 'utterance ',utterance
                                                print 'utterance words ',utterance_words
						raise Exception('utterance does not have end symbol')
					utterance_word_ids = []
					for word in utterance_words:
						word_id = self.vocab_dict.get(word, self.unknown_word_id)
						utterance_word_ids.append(word_id)
						unknowns += 1 * (word_id == self.unknown_word_id)
						freqs[word_id] += 1
					if self.end_word_id not in utterance_word_ids:
                                                print 'utterance word ids ', utterance_word_ids
                                                raise Exception('utterance word ids does not have end word id') 	
					num_terms += len(utterance_words)

					unique_word_indices = set(utterance_word_ids)
					for word_id in unique_word_indices:
						df[word_id] += 1
					binarized_target = utterance_word_ids
				#binarized_corpus_text_context.append(binarized_text_context)
				#binarized_corpus_image_context.append(binarized_image_context)
				#binarized_corpus_target.append(binarized_target)
				binarized_corpus.append([binarized_text_context, binarized_image_context, binarized_target])
		#binarized_corpus = [binarized_corpus_text_context,  binarized_corpus_image_context, binarized_corpus_target]
		self.safe_pickle(binarized_corpus, dialogue_pkl_file)		
		if not os.path.isfile(self.vocab_file):
			self.safe_pickle([(word, word_id, freqs[word_id], df[word_id]) for word, word_id in self.vocab_dict.items()], self.vocab_stats_file)
			inverted_vocab_dict = {word_id:word for word, word_id in self.vocab_dict.iteritems()}	
			self.safe_pickle(inverted_vocab_dict, self.vocab_file)
			print 'dumped vocab in ', self.vocab_file
		self.logger.info("Number of unknowns %d" % unknowns)
		self.logger.info("Number of terms %d" % num_terms)
		self.logger.info("Mean document length %f" % float(sum(map(len, binarized_corpus))/len(binarized_corpus)))
		self.logger.info("Writing training %d dialogues (%d left out)" % (len(binarized_corpus), num_instances + 1 - len(binarized_corpus)))			

