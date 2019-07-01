import nltk
import numpy
import operator
import math
from tqdm import tqdm
import json

class ContentCategorization(object):
	"""The Class is a module of RCUB: Recomendation based on Content Categorization
	and Clusterization of Users Behavior

	The Content Categorization (CC) module:
		helps extracting categories out of documents' text 
		builds a representation model of documents
		classify them into a set of categories

	Attributes:
		data (json):
				'articles':(list) - contains all documents
				'id':(string) - unique identification of a document
				'sentences':(list) - list of phrases
					[]:(list) - list of words (only nouns)
				'n_sent':(int) number of 
				'date':(string) date when the documents were created
		categories (json):
			'category_name':(list) - contains a list of keywords for each category
		sentences (list): list of all sentences
		words (list): list of all words

	"""

	def initialization(self,data,categories):
		"""
		Initialization method
		
		Note:
			Loads the model already created or
			Builds the model to be used for classification

		Args:
			data (json):
				'articles':(list) - contains all documents
				'id':(string) - unique identification of a document
				'sentences':(list) - list of phrases
					[]:(list) - list of words (only nouns)
				'n_sent':(int) number of 
				'date':(string) date when the documents were created
			categories (json):
				'category_name':(list) - contains a list of keywords for each category

		Tests:
			Test commands:	`python3 -m doctest CC.py`
					(verbose):  `python3 -m doctest -v CC.py`

			Data set setup:
				>>> from test_data import *

				>>> categories = testCategories()
				>>> 'seguranca' in categories
				True

				>>> data = jsonData()
				>>> 'articles' in data
				True
			
			Object Initialization:
				>>> cc = ContentCategorization()
				>>> cc.initialization(data,categories)
				
				>>> cc.data == data
				True
				>>> cc.categories == categories
				True
			
			Initial attributte states validation:
				>>> len(cc.unique_words) <= len(cc.bag_words)
				True
				>>> len(cc.unique_sentences) <= len(cc.bag_sentences)
				True

				>>> len(cc.index_words) == 0
				True
				>>> len(cc.words_index) == 0
				True
				>>> len(cc.index_sentences) == 0
				True
				>>> len(cc.sentences_index) == 0
				True
				>>> len(cc.wsm) == 0
				True
				>>> len(cc.ssm) == 0
				True

		"""

		# create a list of all phrases and a list of all words in every document
		bag_sentences = []
		bag_words = []
		for a in data['articles']:
			for s in a['sentences']:
				if (len(s) >= 5):
					bag_sentences.append(tuple(s))
					bag_words.extend(s)
		self.data = data
		self.categories = categories
		self.bag_words = bag_words
		self.unique_words = set(bag_words)
		self.bag_sentences = bag_sentences
		self.unique_sentences = set(bag_sentences)
	
		self.index_words = {}
		self.words_index = {}
		self.index_sentences = {}
		self.sentences_index = {}
		self.wsm = {}
		self.ssm = {}

	def extract_representativity(self):
		"""
		Extract the initial representative phrases for each category
		
		Note:
			Two class attributes are created:
				self.representative_sentences: dictionary of sentences that contains a category keyword
				self.unclassified_sentences: dictionary of sentences that have any category keyword

		Args:
			self.categories (dict): dictionary of keywords for each category.
			self.sentences (list): list of all sentences

		Returns:
			representative_sentences (dict): dictionary of sentences that contains a category keyword
			unclassified_sentences = dictionary of sentences that have any category keyword
		"""
		categories = self.categories
		sentences = self.unique_sentences

		representative_sentences = {}
		unclassified_sentences = []
		for cat,v in categories.items():
			representative_sentences[cat] = set([])
		qtd_sentences = 0

		for sentence in sentences:
			qtd_sentences+=1
			anywhere = 0
			for category,keywords in categories.items():
				for keyword in keywords:
					if keyword in sentence:
						representative_sentences[category].add(tuple(sentence))
					else:
						anywhere+=1
			if anywhere == 	len(list(sum(categories.values(), []))):
				unclassified_sentences.append(tuple(sentence))

		self.representative_sentences = representative_sentences
		self.unclassified_sentences = unclassified_sentences

		# Sentences weighting
		sentences_weights = {}
		words_weights = self.TFICF()

		for cat,rep_l in representative_sentences.items():
			sentences_weights[cat] = {}
			for sent in rep_l:
				sentences_weights[cat][sent] = 0
				for word in sent:
					sentences_weights[cat][sent] += words_weights[cat][word]
				sentences_weights[cat][sent] = sentences_weights[cat][sent]/len(sent)

		# self.sentences_weights = sentences_weights

		# Sorte representative sentences and apply a threshold
		
		THRESHOLD = 0.7
		sorted_rep_sentences = {}
		new_unclassi = []
		for cat, sent_l in sentences_weights.items():
			aux = sorted(sentences_weights[cat].items(), key=operator.itemgetter(1), reverse=True)
			sorted_rep_sentences[cat] = aux[:math.ceil(len(aux)*THRESHOLD)]
			new_unclassi.extend(aux[math.ceil(len(aux)*THRESHOLD):])

		representative_tficf_sentences = sorted_rep_sentences
		
		# self.representative_tficf_sentences = representative_tficf_sentences
		
		# self.unclassified_sentences = set(unclassified_sentences)
		# self.new_unclassi = new_unclassi

		# Remove tficf weights

		for cat,sent in representative_tficf_sentences.items():
			representative_sentences[cat] = []
			for i in sent:
				representative_sentences[cat].append(tuple(i[0]))

		for sent in new_unclassi:
			unclassified_sentences.append(sent[0])
		
		unclassified_sentences = set(unclassified_sentences)

		self.representative_sentences = representative_sentences
		self.unclassified_sentences = unclassified_sentences

		# Extract representative and unclassified words from theirs respective sentences

		new_re = []
		for cat,sent in representative_sentences.items():
			for i in sent:
				new_re.append(tuple(i))
		
		new_re = [list(i) for i in new_re]
		new_un_sentences = [list(i) for i in unclassified_sentences]
		
		new_re_words = list(sum(new_re, []))
		unclassified_words = list(sum(new_un_sentences, []))

		self.representative_words = new_re_words
		self.unclassified_words = unclassified_words

		# return representative_sentences
	
	def extend_representativity(self,category):
		"""
		This is the extend_representativity method

		"""
		self.create_similarity_matrix(category)
		
		# # Firt iteraction
		# self.update_similarity_sentence_matrix(category)
		# self.update_similarity_word_matrix(category)
		
		# # Second Iteraction
		# self.update_similarity_sentence_matrix(category)
		# self.update_similarity_word_matrix(category)
		
		# # Third Iteraction
		# self.update_similarity_sentence_matrix(category)
		# self.update_similarity_word_matrix(category)

		self.run_n_interactions(3, category)
		
		# Final
		self.update_similarity_sentence_matrix(category)

		self.export_probability_unclassified(category)

	def run_n_interactions(self, n, category):
		for _i in range(n):
			self.update_similarity_sentence_matrix(category)
			self.update_similarity_word_matrix(category)

	def select_features(self):
		"""
		This is the select_features method

		"""
		features = 0
		return features

	def classify_text(self):
		"""
		This is the classify_text method

		"""
		classification = 0
		return classification

	# Auxiliar Functions

	def TFICF(self):
		"""
		Term Frequency (TF) + Inverse Category Frequency (ICF)

		Note:
			Five class attributes are created:
				self.representative_words: set of words from the representative sentences
				self.tf: term frequency
				self.cf:content frequency
				self.icf: inverse content frequency
				self.tficf: term frequency + inverse content frequency

		Args:
			self.words (list): class attribute containing the list of global words
			self.representative_sentences (list): class attribute containing the list of representative sentences

		Returns:
			words_tficf: Word weights are computed using Term Frequency (TF) and Inverse Category Frequency (ICF).
		"""
		
		representative_sentences = self.representative_sentences
		words = self.unique_words

		## TF (Term Frequency)
		representative_cat_words = {}
		tf = {}
		for cat,sents in representative_sentences.items():
			tmp_re = [list(i) for i in sents]
			re_words = list(sum(tmp_re, []))
			
			representative_cat_words[cat] = re_words
			tf[cat] = dict(nltk.FreqDist(representative_cat_words[cat]))

		## CF (Category Frequency)
		cf = {}
		for w in words:
			cf[w] = 0
		for w in set(words):
			for cat, bag in representative_cat_words.items():
				if w in bag:
					cf[w] += 1

		## ICF (Inverse Category Frequency)
		icf = {}
		ncategories = len(self.categories.keys())
		for w,v in cf.items():
			if v == 0:
				icf[w] = numpy.log(ncategories) - 0
			else:
				icf[w] = numpy.log(ncategories) - numpy.log(v)

		# TF + ICF
		tficf = {}
		for c,l in tf.items():
			tficf[c] = {}
			for w,v in l.items():
				tficf[c].update({str(w):v*icf[w]})

		# self.representative_words = representative_words
		self.tf = tf
		self.cf = cf
		self.icf = icf
		self.tficf = tficf

		return tficf

	def words_weight(self):
		"""
		This is the weight auxiliar method

		Note:
			A class attributes is created:
				weight: the weight of a word by the product of global frequency and likelihood factors

		Args:
			self.sentences: class attribute containing the list of global sentences
			self.global_frequency: class attribute containing the words global frequency
			self.likelihood: class attribute containing the words likelihood

		Returns:
			weight: the weight of a word by the product of global frequency and likelihood factors

		"""
		sentences = self.unique_sentences
		global_frequency = self.global_frequency
		likelihood = self.likelihood

		weight = {}
		for s in sentences:
			sum_factor = numpy.sum(numpy.array(list(global_frequency.values()))*numpy.array(list(likelihood.values())))
			for w in s:
				prod_factor = global_frequency.get(w)*likelihood.get(w) 
				weight[w] = prod_factor/sum_factor
		
		self.weight = weight
		
		return weight

	def create_similarity_matrix (self,category):
		"""
		This is the create_similarity_matrix auxiliar method

		"""
		index_words = self.index_words
		words_index = self.words_index
		index_sentences = self.index_sentences
		sentences_index = self.sentences_index
		wsm = self.wsm
		ssm = self.ssm

		# Words
		val = self.representative_sentences[category]
		aux_sent = [list(i) for i in val]
		cat_words = list(sum(aux_sent, []))
		# uncla_words = set(cc.unclassified_words)
		# cat_words.extend(uncla_words)

		# Words Index
		index_words[category] =  dict(zip(list(range(0,len(cat_words))),sorted(set(cat_words))))
		words_index[category] =  dict(zip(sorted(set(cat_words)),list(range(0,len(cat_words)))))

		# Sentences
		uncla_sentences = self.unclassified_sentences
		sample_uncla_sentences = list(uncla_sentences)

		cat_rep_sentences = self.representative_sentences[category]
		sample_rep_sentences = list(cat_rep_sentences)

		sample_sentences = []
		sample_sentences.extend(sample_rep_sentences)
		sample_sentences.extend(sample_uncla_sentences)
		# sample_sentences = sample_sentences[:200]

		# Sentences Index
		index_sentences[category] = dict(zip(list(range(0,len(sample_sentences))),sorted(sample_sentences)))
		sentences_index[category] = dict(zip(sorted(sample_sentences),list(range(0,len(sample_sentences)))))

		# WSM and SSM Matrices
		wsm[category] = numpy.identity(len(index_words[category]), dtype = float)
		ssm[category] = numpy.zeros((len(index_sentences[category]),len(index_sentences[category])), dtype = float)

		self.index_words = index_words
		self.words_index = words_index
		self.index_sentences = index_sentences
		self.sentences_index = sentences_index
		self.wsm = wsm

	def aff_w_s(self,word,sentence,category):
		if self.words_index[category].get(word) == None: return [('0'),0] 

		aff = {}
		for word_2 in sentence:
			if self.words_index[category].get(word_2) == None: return [('0'),0]
			
			aff[word_2] = self.wsm[category][self.words_index[category].get(word)][self.words_index[category].get(word_2)]
		
		max_aff = sorted(aff.items(), key=operator.itemgetter(1),reverse=True)
		return max_aff[0]

	def aff_s_w(self,sentence,word,category):
		aff = {}
		for sentence_2 in self.sentences_index[category].keys():
			aff[sentence_2] = self.ssm[category][self.sentences_index[category][sentence]][self.sentences_index[category][sentence_2]]
		max_aff = sorted(aff.items(), key=operator.itemgetter(1),reverse=True)
		if len(max_aff) == 0:
			return [('0'),0]
		else:
			return max_aff[0]
	
	def similarity_sentence(self,sentence_1, sentence_2,category):
		"""
		TODO: show better details about this method

		Test commands:	`python3 -m doctest CC.py`
				(verbose):  `python3 -m doctest -v CC.py`

		Tests:
			Setup:
				>>> from test_data import *
				>>> categories = testCategories()
				>>> data = jsonData()

				>>> cc = ContentCategorization()
				>>> cc.initialization(data,categories)

				>>> category  = 'seguranca'
				>>> sentence1 = ('crime', 'mulher', 'ajuda', 'senhor', 'local')
				>>> sentence2 = ('mp', 'forma', 'homicídio', 'crime', 'somente', 'quantidade',
				... 'disparos', 'policiais', 'abordagem', 'fato', 'parada', 'veículo', 'vítimas',
				... 'ainda', 'reação', 'possibilidade', 'defesa', 'marco', 'antônio', 'tiago')
				>>> expected_similarity = 0.12943956330056436
				
				>>> cc.extract_representativity()
				>>> cc.global_frequency_factor() and None
				>>> cc.log_likelihood_factor() and None
				>>> cc.words_weight() and None
				>>> cc.create_similarity_matrix(category)

			Mehtod Execution:
				>>> sentence_similarity = cc.similarity_sentence(sentence1, sentence2, category)

			Assertions:
				>>> sentence_similarity == expected_similarity
				True

				>>> a = 8 or 5
				>>> a
				8
		"""

		sim = 0
		for word in sentence_1:
			aux_a = self.weight.get(word)
			aux_b = self.aff_w_s(word,sentence_2,category)[1]
			sim += aux_a*aux_b
		return sim

	def similarity_word(self,word_1, word_2, category):
		if word_1 == word_2:
			return 1
		else:
			sim = 0
			for sentence in self.sentences_index[category].keys():
				avg_weight = 0
				if word_1 in sentence:
					for temp_word_1 in sentence:
						avg_weight += self.weight[temp_word_1]
					sim += avg_weight*self.aff_s_w(sentence,word_2,category)[1]
			return sim
	
	def update_similarity_word_matrix(self,category):
		"""
		This is the similarity_formula auxiliar method

		"""
		aux_row = list(self.index_words[category].values())
		aux_column = list(self.index_words[category].values())
		for word_a in tqdm(aux_row):
			for word_b in aux_column:
				word_a_index = self.words_index[category].get(word_a)
				word_b_index = self.words_index[category].get(word_b)
				self.wsm[category][word_a_index][word_b_index] = self.similarity_word(word_a,word_b,category)
		numpy.save('wsm_'+category+'_'+str(1)+'.npy', self.wsm[category])

	def update_similarity_sentence_matrix(self,category):
		"""
		This is the similarity_formula auxiliar method

		"""
		for sentence_a in tqdm(list(self.index_sentences[category].values())):
			for sentence_b in list(self.index_sentences[category].values()):
				sentence_a_index = self.sentences_index[category][sentence_a]
				sentence_b_index = self.sentences_index[category][sentence_b]
				self.ssm[category][sentence_a_index][sentence_b_index] = self.similarity_sentence(sentence_a,sentence_b,category)
		numpy.save('ssm_'+category+'_'+str(1)+'.npy', self.ssm[category])

	def export_probability_unclassified(self,category):
		new_classified = {}
		for a in tqdm(list(self.unclassified_sentences)):
			aux = 0
			for val in self.representative_sentences[category]:
				aux += self.similarity_sentence(a,val,category)
				sum_seg = aux
			new_classified[a] = sum_seg
		new_classified = dict(sorted(new_classified.items(), key=operator.itemgetter(0)))
		json_classified = {}
		for i,v in new_classified.items():
			str_i =  ','.join(i)
			json_classified[str_i]=v
		with open('prob_'+category+'.json', 'w') as outfile:  
			json.dump(json_classified, outfile, indent=4)

	def global_frequency_factor(self):
		"""
		The global word frequency of all words

		Note:
			Two class attributes are created:
				words_sentence_frequency: words's frequency in global sentences
				global_frequency: global word frequency defined in ko,2000 and karov,1998

		Args:
			self.words (list): class attribute containing the list of global words
			self.sentences (list): class attribute containing the list of global sentences

		Returns:
			global_frequency: global word frequency defined in ko,2000 and karov,1998
		"""
		words = self.bag_words
		# sentences = self.unique_sentences

		# WSF (Word Sentence Frequency)

		# wsf = {}
		# for w in words:
		# 	w_count = 0
		# 	for sent in sentences:
		# 		if w in sent:
		# 			w_count+=1
		# 	wsf[w] = w_count

		wsf = dict(nltk.FreqDist(words))
		
		self.word_sentence_frequency = wsf

		# Global Frequency

		word_freq_sort = sorted(wsf.items(), key=operator.itemgetter(1),reverse=True)
		max5_freq = sum(dict(word_freq_sort[:5]).values())
		gb_freq_norm = []
		for i in wsf.values():
			gb_freq_norm.append(max(0,1-(i/max5_freq)))
		
		gb_frequency = dict(zip(wsf.keys(),gb_freq_norm))
		
		gb_frequency = dict(sorted(gb_frequency.items(), key=operator.itemgetter(0),reverse=False))

		self.global_frequency = gb_frequency

		return gb_frequency

	def log_likelihood_factor(self):
		"""
		The likelihood of all words

		Note:
			Two class attributes are created:
				rep_word_sentence_frequency: words's frequency in representative sentences
				likelihood: indicative of the sense usually appear in the representative sentences defined in ko,2000 and karov,1998

		Args:
			self.words (list): class attribute containing the list of global words
			self.sentences (list): class attribute containing the list of global sentences
			self.word_sentence_frequency (list): words's frequency in global sentences

		Returns:
			likelihood: indicative of the sense usually appear in the representative sentences defined in ko,2000 and karov,1998
		"""
		unclassified_words = self.unclassified_words
		unclassified_words = sorted(unclassified_words)

		representative_words = self.representative_words
		representative_words = sorted(representative_words)

		all_words = []
		all_words.extend(representative_words)
		all_words.extend(unclassified_words)
		all_words = sorted(all_words)

		self.all_words = all_words

		# freq_representative = numpy.array(list(nltk.FreqDist(representative_words).values()))
		# freq_unclassified = numpy.array(list(nltk.FreqDist(unclassified_words).values()))
		freq_all = nltk.FreqDist(all_words)
		freq_all = dict(sorted(freq_all.items(), key=operator.itemgetter(0),reverse=False))

		self.freq_all = freq_all

		aux_freq_representative = dict(nltk.FreqDist(representative_words))
		aux_freq_unclassified = dict(nltk.FreqDist(unclassified_words))

		for word in all_words:
			if word not in representative_words:
				aux_freq_representative[word] = 0
			if word not in unclassified_words:
				aux_freq_unclassified[word] = 0

		aux_freq_representative = dict(sorted(aux_freq_representative.items(), key=operator.itemgetter(0),reverse=False))
		aux_freq_unclassified = dict(sorted(aux_freq_unclassified.items(), key=operator.itemgetter(0),reverse=False))

		self.aux_freq_representative = aux_freq_representative
		self.aux_freq_unclassified = aux_freq_unclassified
		
		p_w = numpy.array(list(freq_all.values()))/len(all_words)
		
		p_r_w = numpy.array(list(aux_freq_representative.values()))/len(representative_words)
		sum_p_r_w_p_w = numpy.sum(p_r_w*p_w)
		p_w_r = (p_r_w*p_w)/ sum_p_r_w_p_w

		p_u_w = numpy.array(list(aux_freq_unclassified.values()))/len(unclassified_words)
		sum_p_u_w_p_w = numpy.sum(p_u_w*p_w)
		p_w_u = (p_u_w*p_w)/ sum_p_u_w_p_w

		# min_freq = min(aux_freq_representative
		self.p_w = p_w
		self.p_w_r = p_w_r
		self.p_w_u = p_w_u

		x = p_w_r/p_w
		y = p_w_u/p_w

		min_count = numpy.array(list(aux_freq_representative.values()))/3
		min_count[min_count > 1] = 1

		self.min_count = min_count

		likelihood = abs(numpy.log((x+y))*min_count)

		likelihood = dict(zip(sorted(set(all_words)),likelihood))

		self.likelihood = likelihood


		return likelihood

	def part_speech_factor(self):
		"""
		The part of speech factor (NOT IMPLEMENTED)

		Note:
			Since the content of sentences are only nouns, there is no need to apply a part of the speech factor yet.
	
		"""
		pos = 0
		return pos