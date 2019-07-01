from source.text_categorization import TextCategorization
from source.content_categorization import ContentCategorization
from source.helpers.content_categorization.transform import articles_v1_to_v2
from source.helpers.content_categorization.transform import representative_v2_to_v1
from source.helpers.content_categorization.transform import unclassified_v2_to_v1
from source.helpers.content_categorization.transform import corpus_v2_to_v1
import json
import sys
sys.path.append('./source/')
import nltk
import operator
import random
import numpy
import math
from tqdm import tqdm
import time
import pickle
import math

with open('./data/jan2018.json', 'r') as file:
    articles = json.load(file)

categories = {'seguranca': ['segurança','insegurança','seguranças','polícia','policias','crime', 'bombeiros',
                            'prisão','roubo','crimes','delegacia','suspeitos','suspeito','pm','presos',
                            'rebelião','detentos','tráfico'],
            'justica' : ['justiça','advogados','tribunal','estado','partido','stf','lei','união','tjgo','ofício',
                        'juiz','decisão','julgamento','audiência','audiências','testemunha','testemunhas',
                        'ministra','ministro','stf','jurisprudência','ordem'],
            'educacao' : ['educação','aluno','alunos','ensino','escolas','novatos',
                          'matrícula','prématrícula','mec','ufg','concurso'],
            'cidade' : ['cidade','município','habitantes','região','infraestrutura','prefeitura','reparos',
                        'goiânia','obras','celg'],
            'transito' : ['carro','transito','veículo','veículos','motorista','motoristas','acidente','km',
                          'rua','motociclista','rodovia','sentido','avenida','pedestres','pedestre','multa',
                          'colisão','detrango','cruzamento','rodovias','quilômetros','trecho','pedágios',
                         'triunfo','sinalização','transporte','frota'],
            'saude': ['saúde','casos','tratamento','doença','cirurgias','tratamento','doenças','vida','hospital',
                     'pacientes']
             }

cc = ContentCategorization()
print('Initialization ...')
# cc.initialization(articles,categories)
print('Extracting representativity ...')
# cc.extract_representativity()
print('Extracting representativity using version 2...')
[articles_v2, categories_v2, bag_sentences_v2, unique_sentences_v2, bag_words_v2, unique_words_v2, bag_map] = articles_v1_to_v2(articles, categories)
tc = TextCategorization(articles_v2, categories_v2, threshold=0.7, min_words_per_sentence=1, min_sentences_per_article=1)
[v2_representative_sentences, v2_representative_words] = representative_v2_to_v1(tc.representative_sentences, bag_map, categories) 
[v2_unclassified_sentences, v2_unclassified_words] = unclassified_v2_to_v1(tc.unclassified_sentences, bag_map)
# [v2_bag_words, v2_bag_sentences] = corpus_v2_to_v1(tc.corpus_word, tc.corpus_sentences,  bag_map)
print('Inserting version 2 representativity...')
cc.bag_words = bag_words_v2
cc.bag_sentences = bag_sentences_v2
cc.unclassified_words = v2_unclassified_words
cc.representative_words = v2_representative_words
cc.representative_sentences = v2_representative_sentences
cc.unclassified_sentences = v2_unclassified_sentences
cc.unique_sentences = unique_sentences_v2
cc.unique_words = unique_words_v2
cc.index_words = {}
cc.words_index = {}
cc.index_sentences = {}
cc.sentences_index = {}
cc.wsm = {}
cc.ssm = {}
print('Computing global frequency factor ...')
cc.global_frequency_factor()
print('Computing log likelihood factor ...')
cc.log_likelihood_factor()
print('Computing words weight ...')
cc.words_weight()

cat = sys.argv[1]

print('Extending '+cat.capitalize()+' representativity ...')
cc.extend_representativity(cat)

print('Saving Object File ...')
pickle.dump( cc, open( "cc_"+cat.capitalize()+".object", "wb" ) )

# cc2 = pickle.load( open( "cc.object", "rb" ) )

