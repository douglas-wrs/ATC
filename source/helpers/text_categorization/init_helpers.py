# init_helpers.py

from functools import reduce
import operator
import nltk
import numpy
import math

def initizalize_bag_words_sentences(articles, min_words_per_sentence=1, min_sentences_per_article=1):
    """ Initizalize the bag of sentences and bag of words based on two filter, minimum number of
        sentences per articles and minimum number of words per sentence.

    Args:
        articles (list): The first parameter.
        min_words_per_sentence (integer): The second parameter.
        min_sentences_per_article (integer): The second parameter.

    Returns:
        bag_sentences: The return list of sentences that satifies the minimum number of
                        sentences per articles and minimum number of words per sentence.
        bag_words: The return the list of words from bag_sentences
    """
    bag_words, bag_sentences, filtered_sentences = [], [], []

    for article in articles:
        if len(article) >= min_sentences_per_article:
            filtered_sentences += list( filter( lambda sentences: len(sentences) >= min_words_per_sentence, article) )
            
    bag_sentences = list( map( tuple, filtered_sentences) )
    bag_words = list( reduce( list.__add__, filtered_sentences) )

    return [bag_sentences, bag_words]

def initizalize_words_sentences_indexes (bag_sentences, bag_words):
    """ initizalize_words_sentences_indexes
        and it is sorted by alphabetical or numerical order.

    Args:
        bag_sentences (list): The first parameter.
        bag_words (list): The second parameter.

    Returns:
        sentences: The return list of unique sentences in a bag of sentences
        words: The return the list of unique words from sentences
    """

    sentences = sorted( set( bag_sentences) )
    words = sorted( set( bag_words) )
    
    return [sentences, words]

def split_representative_senteces(sentences, categories):
    """ split_representative_senteces

    Args:
        sentences (list): The first parameter.
        categories (list): The second parameter.

    Returns:
        representative_sentences:
        unclassified_sentences:
    """
    representative_sentences = list(map(lambda keywords:
        list(filter(lambda sentence: len(set(sentence).intersection(set(keywords))) > 0, sentences))
    , categories))
    
    all_categories_sentences = reduce(list.__add__, representative_sentences)
    unclassified_sentences =   list(set(sentences) - set(all_categories_sentences))

    return [representative_sentences, unclassified_sentences]

def get_representative_bag_of_words(representative_sentences):
	return list(map(lambda sentences: list(reduce(list.__add__, map(list, sentences))), representative_sentences))

def compute_term_frequencies (corpus, words):
    """ compute_term_frequencies
    Args:
        corpus (list): The first parameter.
        words (list): The second parameter.

    Returns:
        term_frequencies: 
    """
    
    bag_words_corpus = list( reduce( list.__add__, [corpus, words]) )
    # print('\nbag_words_corpus:', bag_words_corpus)

    words_mapping = dict( nltk.FreqDist(bag_words_corpus) )
    # print('words_mapping:', words_mapping)

    map_values = [value for (_key, value) in sorted(words_mapping.items())]
    # print('map_values:', map_values)

    term_frequency = list( numpy.array( map_values ) -1 )
    # print('term_frequency:', term_frequency)

    return term_frequency

def compute_within_category_word_frequencies(corpus_words, representative_words):
    """ compute_within_category_word_frequencies

    Args:
        corpus_words (list): The first parameter.
        representative_words (list): The second parameter.

    Returns:
        word_frequencies: 
    """
    # print('\n')
    return list( map( lambda word: compute_term_frequencies(corpus_words, word), representative_words) )

def compute_category_frequencies(corpus_words, representative_words):
    """ compute_category_frequencies

    Args:
        corpus_words (list): The first parameter.
        representative_words (list): The second parameter.

    Returns:
        category_frequencies: 
    """

    representative_unique_words = list(map(list, map(set, representative_words)))
    representative_bag_words = reduce(operator.concat, representative_unique_words)
    
    bag_representative_words_corpus = list( reduce( list.__add__, [corpus_words, representative_bag_words]) )

    words_mapping = dict( nltk.FreqDist(bag_representative_words_corpus) )
    # print('words_mapping:', words_mapping)

    map_values = [value for (_key, value) in sorted(words_mapping.items())]
    # print('map_values:', map_values)

    category_frequency = list( numpy.array( map_values ) -1 )
    
    return category_frequency

def compute_inverse_category_frequencies(category_frequencies, number_categories):
    """ compute_inverse_category_frequencies

    Args:
        category_frequencies (list): The first parameter.
        number_categories (list): The second parameter.

    Returns:
        inverse_category_frequencies: 
    """
    return list( map( lambda frequency: numpy.log(number_categories) - numpy.log(frequency) if frequency else 0, category_frequencies) )

def compute_word_weights(term_frequencies, inverse_category_frequencies):
    """ compute_word_weights

    Args:
        term_frequencies (list): The first parameter.
        inverse_category_frequencies (list): The second parameter.

    Returns:
        word_weights: 
    """
    return list( map( lambda term_frequency: list( numpy.array(term_frequency)*numpy.array(inverse_category_frequencies) ), term_frequencies) )

def compute_sentence_weights(words, representative_sentences, word_weights):
    """ compute_sentence_weights

    Args:
        representative_sentences (list): The first parameter.
        word_weights (list): The second parameter.

    Returns:
        representative_sentences_weight: 
    """
    representative_sentences_weight = []
    for index in range(len(representative_sentences)):
        representative_sentences_weight.append(
            list(map(
                lambda sentence: numpy.array([word_weights[index][words.index(word)] for word in sentence]).mean(),
                representative_sentences[index]
                )
            )
        )

    return representative_sentences_weight

def filter_representative_category_sentences(representative_sentences, weights, threshold):
    """ filter_representative_sentences

    Args:
        sentences (list): The first parameter.
        weights (list): The second parameter.
        threshold (float): The third parameter.

    Returns:
        sentences_flags: 
    """
    qtd = len(weights) - math.floor(len(weights)*threshold)
    
    unclassified_sentences = []
    for _i in range(qtd):
        min_value_index = numpy.argmin(numpy.array(weights))
        
        weights.pop(min_value_index)
        unclassified_sentences.append(representative_sentences.pop(min_value_index))

    return [representative_sentences, unclassified_sentences]

def filter_representative_sentences(initial_representative_sentences, unclassified_sentences, representative_sentences_weight, threshold):
    representative_sentences = []
    number_of_categories = len(initial_representative_sentences)
    
    for i in range(number_of_categories):
        [representative,
         unclassified] = filter_representative_category_sentences(
                            initial_representative_sentences[i], representative_sentences_weight[i], threshold)
        representative_sentences.append(representative)
        unclassified_sentences.extend(unclassified)

    all_representative_sentences = reduce(operator.concat, representative_sentences)
    unclassified_size = len(unclassified_sentences)

    unclassified_sentences = list(set(unclassified_sentences) - set(all_representative_sentences))

    return [representative_sentences, unclassified_sentences]

def extract_representative_setences(articles, categories, threshold, min_words_per_sentence, min_sentences_per_article):
    """ split_representative_senteces

    Args:
        articles (list): The first parameter.
        categories (list): The second parameter.
        threshold (list): The third parameter.
        min_words_per_sentence (list): The 4th parameter.
        min_sentences_per_article (list): The 5th parameter.

    Returns:
        representative_sentences:
        unclassified_sentences:
    """
    
    number_of_categories = len(categories)

    # Coments for further thougts: "Eu avisei"
    [bag_sentences, bag_words] = initizalize_bag_words_sentences(articles, min_words_per_sentence, min_sentences_per_article)

    [sentences, words] = initizalize_words_sentences_indexes(bag_sentences, bag_words)
    [initial_representative_sentences, unclassified_sentences] = split_representative_senteces(sentences, categories)
    
    representative_words = get_representative_bag_of_words(initial_representative_sentences)

    term_frequencies = compute_within_category_word_frequencies(words, representative_words)
    category_frequencies = compute_category_frequencies(words, representative_words)
    inverse_category_frequencies = compute_inverse_category_frequencies(category_frequencies, number_of_categories)

    word_weight = compute_word_weights(term_frequencies, inverse_category_frequencies)  
    representative_sentences_weight = compute_sentence_weights(words, initial_representative_sentences, word_weight)

    [representative_sentences,
     unclassified_sentences] = filter_representative_sentences( initial_representative_sentences,
                                    unclassified_sentences, representative_sentences_weight, threshold)
    
    # Coments for further thougts: "Eu avisei"
    return [representative_sentences, unclassified_sentences]
