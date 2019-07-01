# init_helpers_doctest.txt
    >>> from source.helpers.text_categorization.init_helpers import *

## DOCTEST SETUP

    >>> articles = [
    ... [[1,2,3],[5,16,9],[6,18],[6,6,18]],
    ... [[3,9,14],[12,18],[0,1,8],[11,2,4,7]],
    ... [[1,7,1],[0,0,14],[17,8,14],[10,4,1]],
    ... [[2,4,9],[6,12],[4,8,9],[6,6,6]],
    ... [[17,13,3],[4,19,3],[14,10,8],[15,13,0]]
    ... ]
    >>> categories = [(0,15),(5,10),(3,17), (11,7)]
    >>> threshold = 0.7 
    >>> number_of_categories = len(categories)
    >>> number_of_categories
    4

## INITIALIZE BAGS

    >>> [bag_sentences, bag_words] = initizalize_bag_words_sentences(articles)
    >>> bag_sentences
    [(1, 2, 3), (5, 16, 9), (6, 18), (6, 6, 18), (3, 9, 14), (12, 18), (0, 1, 8), (11, 2, 4, 7), (1, 7, 1), (0, 0, 14), (17, 8, 14), (10, 4, 1), (2, 4, 9), (6, 12), (4, 8, 9), (6, 6, 6), (17, 13, 3), (4, 19, 3), (14, 10, 8), (15, 13, 0)]
    >>> bag_words
    [1, 2, 3, 5, 16, 9, 6, 18, 6, 6, 18, 3, 9, 14, 12, 18, 0, 1, 8, 11, 2, 4, 7, 1, 7, 1, 0, 0, 14, 17, 8, 14, 10, 4, 1, 2, 4, 9, 6, 12, 4, 8, 9, 6, 6, 6, 17, 13, 3, 4, 19, 3, 14, 10, 8, 15, 13, 0]

## INITIALIZE INDEXES

    >>> [bag_sentences, bag_words] = initizalize_words_sentences_indexes(bag_sentences, bag_words)
    >>> bag_sentences
    [(0, 0, 14), (0, 1, 8), (1, 2, 3), (1, 7, 1), (2, 4, 9), (3, 9, 14), (4, 8, 9), (4, 19, 3), (5, 16, 9), (6, 6, 6), (6, 6, 18), (6, 12), (6, 18), (10, 4, 1), (11, 2, 4, 7), (12, 18), (14, 10, 8), (15, 13, 0), (17, 8, 14), (17, 13, 3)]
    >>> bag_words
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

## SPLIT REPRESENTAVIE SENTENCES

    >>> [representative_sentences,
    ... unclassified_sentences] = split_representative_senteces(bag_sentences, categories)
    >>> categories
    [(0, 15), (5, 10), (3, 17), (11, 7)]
    >>> representative_sentences
    [[(0, 0, 14), (0, 1, 8), (15, 13, 0)], [(5, 16, 9), (10, 4, 1), (14, 10, 8)], [(1, 2, 3), (3, 9, 14), (4, 19, 3), (17, 8, 14), (17, 13, 3)], [(1, 7, 1), (11, 2, 4, 7)]]
    >>> unclassified_sentences
    [(4, 8, 9), (6, 6, 18), (2, 4, 9), (6, 18), (6, 6, 6), (12, 18), (6, 12)]
    >>> len(bag_sentences) == len(unclassified_sentences) + 13
    True

## GET REPRESENTATIVE WORDS

    >>> representative_words = get_representative_bag_of_words(representative_sentences)
    >>> representative_words
    [[0, 0, 14, 0, 1, 8, 15, 13, 0], [5, 16, 9, 10, 4, 1, 14, 10, 8], [1, 2, 3, 3, 9, 14, 4, 19, 3, 17, 8, 14, 17, 13, 3], [1, 7, 1, 11, 2, 4, 7]]

## WORD FREQUENCIES

    >>> term_frequencies = compute_within_category_word_frequencies(bag_words, representative_words)
    >>> term_frequencies
    [[4, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 0, 0, 0, 1, 0, 1, 0, 0, 0], [0, 1, 1, 4, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2, 0, 0, 2, 0, 1], [0, 2, 1, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]

## CATEGORY FREQUENCIES

    >>> category_frequencies = compute_category_frequencies(bag_words, representative_words)
    >>> category_frequencies
    [1, 4, 2, 1, 3, 1, 0, 1, 3, 2, 1, 1, 0, 2, 3, 1, 1, 1, 0, 1]

## CATEGORY FREQUENCIES

    >>> inverse_category_frequencies = compute_inverse_category_frequencies(category_frequencies, number_of_categories)
    >>> inverse_category_frequencies
    [1.3862943611198906, 0.0, 0.6931471805599453, 1.3862943611198906, 0.2876820724517808, 1.3862943611198906, 0, 1.3862943611198906, 0.2876820724517808, 0.6931471805599453, 1.3862943611198906, 1.3862943611198906, 0, 0.6931471805599453, 0.2876820724517808, 1.3862943611198906, 1.3862943611198906, 1.3862943611198906, 0, 1.3862943611198906]

## WORD WEIGHTS

    >>> word_weights = compute_word_weights(term_frequencies, inverse_category_frequencies)
    >>> word_weights
    [[5.545177444479562, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2876820724517808, 0.0, 0.0, 0.0, 0.0, 0.6931471805599453, 0.2876820724517808, 1.3862943611198906, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.2876820724517808, 1.3862943611198906, 0.0, 0.0, 0.2876820724517808, 0.6931471805599453, 2.772588722239781, 0.0, 0.0, 0.0, 0.2876820724517808, 0.0, 1.3862943611198906, 0.0, 0.0, 0.0], [0.0, 0.0, 0.6931471805599453, 5.545177444479562, 0.2876820724517808, 0.0, 0.0, 0.0, 0.2876820724517808, 0.6931471805599453, 0.0, 0.0, 0.0, 0.6931471805599453, 0.5753641449035616, 0.0, 0.0, 2.772588722239781, 0.0, 1.3862943611198906], [0.0, 0.0, 0.6931471805599453, 0.0, 0.2876820724517808, 0.0, 0.0, 2.772588722239781, 0.0, 0.0, 0.0, 1.3862943611198906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

## SENTENCES WEIGHTS

    >>> representative_sentences_weight = compute_sentence_weights(representative_sentences, word_weights)
    >>> representative_sentences_weight
    [[3.7926789871369686, 1.944286505643781, 2.5415396620531325], [1.1552453009332422, 1.0200902648971872, 1.115984289047781], [2.0794415416798357, 2.271229589981023, 2.406384626017078, 1.2118783131983746, 3.0036377824264293], [0.9241962407465937, 1.2849280840928494]]

## FILTER REPRESENTAVIE SENTENCES

    >>> threshold
    0.7
    >>> [representative_sentences,
    ... unclassified_sentences] = filter_representative_sentences(representative_sentences, unclassified_sentences, representative_sentences_weight, threshold)
    >>> representative_sentences
    [[(0, 0, 14), (15, 13, 0)], [(5, 16, 9), (14, 10, 8)], [(3, 9, 14), (4, 19, 3), (17, 13, 3)], [(11, 2, 4, 7)]]
    >>> unclassified_sentences
    [(4, 8, 9), (6, 6, 18), (2, 4, 9), (6, 18), (0, 1, 8), (17, 8, 14), (1, 7, 1), (6, 6, 6), (1, 2, 3), (12, 18), (10, 4, 1), (6, 12)]
    >>> len(bag_sentences) == len(unclassified_sentences) + 8
    True
