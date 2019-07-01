import unittest
from source.helpers.text_categorization.init_helpers import *

class TestRepresentativeExtraction(unittest.TestCase):
  
    def setUp(self):
        pass

    def test_initizalize_bag_words_sentences(self):
        """
        Tests initizalize_bag
        """
        articles = [
            [[0, 1], [1, 2],  [0, 1, 2]],
            [[1, 0], [1, 2], [2, 0]]
        ]

        expected_sentences = [(0, 1), (1, 2), (0, 1, 2), (1, 0), (1, 2), (2, 0)]
        expected_words = [0, 1, 1, 2, 0, 1, 2, 1, 0, 1, 2, 2, 0]

        [sentences, words] = initizalize_bag_words_sentences(articles)
        self.assertEqual(sentences, expected_sentences)
        self.assertEqual(words, expected_words)
    
    def test_initizalize_bag_with_low_amount_words(self):
        """
        Tests initizalize_bag_with_low_amount_of_words
        """
        minWords = 3
        articles = [
            [[0, 1], [1, 2], [0, 1, 2]],
            [[1, 0], [2, 0], [0, 1, 2]]
        ]

        expected_sentences = [(0, 1, 2), (0, 1, 2)]
        expected_words = [0, 1, 2, 0, 1, 2]

        [sentences, words] = initizalize_bag_words_sentences(articles,min_words_per_sentence=minWords)
        
        self.assertEqual(sentences, expected_sentences)
        self.assertEqual(words, expected_words)
    
    def test_initizalize_bag_with_unused_filtered_words(self):
        """
        Tests initizalize_bag_with_low_amount_of_sentences
        """
        minSentences = 3
        articles = [
                [[0, 1], [1, 2]],
                [[4, 3], [5, 3], [3, 4, 5]]
               ]

        expected_sentences = [(4, 3), (5, 3), (3, 4, 5)]
        expected_words = [4, 3, 5, 3, 3, 4, 5]

        [sentences, words] = initizalize_bag_words_sentences(articles,min_sentences_per_article=minSentences)
        self.assertEqual(sentences, expected_sentences)
        self.assertEqual(words, expected_words)
    
    def test_initizalize_bag_with_low_amount_of_sentences(self):
        """
        Tests initizalize_bag_with_low_amount_of_sentences
        """
        minSentences = 3
        articles = [
            [[0, 1], [1, 2]],
            [[1, 0], [2, 0], [0, 1, 2]]
        ]

        expected_sentences = [(1, 0), (2, 0), (0, 1, 2)]
        expected_words = [1, 0, 2, 0, 0, 1, 2]

        [sentences, words] = initizalize_bag_words_sentences(articles,min_sentences_per_article=minSentences)
        self.assertEqual(sentences, expected_sentences)
        self.assertEqual(words, expected_words)

    def test_initizalize_words_sentences_indexes(self):
        """
        Tests initizalize_index
        """
        bag_sentences = [(0, 1), (1, 2), (0, 1, 2), (1, 0), (1, 2), (2, 0)]
        bag_words = [0, 1, 1, 2, 0, 1, 2, 1, 0, 1, 2, 2, 0]

        expected_sentences = [(0, 1), (0, 1, 2), (1, 0), (1, 2), (2, 0)]
        expected_words = [0, 1, 2]

        [sentences, words] = initizalize_words_sentences_indexes(bag_sentences,bag_words)
        self.assertEqual(sentences, expected_sentences)
        self.assertEqual(words, expected_words)

    def test_split_representative_senteces(self):
        """
        Tests Sentence Weight
        """
        sentences =  [(0, 4), (2, 4, 5), (1, 6, 7), (8, 9), (0, 1)]
        categories = [(0, 2), (1, 3)]

        expected_representative_sentences = [
            [(0,4),(2,4,5),(0,1)],
            [(1,6,7),(0,1)]
        ]
        expected_unclassified_sentences = [(8,9)]
        [representative_sentences, unclassified_sentences] = split_representative_senteces(sentences,categories)
        
        self.assertEqual(representative_sentences, expected_representative_sentences)
        self.assertEqual(unclassified_sentences, expected_unclassified_sentences)
    
    def test_split_representative_senteces_with_letters(self):
        """
        Tests Sentence Weight
        """
        sentences =  [('a', 'e'), ('c', 'e', 'f'), ('b', 'g', 'h'), ('i', 'j'), ('a', 'b')]
        categories = [('a', 'c'), ('b', 'd')]

        expected_representative_sentences = [[('a','e'),('c','e','f'),('a','b')], [('b','g','h'),('a','b')]]
        expected_unclassified_sentences = [('i','j')]
        [representative_sentences, unclassified_sentences] = split_representative_senteces(sentences,categories)
        
        self.assertEqual(representative_sentences, expected_representative_sentences)
        self.assertEqual(unclassified_sentences, expected_unclassified_sentences)
    
    def test_get_representative_bag_of_words(self):
        representative_sentences = [[(0, 0, 3), (0, 1), (0, 3, 2), (1, 2, 2, 5, 3), (2, 0)], [(0, 0, 3), (0, 1), (0, 3, 2), (1, 1, 5), (1, 2, 2, 5, 3), (4, 3), (5, 3)]]
        expected_representative_bag_words = [[0, 0, 3, 0, 1, 0, 3, 2, 1, 2, 2, 5, 3, 2, 0], [0, 0, 3, 0, 1, 0, 3, 2, 1, 1, 5, 1, 2, 2, 5, 3, 4, 3, 5, 3]]

        representative_bag_words = get_representative_bag_of_words(representative_sentences)

        self.assertEqual(representative_bag_words, expected_representative_bag_words)
    
    def test_compute_term_frequencies(self):
        """
        Tests compute_term_frequency
        """
        bag_words = [0, 1, 1, 2, 0, 1, 2, 1, 0, 1, 2, 2, 0, 4]
        corpus_words = [0, 1, 2, 3, 4]
    
        expected_word_frequency = [4, 5, 4, 0, 1]

        frequency = compute_term_frequencies(corpus_words,bag_words)
        self.assertEqual(frequency, expected_word_frequency)

    def test_compute_within_category_word_frequencies(self):
        """
        Tests compute_within_category_word_frequency
        """
        corpus_words = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        representative_words = [
                                [0, 4, 2, 4, 5, 0, 1],
                                [1, 6, 7, 0, 1]
                               ]

        expected_word_frequencies = [
            [2, 1, 1, 0, 2, 1, 0, 0, 0],
            [1, 2, 0, 0, 0, 0, 1, 1, 0]
        ]

        frequencies = compute_within_category_word_frequencies(corpus_words, representative_words)
        self.assertEqual(frequencies, expected_word_frequencies)

    def test_compute_category_frequencies(self):
        """
        Tests compute_category_frequency
        """
        representative_words = [
                                [1,2,4],
                                [1,2,3],
                                [2,4]
                               ]
        corpus_words = [0, 1, 2, 3, 4, 5, 6, 7]

        expected_category_frequencies = [0, 2, 3, 1, 2, 0, 0, 0]

        category_frequency = compute_category_frequencies(corpus_words,representative_words)
        self.assertEqual(category_frequency, expected_category_frequencies)

    def test_compute_category_frequencies_with_duplicated_words(self):
        """
        Tests compute_category_frequency
        """
        representative_words = [
            [0, 0, 3, 0, 1, 0, 3, 2, 1, 2, 2, 5, 3, 2, 0],
            [0, 0, 3, 0, 1, 0, 3, 2, 1, 1, 5, 1, 2, 2, 5, 3, 4, 3, 5, 3]
        ]
        corpus_words = [0, 1, 2, 3, 4, 5, 6, 7]

        expected_category_frequencies = [2, 2, 2, 2, 1, 2, 0, 0]

        category_frequency = compute_category_frequencies(corpus_words,representative_words)
        self.assertEqual(category_frequency, expected_category_frequencies)

    def test_compute_inverse_category_frequencies(self):
        """
        Tests compute_inverse_category_frequency
        """

        category_frequencies = [0, 1, 2, 3, 4, 5]
        number_categories = 5

        expected_inverse_category_frequencies = [0, 1.6094379124341003, 0.916290731874155, 0.5108256237659905, 0.2231435513142097, 0]

        inverse_category_frequencies = compute_inverse_category_frequencies(category_frequencies, number_categories)
        self.assertEqual(inverse_category_frequencies, expected_inverse_category_frequencies)

    def test_compute_word_weights(self):
        """
        Tests compute_word_weight
        """
        term_frequencies = [
            [1,2,3],
            [4,0,6]
        ]
        inverse_category_frequencies = [1.6094379124341003, 0.916290731874155, 0.5108256237659905]

        expected_word_weights = [
            [1.6094379124341003, 1.83258146374831, 1.5324768712979715],
            [6.437751649736401, 0.0, 3.064953742595943]
        ]

        word_weights = compute_word_weights(term_frequencies, inverse_category_frequencies)
        self.assertEqual(word_weights, expected_word_weights)

    def test_compute_sentence_weights(self):
        """
        Tests compute_sentence_weight
        """
        sentences = [
                     [(0, 2), (0, 1, 2)],
                     [(0, 1)]
                    ]
        
        word_weights = [
                         [1.6, 0.9, 0.8],
                         [1.6, 0.9, 0.0]
                       ]

        words_index = [0, 1, 2]

        expected_sentence_weights = [
            [1.2000000000000002, 1.0999999999999999],
            [1.25]
        ]

        sentence_weights = compute_sentence_weights(words_index, sentences, word_weights)
        self.assertEqual(sentence_weights, expected_sentence_weights)

    def test_filter_representative_category_sentences_case_1(self):
        """
        Tests filter_representative_sentences
        """
        sentences = [(0, 1), (0, 1, 2), (1, 2)]
        weight = [1.7210096880912051, 1.6581654158267938, 1.6825291675231409 ]
        threshold = 0.7

        expected_filtered_representative_sentences = [(0, 1), (1, 2)]
        expected_unclassified_sentences = [(0, 1, 2)]
        [filtered_representative_sentences, unclassified_sentences] = filter_representative_category_sentences(sentences, weight, threshold)
        
        self.assertEqual(filtered_representative_sentences, expected_filtered_representative_sentences)
        self.assertEqual(unclassified_sentences, expected_unclassified_sentences)

    def test_filter_representative_category_sentences_case_2(self):
        """
        Tests filter_representative_sentences
        """
        sentences = [(0, 0, 3), (0, 1), (0, 3, 2), (1, 1, 5), (1, 2, 2, 5, 3), (2, 0)]
        weight = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        threshold = 0.7

        expected_filtered_representative_sentences = [(0, 3, 2), (1, 1, 5), (1, 2, 2, 5, 3), (2, 0)]
        expected_unclassified_sentences = [(0, 0, 3), (0, 1)]
        [filtered_representative_sentences, unclassified_sentences] = filter_representative_category_sentences(sentences, weight, threshold)
        
        self.assertEqual(filtered_representative_sentences, expected_filtered_representative_sentences)
        self.assertEqual(unclassified_sentences, expected_unclassified_sentences)

    def test_filter_representative_category_sentences_case_3(self):
        """
        Tests filter_representative_sentences
        """
        sentences = [(4, 3), (0, 0, 3), (0, 3, 2), (1, 2, 2, 5, 3), (4, 5), (5, 3)]
        weight = [0.6931471805599453, 0.0, 0.0, 0.0, 0.6931471805599453, 0.0]
        threshold = 0.7

        expected_filtered_representative_sentences = [(4, 3), (1, 2, 2, 5, 3), (4, 5), (5, 3)]
        expected_unclassified_sentences = [(0, 0, 3), (0, 3, 2)]

        [filtered_representative_sentences, unclassified_sentences] = filter_representative_category_sentences(sentences, weight, threshold)
        
        self.assertEqual(filtered_representative_sentences, expected_filtered_representative_sentences)
        self.assertEqual(unclassified_sentences, expected_unclassified_sentences)
    
    def test_filter_representative_sentences(self):
        """
        Tests filter_representative_senteces
        """
        representative_sentences_weight = [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.6931471805599453, 0.6931471805599453, 0.0]
            ]

        initial_representative_sentences = [
            [(0, 0, 3), (0, 1), (0, 3, 2), (1, 2, 2, 5, 3), (2, 0)],
            [(0, 0, 3), (0, 1), (0, 3, 2), (1, 1, 5), (1, 2, 2, 5, 3), (4, 3), (5, 3)]
        ]
        unclassified_sentences = [(4, 5)]
        threshold = 0.7

        expected_representative_sentences = [[(0, 3, 2), (1, 2, 2, 5, 3), (2, 0)], [(0, 3, 2), (1, 1, 5), (1, 2, 2, 5, 3), (4, 3), (5, 3)]]
        expected_unclassified_sentences = [(4, 5), (0, 1), (0, 0, 3)]

        [representative_sentences,
         unclassified_sentences] = filter_representative_sentences(
                                        initial_representative_sentences, unclassified_sentences,
                                        representative_sentences_weight, threshold)
        
        self.assertEqual(representative_sentences, expected_representative_sentences)
        self.assertEqual(unclassified_sentences, expected_unclassified_sentences)
    
    # @unittest.skip("Waiting for scenario definitions")
    def test_extract_representative_senteces_case_1(self):
        """
        Tests extract_representative_senteces
        """
        articles = [ 
            [[1,2,3],[5,16,9],[6,18],[6,6,18]],
            [[3,9,14],[12,18],[0,1,8],[11,2,4,7]],
            [[1,7,1],[0,0,14],[17,8,14],[10,4,1]],
            [[2,4,9],[6,12],[4,8,9],[6,6,6]],
            [[17,13,3],[4,19,3],[14,10,8],[15,13,0]]
        ]

        categories = [(0,15),(5,10),(3,17), (11,7)]

        threshold = 1
        min_words_per_sentence = 1
        min_sentences_per_article = 1 

        expected_representative_sentences = [
            [(0, 0, 14), (0, 1, 8), (15, 13, 0)],
            [(5, 16, 9), (10, 4, 1), (14, 10, 8)],
            [(1, 2, 3), (3, 9, 14), (4, 19, 3), (17, 8, 14), (17, 13, 3)],
            [(1, 7, 1), (11, 2, 4, 7)]
        ]
        expected_unclassified_sentences = [(4, 8, 9), (6, 6, 18), (2, 4, 9), (6, 18), (6, 6, 6), (12, 18), (6, 12)]

        [representative_sentences, unclassified_sentences] = extract_representative_setences(articles, categories, threshold, min_words_per_sentence, min_sentences_per_article)
        
        self.assertEqual(representative_sentences, expected_representative_sentences)
        self.assertEqual(unclassified_sentences, expected_unclassified_sentences)
    
    # @unittest.skip("Waiting for scenario definitions")
    def test_extract_representative_senteces_case_2(self):
        """
        Tests extract_representative_senteces, with 0.7 of threshold
        """
        articles = [ 
            [[1,2,3],[5,16,9],[6,18],[6,6,18]],
            [[3,9,14],[12,18],[0,1,8],[11,2,4,7]],
            [[1,7,1],[0,0,14],[17,8,14],[10,4,1]],
            [[2,4,9],[6,12],[4,8,9],[6,6,6]],
            [[17,13,3],[4,19,3],[14,10,8],[15,13,0]]
        ]

        categories = [(0,15),(5,10),(3,17), (11,7)]

        threshold = 0.7
        min_words_per_sentence = 1
        min_sentences_per_article = 1 

        expected_representative_sentences = [
            [(0, 0, 14), (15, 13, 0)],
            [(5, 16, 9), (14, 10, 8)],
            [(3, 9, 14), (4, 19, 3), (17, 13, 3)],
            [(11, 2, 4, 7)]
        ]
        expected_unclassified_sentences = [
            (4, 8, 9), (6, 6, 18), (2, 4, 9), (6, 18), (0, 1, 8), (17, 8, 14),
            (1, 7, 1), (6, 6, 6), (1, 2, 3), (12, 18), (10, 4, 1), (6, 12)
        ]

        [representative_sentences, unclassified_sentences] = extract_representative_setences(articles, categories, threshold, min_words_per_sentence, min_sentences_per_article)
        
        self.assertEqual(representative_sentences, expected_representative_sentences)
        self.assertEqual(unclassified_sentences, expected_unclassified_sentences)
    
    # @unittest.skip("Waiting for scenario definitions")
    def test_extract_representative_senteces_case_3_with_letters(self):     
        """
        Tests extract_representative_senteces, with 'Z'.'g' of threshold
        """
        articles = [ 
            [['a','b','c'],['e','p','i'],['f','r'],['f','f','r']],
            [['c','i','n'],['l','r'],['Z','a','h'],['k','b','d','g']],
            [['a','g','a'],['Z','Z','n'],['q','h','n'],['j','d','a']],
            [['b','d','i'],['f','l'],['d','h','i'],['f','f','f']],
            [['q','m','c'],['d','s','c'],['n','j','h'],['o','m','Z']]
        ]

        categories = [('Z','o'),('e','j'),('c','q'), ('k','g')]

        threshold = 0.7
        min_words_per_sentence = 1
        min_sentences_per_article = 1 

        expected_representative_sentences = [
            [('Z', 'Z', 'n'), ('o', 'm', 'Z')],
            [('e', 'p', 'i'), ('n', 'j', 'h')],
            [('c', 'i', 'n'), ('d', 's', 'c'), ('q', 'm', 'c')],
            [('k', 'b', 'd', 'g')]
        ]
        expected_unclassified_sentences = [
            ('d', 'h', 'i'), ('f', 'f', 'r'), ('b', 'd', 'i'), ('f', 'r'), ('Z', 'a', 'h'), ('q', 'h', 'n'),
            ('a', 'g', 'a'), ('f', 'f', 'f'), ('a', 'b', 'c'), ('l', 'r'), ('j', 'd', 'a'), ('f', 'l')
        ]

        [representative_sentences, unclassified_sentences] = extract_representative_setences(articles, categories, threshold, min_words_per_sentence, min_sentences_per_article)

        set_A = set(expected_unclassified_sentences)
        set_B = set(unclassified_sentences)

        self.assertEqual(representative_sentences, expected_representative_sentences)
        self.assertEqual(len(set_A - set_B), 0)
        self.assertEqual(len(set_B - set_A), 0)
        # self.assertEqual(unclassified_sentences, expected_unclassified_sentences)