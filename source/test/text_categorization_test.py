import unittest
from source.text_categorization import TextCategorization

class TestTextExtraction(unittest.TestCase):
  
    def setUp(self):
        pass
    
    # @unittest.skip("Waiting for scenario definitions")
    def test_init_case_1(self):
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

        demo = TextCategorization(articles, categories, threshold, min_words_per_sentence, min_sentences_per_article)
        
        self.assertEqual(demo.representative_sentences, expected_representative_sentences)
        self.assertEqual(demo.unclassified_sentences , expected_unclassified_sentences)