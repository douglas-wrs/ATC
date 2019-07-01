import source.helpers.text_categorization.init_helpers as init

class TextCategorization(object):

    representative_sentences = None
    unclassified_sentences = None
    corpus_word = None
    corpus_sentences = None

    def __init__(self, articles, categories, threshold, min_words_per_sentence, min_sentences_per_article):
        [self.representative_sentences,
        self.unclassified_sentences] = init.extract_representative_setences(articles, categories, threshold, min_words_per_sentence, min_sentences_per_article)

    def fit(self):
        pass

    def classify(self):
        pass

    def accurary(self):
        pass