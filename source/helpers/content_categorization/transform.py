import json

def articles_v1_to_v2 (articles_v1,categories_v1):
    articles_v2 = []
    bag_words_v2 = []

    for art in articles_v1['articles']:
        articles_v2.append(art['sentences'])
        for sent in art['sentences']:
            bag_words_v2.extend(sent)

    corpus_bag = set(bag_words_v2)
    bag_map = dict(zip(corpus_bag, range(0,len(corpus_bag))))

    mapped_articles_v2 = []
    for art in articles_v2:
        art_aux = []
        for sent in art:
            sent_aux = []
            for word in sent:
                sent_aux.append(bag_map.get(word))
            art_aux.append(sent_aux)
        mapped_articles_v2.append(art_aux)
    
    categories_v2 = []
    for cat in categories_v1.values():
        keywords = []
        for key in cat:
            keywords.append(bag_map.get(key))
        categories_v2.append(tuple(keywords))
	
    bag_sentences_v2 = []
    bag_words_v2 = []
    for a in articles_v1['articles']:
        for s in a['sentences']:
            if (len(s) >= 1):
                bag_sentences_v2.append(tuple(s))
                bag_words_v2.extend(s)
    
    unique_sentences_v2 = set(bag_sentences_v2)
    unique_words_v2 = set(bag_words_v2)

    return [mapped_articles_v2, categories_v2, bag_sentences_v2, unique_sentences_v2, bag_words_v2, unique_words_v2, bag_map]

def corpus_v2_to_v1(corpus_word, corpus_sentences,  bag_map):
    map_bag = dict(zip(bag_map.values(),bag_map.keys()))
    
    bag_sentences_v2 = []
    for a in corpus_sentences:
        aux = []
        for w in a:
            aux.append(map_bag[w])
        bag_sentences_v2.append(aux)
    
    bag_words_v2 = []
    for w in corpus_word:
        bag_words_v2.append(map_bag[w])
    return [bag_words_v2, bag_sentences_v2]

def representative_v2_to_v1 (representative_v2, bag_map, categories):
    map_bag = dict(zip(bag_map.values(),bag_map.keys()))

    representative_v1 = []
    for cat in representative_v2:
        categ = []
        for sent in cat:
            sente = []
            for w in sent:
                sente.append(map_bag.get(w))
            categ.append(tuple(sente))
        representative_v1.append(categ)

    representative_sentences_v1 = dict(zip(categories.keys(),representative_v1))
    new_re = []
    for cat,sent in representative_sentences_v1.items():
        for i in sent:
            new_re.append(tuple(i))
    
    new_representative_sentences_v1 = [list(i) for i in new_re]

    representative_words_v1 = list(sum(new_representative_sentences_v1, []))

    return [representative_sentences_v1, representative_words_v1]

def unclassified_v2_to_v1 (unclassified_v2, bag_map):
    map_bag = dict(zip(bag_map.values(),bag_map.keys()))
    unclassified_v1 = []
    for sent in unclassified_v2:
        sente = []
        for word in sent:
            sente.append(map_bag.get(word))
        unclassified_v1.append(tuple(sente))


    new_un_sentences = [list(i) for i in unclassified_v1]
    unclassified_words = list(sum(new_un_sentences, []))

    return [unclassified_v1, unclassified_words]