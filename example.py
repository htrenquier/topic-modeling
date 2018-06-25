from gensim.models.ldamodel import LdaModel
from gensim.models import KeyedVectors
from gensim.models import coherencemodel
from gensim.corpora.dictionary import Dictionary
import itertools
import os


def print_docs_topics(dtexts, dlda_model, ddictionary):
    for text in dtexts:
        bow = dictionary.doc2bow(text)
        cm = coherencemodel.CoherenceModel(dlda_model, texts=dtexts, corpus=bow, dictionary=ddictionary)
        print("")
        print("Coherence = " + str(cm.get_coherence()))
        print("Bag of word = " + str(text))
        print("Share of topics = " + str(dlda_model.get_document_topics(bow)))


def my_topic_coherence(top_words_list, vec_model):
    """

    :param top_words_list: list of top_words, top_words is a list of words representing 1 topic
    :param vec_model:
    :return:
    """
    inter_ts = []
    intra_ts = []
    for top_words in top_words_list:
        #intra
        intra_ts.append(intra_topic_coh(top_words, vec_model))
        #inter
        inter_ts.append(inter_topics_coh(top_words_list, top_words, vec_model))
    return sum(intra_ts)*len(inter_ts)/(len(intra_ts)*sum(inter_ts))


def intra_topic_coh(tw, vec_model):
    """
    computes intra topic coherence given a list of top words (tw)
    :param tw: top_words
    :return:
    """
    sims = []
    for v, w in itertools.combinations(tw, 2):
        sims.append(vec_model.similarity(v, w))
        # print(v + " / " + w + " => " + str(sims[-1]))
    return sum(sims)/len(sims)


def inter_topics_coh(twl, tw, vec_model):
    sims = []
    for other_top_words in twl:
        if other_top_words != tw:
            for w in tw:
                for v in other_top_words:
                    sims.append(vec_model.similarity(w, v))
    return sum(sims)/len(sims)


texts = [['human', 'interface', 'computer'],
         ['survey', 'user', 'computer', 'system', 'response', 'time'],
         ['eps', 'user', 'interface', 'system'],
         ['system', 'human', 'system', 'eps'],
         ['user', 'response', 'time'],
         ['trees'],
         ['graph', 'trees'],
         ['graph', 'minors', 'trees'],
         ['graph', 'minors', 'survey']]

dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

print("loading w2vec")
modelName = '../GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(modelName, binary=True)
print("w2vec loaded")


def get_lda_models(regenmod,glmod,blmod):
    list_models = []
    if os.path.isfile(glmod) and not regenmod:
        goodLdaModel = LdaModel.load(glmod)
    else:
        goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=2)
        goodLdaModel.save(glmod)
    list_models.append(goodLdaModel)
    if os.path.isfile(blmod) and not regenmod:
        badLdaModel = LdaModel.load(blmod)
    else:
        badLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=1, num_topics=2)
        badLdaModel.save(blmod)
    list_models.append(badLdaModel)
    return list_models


def print_model_topics(m):
    print(str(len(m.get_topics())) + " topics")
    print("Word distribution in topics: ")
    print("topic_0: " + str(m.print_topic(0, 12)))
    print("topic_1: " + str(m.print_topic(1, 12)))


def get_topn_pertopic(m,t,n):
    topn = []
    for word, freq in m.show_topic(t, n):
        topn.append(word)
    return topn





def compute_exp():
    good_model_name = "../glmod"
    bad_model_name = "../blmod"
    regen_models = True  # False
    models = get_lda_models(regen_models, good_model_name, bad_model_name)
    l = []
    for m in models:
        tt_u_mass = m.top_topics(corpus=corpus, texts=texts, dictionary=dictionary, coherence='c_v',
                     topn=5, processes=4)
        l.append((get_topn_pertopic(m, 0, 5), my_topic_coherence(get_topn_pertopic(m, 0, 5), w2v_model), tt_u_mass[0][1]))
        l.append((get_topn_pertopic(m, 1, 5), my_topic_coherence(get_topn_pertopic(m, 1, 5), w2v_model), tt_u_mass[1][1]))
    return l


def compute_exp_k_times(k):
    gmc = []        #good model my coherence
    gumass = []
    bmc = []
    bumass = []     #bad model umass coherence
    for i in range(0, k):
        a = compute_exp()
        for b in a:
            print(b)

        gmc.append(a[0][1])
        gmc.append(a[1][1])
        gumass.append(a[0][2])
        gumass.append(a[1][2])
        bmc.append(a[2][1])
        bmc.append(a[3][1])
        bumass.append(a[2][2])
        bumass.append(a[3][2])
    #return gmc, gumass, bmc, bumass
    return sum(gmc) / len(gmc), \
           sum(gumass) / len(gumass), \
           sum(bmc) / len(bmc), \
           sum(bumass) / len(bumass)


print(compute_exp_k_times(5))

#print_docs_topics(texts, goodLdaModel, dictionary)

