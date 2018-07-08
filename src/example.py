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
        intra_ts.append(intra_topic_sim(top_words, vec_model))
    # inter
    inter_ts.append(inter_topics_sim(top_words_list, vec_model))
    return sum(intra_ts)*len(inter_ts)/(len(intra_ts)*sum(inter_ts))


def intra_topic_sim(tw, vec_model):
    """
    computes intra topic coherence given a list of top words (tw)
    :param tw: top_words
    :return:
    """
    sims = []
    for v, w in itertools.combinations(tw, 2):
        sims.append(vec_model.similarity(v, w)+1)
        # print(v + " / " + w + " => " + str(sims[-1]))
    return sum(sims)/len(sims)


def inter_topics_sim(twl, vec_model):
    """
    Computes similarity between one topic and all the others
    :param twl: top_words list: top_words of other topics
    :param tw: top_words of the topic to compare
    :param vec_model:
    :return:
    """
    sims = []
    for v, w in itertools.combinations(twl, 2):
        print(v, w)
        sims.append(vec_model.n_similarity(v, w)+1)
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


def get_lda_model(regenmod, lmod):
    list_models = []
    if os.path.isfile(lmod) and not regenmod:
        model = LdaModel.load(lmod)
    else:
        model = LdaModel(corpus=corpus, id2word=dictionary, iterations=50, num_topics=2)
    list_models.append(model)
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
    model_name = "../lmod"
    regen_models = True
    models = get_lda_model(regen_models, model_name)
    l = []
    nb_topics = 2
    for m in models:
        cm_umass = coherencemodel.CoherenceModel(model=m, corpus=corpus, texts=texts, coherence='u_mass')
        cm_cv = coherencemodel.CoherenceModel(model=m, corpus=corpus, texts=texts, coherence='c_v')
        top_words_list = []
        for k in range(0, nb_topics):
            topn = get_topn_pertopic(m, k, 5)
            top_words_list.append(topn)

        l.append((top_words_list, my_topic_coherence(top_words_list, w2v_model), cm_umass.get_coherence(), cm_cv.get_coherence()))
    return l


def compute_exp_k_times(k):
    for i in range(0, k):
        a = compute_exp()
        for b in a:
            print(b)
    return a

compute_exp_k_times(10)

#print_docs_topics(texts, goodLdaModel, dictionary)

