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
        print("Share of topics = " + str(goodLdaModel.get_document_topics(bow)))


def my_topic_coherence(top_words, vec_model):
    sims = []
    for v, w in itertools.combinations(top_words, 2):
        sims.append(vec_model.similarity(v, w))
        #print(v + " / " + w + " => " + str(sims[-1]))
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
        tt_u_mass = m.top_topics(corpus=corpus, texts=texts, dictionary=dictionary, coherence='u_mass',
                     topn=5, processes=4)
        l.append((get_topn_pertopic(m, 0, 5), my_topic_coherence(get_topn_pertopic(m, 0, 5), w2v_model), tt_u_mass[0][1]))
        l.append((get_topn_pertopic(m, 1, 5), my_topic_coherence(get_topn_pertopic(m, 1, 5), w2v_model), tt_u_mass[1][1]))
    return l


def compute_exp_k_times(k):
    gmt0 = []  # good model topic 0
    gmt1 = []
    bmt0 = []
    bmt1 = []  # bad model topic 1
    for i in range(0, k):
        a = compute_exp()
        for b in a:
            print(b)

        gmt0.append(a[0])
        gmt1.append(a[1])
        bmt0.append(a[2])
        bmt1.append(a[3])
    #return gmt0, gmt1, bmt0, bmt1


compute_exp_k_times(5)

#print_docs_topics(texts, goodLdaModel, dictionary)

