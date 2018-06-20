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
        print(v + " / " + w + " => " + str(sims[-1]))
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
model_name = "../glmod"
regen_model = True  # False

if os.path.isfile(model_name) and not regen_model:
    goodLdaModel = LdaModel.load(model_name)
else:
    goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=500, num_topics=2)
    goodLdaModel.save(model_name)

print(len(goodLdaModel.get_topics()))

print("Word distribution in topics: ")
print("topic_0: " + str(goodLdaModel.print_topic(0, 12)))
print("topic_1: " + str(goodLdaModel.print_topic(1, 12)))

# top n words into list for 2 topics
topn0 = []
for word, freq in goodLdaModel.show_topic(0, 5):
    topn0.append(word)
topn1 = []
for word, freq in goodLdaModel.show_topic(1, 5):
    topn1.append(word)

print("loading w2vec")
modelName = '../GoogleNews-vectors-negative300.bin'
w2v_model = KeyedVectors.load_word2vec_format(modelName, binary=True)
print("w2vec loaded")

print(topn0)
print(my_topic_coherence(topn0, w2v_model))
print(topn1)
print(my_topic_coherence(topn1, w2v_model))

#print_docs_topics(texts, goodLdaModel, dictionary)

