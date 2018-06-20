from gensim.models.ldamodel import LdaModel
from gensim.models import coherencemodel
from gensim.corpora.dictionary import Dictionary
import os

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
regen_model = False

if os.path.isfile(model_name) and not regen_model:
    goodLdaModel = LdaModel.load(model_name)
else:
    goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=5, num_topics=2)
    goodLdaModel.save(model_name)

print("Word disribution in topics: ")
print("topic_0: " + str(goodLdaModel.print_topic(0, 12)))
print("topic_1: " + str(goodLdaModel.print_topic(1, 12)))


for text in texts:
    bow = dictionary.doc2bow(text)
    cm = coherencemodel.CoherenceModel(goodLdaModel, texts=texts, corpus=bow, dictionary=dictionary)
    print("")
    print("Coherence = " + str(cm.get_coherence()))
    print("Bag of word = " + str(text))
    print("Share of topics = " + str(goodLdaModel.get_document_topics(bow)))

