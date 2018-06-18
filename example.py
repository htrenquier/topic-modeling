from gensim.models.ldamodel import LdaModel
from gensim.models import coherencemodel
from gensim.corpora.dictionary import Dictionary

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
goodLdaModel = LdaModel(corpus=corpus, id2word=dictionary, iterations=5, num_topics=2)

print(goodLdaModel.print_topic(0, 12))
print(goodLdaModel.print_topic(1, 12))

for text in texts:
    bow = dictionary.doc2bow(text)
    cm = coherencemodel.CoherenceModel(goodLdaModel, texts=texts, corpus=bow, dictionary=dictionary)
    print(cm.get_coherence())
    print(text)
    print(goodLdaModel.get_document_topics(bow))

