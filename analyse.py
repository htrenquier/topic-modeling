from gensim import models

lda_model = models.LdaModel.load("../saved_topics/ldamodel10topics", mmap='r')

print(lda_model.print_topics(num_topics=10, num_words=10))
print(lda_model.get_topic_terms(topicid=0, topn=10))
print(lda_model.top_topics(corpus=None, texts=None, dictionary=None, window_size=None, coherence='u_mass', topn=20, processes=-1))

# word2vec
modelName = '../GoogleNews-vectors-negative300.bin'
w2v_model = models.KeyedVectors.load_word2vec_format(modelName, binary=True)



