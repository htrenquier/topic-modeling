import os
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim import models
import numpy as np
import time

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
# numpy warning
print(np.geterr())
np.seterr(all='ignore')
#np.warnings.filterwarnings('ignore')

print(np.geterr())
# directory to scan
ScanDir = '/Users/htrenqui/Documents/Travail/UvA/rp2/proj/res10k'

texts = []
num_empty_files = 0

print("upload & stemming")
time_start = time.time()
print(int(time.time() - time_start))
for root, directories, filenames in os.walk(ScanDir):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        # word list from csv
        if os.stat(file_path).st_size <= 2:
            # empty file
            num_empty_files += 1
        else:
            filtered_text = np.genfromtxt(file_path, delimiter=',', dtype=str).tolist()
            stemmed_text = [p_stemmer.stem(i) for i in filtered_text]
            texts.append(stemmed_text)

print(str(num_empty_files) + " empty files")
print("len texts = " + str(len(texts)))
print(int(time.time() - time_start))
print("dictionary")
dictionary = corpora.Dictionary(texts)
print(int(time.time() - time_start))
print("corpus")
corpus = [dictionary.doc2bow(text) for text in texts]
print(int(time.time() - time_start))
print("model")
print("50")
ldamodel50 = models.ldamodel.LdaModel(corpus, num_topics=50, id2word=dictionary)
print("200")
ldamodel200 = models.ldamodel.LdaModel(corpus, num_topics=200, id2word=dictionary)
print("500")
ldamodel500 = models.ldamodel.LdaModel(corpus, num_topics=500, id2word=dictionary)
print(int(time.time() - time_start))
print("top_topics:")
print(ldamodel.top_topics(corpus=corpus, texts=texts, dictionary=dictionary, window_size=None, coherence='u_mass',
                          topn=5, processes=-1))


print(int(time.time() - time_start))
print("loading ggl model")
#modelName = '../GoogleNews-vectors-negative300.bin'
#w2v_model = models.KeyedVectors.load_word2vec_format(modelName, binary=True)
print("coherence based on ggl model:")
print(int(time.time() - time_start))
print("50")
cm50 = models.CoherenceModel(model=ldamodel50, corpus=corpus, dictionary=dictionary, coherence='u_mass')  # tm is the trained topic model
print("200")
cm200 = models.CoherenceModel(model=ldamodel200, corpus=corpus, dictionary=dictionary, coherence='u_mass')  # tm is the trained topic model
print("500")
cm500 = models.CoherenceModel(model=ldamodel500, corpus=corpus, dictionary=dictionary, coherence='u_mass')  # tm is the trained topic model
print("cm.get_coherence()")
print(int(time.time() - time_start))
print("50:")
print(cm50.get_coherence())
print("200:")
print(cm200.get_coherence())
print("500:")
print(cm500.get_coherence())
print(int(time.time() - time_start))

#ldamodel.save("../saved_topics/ldamodel10topics")



# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

