import os
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim import models
from numpy import genfromtxt


# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
# directory to scan
ScanDir = '/Users/htrenqui/Documents/Travail/UvA/rp2/res10k'

texts = []


for root, directories, filenames in os.walk(ScanDir):
    for filename in filenames:
        file_path = os.path.join(root, filename)
        # word list from csv
        filtered_text = genfromtxt(file_path, delimiter=',', dtype=str).tolist()
        stemmed_text = [p_stemmer.stem(i) for i in filtered_text]
        stemmed_text = filtered_text
        texts.append(stemmed_text)

print("len texts = "+ str(len(texts)))
print("dictionary")
dictionary = corpora.Dictionary(texts)
print("corpus")
corpus = [dictionary.doc2bow(text) for text in texts]
print("")
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=10, num_words=10))





# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

