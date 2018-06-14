import os
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim import models
import numpy as np
import time
import csv
import argparse


# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-g", "--gen_texts", type=str, default="no",
        help="Generate text list")
ap.add_argument("-c", "--stem", type=str, default="no",
        help="Stems tokenized words. Only if generating text list: -g")
ap.add_argument("-Z", "--scanpath", type=str, default="no",
        help="Set the scanning path. Only if generating text list: -g")
ap.add_argument("-s", "--save_texts", type=str, default="no",
        help="Saves the generated texts to a file. Only if generating text list: -g")
ap.add_argument("-l", "--load_texts", type=str, default="no",
        help="Loads texts from a file. arg is path of the texts file.")
args = vars(ap.parse_args())


# numpy warning
#np.seterr(all='ignore')
#np.warnings.filterwarnings('ignore')
#print(np.geterr())


# directory to scan
ScanDir = '/Users/htrenqui/Documents/Travail/UvA/rp2/proj/res10k'
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
# text collection
texts = []
# time ref
time_start = time.time()

if args["gen_texts"] == 'yes':
    # directory to scan
    if args["scanpath"] != 'no':
        ScanDir = args["scanpath"]

    print("Scan path: " + str(ScanDir))
    num_empty_files = 0

    k = 0
    print("upload & stemming")
    time_start = time.time()
    print(int(time.time() - time_start))
    for root, directories, filenames in os.walk(ScanDir):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            k += 1
            if k % 100 == 0:
                print(k)
            # word list from csv
            if os.stat(file_path).st_size <= 2:
                # empty file
                num_empty_files += 1
            else:
                filtered_text = np.genfromtxt(file_path, delimiter=',', dtype=str).tolist()
                if args["stem"] == 'yes':
                    filtered_text = [p_stemmer.stem(i) for i in filtered_text]
                texts.append(filtered_text)

    print(str(num_empty_files) + " empty files")
    if args["save_texts"] == 'yes':
        print("Saving texts in file")
        doc_name = "../res_texts/not_stemmed_texts.csv"
        texts_file = open(doc_name, "w")
        writer = csv.writer(texts_file)
        for text in texts:
            writer.writerow(text)
            k -= 1
            if k % 1000 == 0:
                print(k)

        texts_file.close()
        print("~~~ texts file saved !")

print("time: " + str(int(time.time() - time_start)))
print("~")
if args["load_texts"] != 'no':
    texts_file_path = args["load_texts"]
    print("Starting uploading from texts file...")
    with open(texts_file_path,"r") as texts_file:
        for l in texts_file.readlines():
            texts.append(l.split(","))
    print("texts uploaded! with " + str(len(texts)))
    print("proof:")
    for ih in range(0, 10):
        print(texts[ih])

if len(texts) == 0:
    print("Error texts empty")
    print("len texts = " + str(len(texts)))


print("time: " + str(int(time.time() - time_start)))
print("~")
print("Generating dictionary:")
dictionary = corpora.Dictionary(texts)
print("time: " + str(int(time.time() - time_start)))
print("~")
print("Generating Corpus:")
corpus = [dictionary.doc2bow(text) for text in texts]
print("time: " + str(int(time.time() - time_start)))
print("~")

lda_models = []
rg = range(10, 101, 10)
print("Generating models (k: # of topics):") #  k = number of topics
for k in rg:
    print("k = "+str(k))
    lda_models.append(models.ldamodel.LdaModel(corpus, num_topics=k, id2word=dictionary, iterations=10))
    lda_models[-1].save("../res_models/lda_model_k"+str(k))
    print("time: " + str(int(time.time() - time_start)))
    print("~")

print("Coherence")
for m in lda_models:
    print("U_MASS: (k = " + str(rg[lda_models.index(m)])+")")
    print(m.top_topics(corpus=corpus, texts=texts, dictionary=dictionary, window_size=None, coherence='u_mass',
                          topn=5, processes=4))
    print("time: " + str(int(time.time() - time_start)))
    print("~")
    print("C_V: (k = " + str(rg[lda_models.index(m)]) + ")")
    print(m.top_topics(corpus=corpus, texts=texts, dictionary=dictionary, window_size=None, coherence='c_v',
                       topn=5, processes=4))
    print("time: " + str(int(time.time() - time_start)))
    print("~")

#modelName = '../GoogleNews-vectors-negative300.bin'
#w2v_model = models.KeyedVectors.load_word2vec_format(modelName, binary=True)

# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
# https://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html

