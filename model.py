import os
from nltk.stem.porter import PorterStemmer
from gensim import corpora
from gensim import models
import numpy as np
import time
import csv
import argparse
import itertools


def load_w2vec_model(mn):
    """
    Loads word2vec model and prints time
    :param modelName: Name (path) of the model
    :return: word2vec model
    """
    print("loading w2vec")
    print("time: " + str(int(time.time() - time_start)))
    m = models.KeyedVectors.load_word2vec_format(mn, binary=True)
    print("w2vec loaded")
    print("time: " + str(int(time.time() - time_start)))
    print("~")
    return m


def intra_topic_coherence(top_words, vec_model):
    """
    Computes a custom coherence measure from top words of a topic
    :param top_words:
    :param vec_model:
    :return: custom coherence
    """
    sims = []
    for v, w in itertools.combinations(top_words, 2):
        sims.append(vec_model.similarity(v, w))
        # print(v + " / " + w + " => " + str(sims[-1]))
    return sum(sims)/len(sims)


def get_mycoh(m):
    """
    Computes the custom coherence for a LDA model
    :param m: LDA model
    :return: Average custom coherence of all topics
    """
    topns = []
    num_topics = len(m.get_topics())
    topic_sim = []
    ukn_words = 0
    # for each topic of the model get the top n words and compute my_topic_coherence()
    for k in range(0, num_topics):
        # print("topn for model " + str(k) + " topics:, topic no " + str(k))
        topns.append(get_topn_pertopic(m, k, 10))
        try:
            topic_sim.append(intra_topic_coherence(topns[-1], w2v_model))
        except KeyError as ke:
            ukn_words += 1
            # print("Unknown word in model k=" + str(rg[lda_models.index(model)]))
    if len(topic_sim) != 0:
        avg_sim = sum(topic_sim) / len(topic_sim)
        print("my_coherence: (k = " + str(rg[lda_models.index(m)]) + ") = " + str(avg_sim))
    else:
        print(str(rg[lda_models.index(m)]) + " : model failed")
    print(str(ukn_words) + " unknown words in model k = " + str(rg[lda_models.index(m)]))
    return avg_sim


def my_model_coherence(top_words_list, vec_model):
    """

    :param top_words_list: list of top_words, top_words is a list of words representing 1 topic
    :param vec_model:
    :return:
    """
    inter_ts = []
    intra_ts = []
    for top_words in top_words_list:
        # inter
        inter_ts.append(inter_topics_sim(top_words_list, top_words, vec_model))
        # intra
        intra_ts.append(intra_topic_sim(top_words, vec_model))

    return sum(intra_ts)*len(inter_ts)/(len(intra_ts)*sum(inter_ts))


def intra_topic_sim(tw, vec_model):
    """
    computes intra topic coherence given a list of top words (tw)
    :param tw: top_words
    :return:
    """
    sims = []
    for v, w in itertools.combinations(tw, 2):
        try:
            sims.append(vec_model.similarity(v, w)+1)
        except KeyError as ke:
            #ukn_words += 1
            pass
        # print(v + " / " + w + " => " + str(sims[-1]))
    if len(sims) == 0:
        print("Bad topic: Unknown words")
        return 0
    else:
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
    for lv, lw in itertools.combinations(twl, 2):
        try:
            sims.append(vec_model.n_similarity(lv, lw) + 1)
        except KeyError:
            pass
    if len(sims) == 0:
        print("Bad topic: Unknown words")
        return 0
    else:
        return sum(sims) / len(sims)


def get_topn_pertopic(m, t, n):
    """

    :param m: model
    :param t: topic number
    :param n: number of words in the top words
    :return:
    """
    topn = []
    for word, freq in m.show_topic(t, n):
        topn.append(word)
    return topn


def build_texts(args, scandir):
    """
    Loads or generate texts
    :param args: input arguments
    :param scandir: directory to scan for source documents
    :return: texts
    """
    # text collection
    texts = []
    if args["gen_texts"] == 'yes':
        # directory to scan
        if args["scanpath"] != 'no':
            scandir = args["scanpath"]

        print("Scan path: " + str(scandir))
        num_empty_files = 0

        k = 0
        print("upload & stemming")
        time_start = time.time()
        print(int(time.time() - time_start))
        for root, directories, filenames in os.walk(scandir):
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
    else:
        if args["load_texts"] != 'no':
            texts_file_path = args["load_texts"]
            print("Starting uploading from texts file...")
            with open(texts_file_path, "r") as texts_file:
                for l in texts_file.readlines():
                    texts.append(l.split(","))
            print("texts uploaded with " + str(len(texts)) + " lines.")
            texts_file.close()
        else:
            exit("No texts file given in --load_texts option")
        if len(texts) == 0:
            exit(code="texts empty")
    return texts


def gen_dict_and_corpus(texts):
    print("Generating dictionary:")
    d = corpora.Dictionary(texts)
    print("time: " + str(int(time.time() - time_start)))
    print("~")
    print("Generating Corpus:")
    c = [d.doc2bow(text) for text in texts]
    print("time: " + str(int(time.time() - time_start)))
    print("~")
    return d, c


def build_lda_models(d, c, it, r):
    """
    Generate or load models according to the range, num_it, etc.
    :param d: dictionary
    :param c: corpus
    :param it: number of iterations
    :param r: range
    :return: list of models
    """
    lms = []  #lda models
    print("Generating models (k: # of topics):")  # k = number of topics
    for k in r:
        print("k = "+str(k))
        mn = "../res_models/lda_model_k"+str(k)+"_it"+str(it)  # mn for model name
        if os.path.isfile(mn):
            lms.append(models.LdaModel.load(mn, mmap='r'))
            print(mn + " loaded.")
        else:
            lms.append(models.ldamodel.LdaModel(c, num_topics=k, id2word=d, iterations=it))
            lms[-1].save(mn)
            print(mn + " generated.")
        print("time: " + str(int(time.time() - time_start)))
        print("~")
    return lms


def get_coherences(m):
    """
    Computes custom coherence and u_mass, c_v coherences from coherence model for a model
    :param m: LDA model
    :return: 3-tuple of coherences of m
    """
    print("~")
    print("time: " + str(int(time.time() - time_start)))
    print("k = " + str(rg[lda_models.index(m)]))
    mc = get_mycoh(m)
    # cm = models.CoherenceModel(model=m, corpus=corpus, texts=texts, coherence='u_mass')
    # gc_u_mass = cm.get_coherence()
    # cm = models.CoherenceModel(model=m, corpus=corpus, texts=texts, coherence='c_v')
    # gc_c_v = cm.get_coherence()
    gc_u_mass = 0
    gc_c_v = 0
    return mc, gc_u_mass, gc_c_v

### MAIN ###


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
arguments = vars(ap.parse_args())

# directory to scan
scan_dir = '/Users/htrenqui/Documents/Travail/UvA/rp2/proj/res10k'
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
# time ref
time_start = time.time()
# ref word2vec model
w2v_model_name = '../GoogleNews-vectors-negative300.bin'
# range of models
rg = range(2, 67, 2)  #rg = range(10, 21, 10)
# number of iteration for model gen
num_it = 10

texts = build_texts(arguments, scan_dir)
w2v_model = load_w2vec_model(w2v_model_name)
dictionary, corpus = gen_dict_and_corpus(texts)

lda_models = build_lda_models(dictionary, corpus, num_it, rg)

print("Coherence")
for model in lda_models:
    mc, um, cv = get_coherences(model)
    with open("../res_coherence_"+str(num_it)+"it.csv", "a") as res_coherence_file:
        res_coherence_file.write(str(rg[lda_models.index(model)]) +
                                 "," + str(mc) +
                                 "," + str(um) +
                                 "," + str(cv) + "\r\n")
    res_coherence_file.close()
