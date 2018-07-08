from gensim import models
import time
import itertools

#tw = ['th', 'de', 'er', 'ed', 'ng', 'enron', 'nd', 'es', 'al', 'ing']
#tw = ['spy','spi','happi','happy']
tw = ['bacteria', 'poker']


def load_w2vec_model(mn):
    """
    Loads word2vec model and prints time
    :param modelName: Name (path) of the model
    :return: word2vec model
    """
    print("loading w2vec")
    m = models.KeyedVectors.load_word2vec_format(mn, binary=True)
    print("w2vec loaded")
    print("~")
    return m


def intra_topic_sim(tw, vec_model):
    """
    computes intra topic coherence given a list of top words (tw)
    :param tw: top_words
    :return:
    """
    sims = []
    for v, w in itertools.combinations(tw, 2):
        try:
            sim = vec_model.similarity(v, w)+1
            print(v, w, sim)
            sims.append(sim)
        except KeyError as ke:
            #ukn_words += 1
            pass
        # print(v + " / " + w + " => " + str(sims[-1]))
    if len(sims) == 0:
        print("Bad topic: Unknown words")
        return 0
    else:
        return sum(sims)/len(sims)



w2v_model_name = '../GoogleNews-vectors-negative300.bin'
w2v_model = load_w2vec_model(w2v_model_name)

print(intra_topic_sim(tw, w2v_model))