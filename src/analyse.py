from gensim import models
import argparse
from gensim.corpora.dictionary import Dictionary

# arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--model_file", type=str, default="model",
        help="Generate text list")
args = vars(ap.parse_args())


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

file = args["model_file"]
lda_model = models.LdaModel.load(file, mmap='r')

try:
    print(str(file).split("_k")[-1].split("_it")[0])
    num_topics = int(str(file).split("_k")[-1].split("_it")[0])
except:
    exit("Wrong file name: " + str(file))

for k in range(0, num_topics):
    print(get_topn_pertopic(lda_model, k, 10))

target_topic_no = 8
i = 0
most_relatable_email = ["bs"]
max_prob = 0



texts_file_path = "/Users/htrenqui/Desktop/not_stemmed_texts.csv"

texts = []
with open(texts_file_path, "r") as texts_file:
    for l in texts_file.readlines():
        texts.append(l.split(","))

dictionary = Dictionary(texts)

with open("/Users/htrenqui/Desktop/not_stemmed_texts.csv", "r") as texts:
    for line in texts.readlines():
        doc = line.split(",")
        bow = dictionary.doc2bow(doc)
        topic, topic_probability = lda_model.get_document_topics(bow)[0]
        if topic_probability > max_prob:
            most_relatable_email = doc
            max_prob = topic_probability
        i += 1
        if i % 5000 == 0:
            print("line : " + str(i))
            print("Most r'able email = " + str(most_relatable_email) + "(" + str(max_prob) + ')')

texts.close()