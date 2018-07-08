# Improving Semantic Quality of Topic Models

In this research, we study the quality of a topic modeling technique called
Latent Dirichlet Allocation. We wonder if it could be reliably used in forensic
investigations to cope with large evidence datasets. We make several
measurements to understand the impact of two important parameters on
the quality of LDA models. For that, we consider two existing evaluation
metrics to estimate the quality of a model and design our own. The dataset
used in the experiments is the Enron e-mail database which is often used in
Natural Language Processing.

### How to improve semantic quality of LDA models ?
* What is the optimal number of topics for a LDA model?
* How does the number of iterations influence the quality of models?
* Can we improve the semantic quality evaluation?

This repository contains the source code used during a research project. The paper written for this project is available in this repository as well.

[Word2vec Code Archive - Google Code Project Hosting.](https://code.google.com/archive/p/word2vec/)  
[Enron Email Dataset](https://www.cs.cmu.edu/~enron/)  

___
Python 2.7  
Required libraries are NLTK and Gensim.
Make sure the subfolders mentionned in the code are already created.

- extract.py  
No arguments. Preprocessing code.  
Walks the raw e-mails folder and extracts for each file the most meaningful content. Output files are stored in (automatically created) 'batch' folders of maximum 5000 preprocessed e-mails.

- model.py  
From the preprocessed data:
  - builds a file a all bags of words: 'texts'
  - builds models from 'texts' and save them.
  - computes the coherence for 'c_v', 'u_mass' and the 'c_word2vec' introduced in the paper.
  - outputs results files.

If the extract.py script has been executed and the result stored in the folder  ``../res``, one can run the following commands to generate the 'texts' file or generate the models with an already existing file.
```
python model.py --gen_texts yes --scanpath ../res --save_texts yes
python model.py --load_texts ../res_texts/texts.csv
```

- analyse.py  
Allows to analyse a generated model. The code should be reshaped for the user's intention.

The code is not optimized for the LDA models generation, nor the models space storage.
