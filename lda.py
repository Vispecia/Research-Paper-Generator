#doc_test = "Boltzmann Machines (BMs) are a particular form of log-linear Markov Random Field (MRF), i.e., for which the energy function is linear in its free parameters. To make them powerful enough to represent complicated distributions (i.e., go from the limited parametric setting to a non-parametric one), we consider that some of the variables are never observed (they are called hidden). By having more hidden variables (also called hidden units), we can increase the modeling capacity of the Boltzmann Machine (BM). Restricted Boltzmann Machines further restrict BMs to those without visible-visible and hidden-hidden connections. A graphical depiction of an RBM is shown below."

import sys
from time import time
sys.path.append("E:/PythonProjects/DeepLearning/")
from textGenerator import sendText


from gensim import corpora, models
import gensim
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *


doc_a = "A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. This allows it to exhibit temporal dynamic behavior. Derived from feedforward neural networks, RNNs can use their internal state (memory) to process variable length sequences of inputs.[1] This makes them applicable to tasks such as unsegmented, connected handwriting recognition[2] or speech recognition.The term recurrent neural network is used indiscriminately to refer to two broad classes of networks with a similar general structure, where one is finite impulse and the other is infinite impulse. Both classes of networks exhibit temporal dynamic behavior.[5] A finite impulse recurrent network is a directed acyclic graph that can be unrolled and replaced with a strictly feedforward neural network, while an infinite impulse recurrent network is a directed cyclic graph that can not be unrolled."
doc_b = "CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The fully-connectedness of these networks makes them prone to overfitting data. Typical ways of regularization include adding some form of magnitude measurement of weights to the loss function. CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble more complex patterns using smaller and simpler patterns. Therefore, on the scale of connectedness and complexity, CNNs are on the lower extreme."
doc_c = "An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore signal noise. Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input, hence its name. Several variants exist to the basic model, with the aim of forcing the learned representations of the input to assume useful properties. Examples are the regularized autoencoders (Sparse, Denoising and Contractive autoencoders), proven effective in learning representations for subsequent classification tasks, and Variational autoencoders, with their recent applications as generative models. Autoencoders are effectively used for solving many applied problems, from face recognition to acquiring the semantic meaning of words."

doc_test = sendText()
doc_set = [doc_a,doc_b,doc_c]



def lemmatize_stemming(text):
    stemmer = SnowballStemmer(language='english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result



texts = []

for d in doc_set:
    # raw = d.lower()
    # tk = tokenizer.tokenize(raw)   
    # stopped_tokens = [i for i in tk if not i in en_stop]
    # stemmed = [p_stemmer.stem(i) for i in stopped_tokens]
    texts.append(preprocess(d))
    
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word = dictionary, passes=20)



test = dictionary.doc2bow(preprocess(doc_test)) 

for index, score in sorted(ldamodel[test], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t Topic: {}".format(score, ldamodel.print_topic(index, 5)))



























# from nltk.tokenize import RegexpTokenizer
# tokenizer = RegexpTokenizer(r'\w+')

# from stop_words import get_stop_words
# en_stop = get_stop_words('en')

# from nltk.stem.porter import PorterStemmer
# p_stemmer = PorterStemmer()
