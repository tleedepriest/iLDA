"""
will define a wrapper around the LDA Model of gensim
"""
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

# Create a corpus from a list of texts
print(common_texts)
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]


class iLDA():
    """
    This class will use LdaModel in a recursive fashion. Using the first
    instance of the model to train models based on each cluster in the first.
    """
    def __init__(self, first_trained_model, **kwargs):

        self.__dict__.update(kwargs)
        self.first_trained_model = first_trained_model
        self.addtional_splits = len(kwargs.items())

def test():
    lda = iLDA(common_corpus, num_topics=i)
    topics = lda.get_topics()
    
    #distribution of words over topics
    #Gives you the word weights for each topic.
    all_word_weights = []
    for num, topic in enumerate(topics):
        print(f"{num} topic")
        word_weight_topic_vector = []
        for word_ind, weight in enumerate(topic):
            word = common_dictionary[word_ind]
            word_weight_topic_vector.append((word, weight))
            word_weight_topic_vector.sort(key=lambda x:x[1], reverse=True)
        all_word_weights.append(word_weight_topic_vector)
        #distribution of topics over documents
        #gives you the topic weights for the first document.
    print([(all_word_weights[x], y) for x, y in lda[common_corpus[0]]])

def main():
    lda = LdaModel(common_corpus, num_topics=5)
    kwargs = {
            "level_one_max_splits": 5,
            "level_two_max_splits": 5}
    ilda = iLDA(lda, **kwargs)

if __name__ == "__main__":
    main()
