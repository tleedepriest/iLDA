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

class iLDA(LdaModel):

    def __init__(self, corpus=None, num_topics=100, id2word=None,
            distributed=False, chunksize=2000, passes=1, update_every=1,
            alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
            iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
            random_state=None, ns_conf=None, minimum_phi_value=0.01,
            per_word_topics=False, callbacks=None, dtype=np.float32, 
            hierarchy_levels=3, level_two_max_topics=5, 
            level_three_max_topics=5):
        
        super().__init__(corpus, num_topics, id2word,
            distributed, chunksize, passes, update_every,
            alpha, eta, decay, offset, eval_every,
            iterations, gamma_threshold, minimum_probability,
            random_state, ns_conf, minimum_phi_value,
            per_word_topics, callbacks, dtype)

        # number of levels to the hierarchy.
        self.hierarchy_levels = hierarchy_levels

        # max level of topics you want each cluster in
        # first hierarhcy to be split into. Minimum is 2.
        self.level_two_range = level_two_max_topics
        self.level_three_range = level_three_max_topics
    
lda = iLDA(common_corpus, num_topics=10)
topics = lda.get_topics()

# distribution of words over topics
# Gives you the word weights for each topic.
all_word_weights = []
for num, topic in enumerate(topics):
    print(f"{num} topic")
    word_weight_topic_vector = []
    for word_ind, weight in enumerate(topic):
        word = common_dictionary[word_ind]
        word_weight_topic_vector.append((word, weight))
        word_weight_topic_vector.sort(key=lambda x:x[1], reverse=True)
    all_word_weights.append(word_weight_topic_vector)

# distribution of topics over documents
# gives you the topic weights for the first document.
print([(all_word_weights[x], y) for x, y in lda[common_corpus[0]]])
