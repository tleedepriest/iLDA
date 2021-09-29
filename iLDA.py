"""
will define a wrapper around the LDA Model of gensim
"""
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

# Create a corpus from a list of texts

common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]

class iLDA(LdaModel):

    def __init__(self, corpus=None, num_topics=100, id2word=None,
            distributed=False, chunksize=2000, passes=1, update_every=1,
            alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
            iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
            random_state=None, ns_conf=None, minimum_phi_value=0.01,
            per_word_topics=False, callbacks=None, dtype=np.float32, 
            hierarchy_levels=3):
        
        super().__init__(corpus, num_topics, id2word,
            distributed, chunksize, passes, update_every,
            alpha, eta, decay, offset, eval_every,
            iterations, gamma_threshold, minimum_probability,
            random_state, ns_conf, minimum_phi_value,
            per_word_topics, callbacks, dtype)
        
        self.hierarchy_levels = hierarchy_levels
    
lda = iLDA(common_corpus, num_topics=10)
print(lda.hierarchy_levels)

