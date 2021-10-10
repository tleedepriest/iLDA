"""
will define a wrapper around the LDA Model of gensim
"""
import numpy as np
from pathlib import Path
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import split_on_space

# Create a corpus from a list of texts
print(common_texts)
common_dictionary = Dictionary(common_texts)
common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
print(common_corpus)
class iLDA(LdaModel):

    def __init__(self, corpus=None, num_topics=100, id2word=None,
            distributed=False, chunksize=2000, passes=1, update_every=1,
            alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
            iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
            random_state=None, ns_conf=None, minimum_phi_value=0.01,
            per_word_topics=False, callbacks=None, dtype=np.float32,
            hierarchy_levels=3, docs_dir_path=None, **kwargs):
        
        super().__init__(corpus, num_topics, id2word,
            distributed, chunksize, passes, update_every,
            alpha, eta, decay, offset, eval_every,
            iterations, gamma_threshold, minimum_probability,
            random_state, ns_conf, minimum_phi_value,
            per_word_topics, callbacks, dtype)

        # number of levels to the hierarchy.
        
        self.__dict__.update(kwargs) 
        self.corpus = corpus
        self.num_topics = num_topics
        self.id2word = id2word
        self.distibuted = distributed
        self.chunksize = chunksize
        self.passes = passes
        self.update_every = update_every
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.offset = offset
        self.eval_every = eval_every
        self.iterations = iterations
        self.gamma_threshold = gamma_threshold
        self.ns_conf = ns_conf
        self.minimum_phi_value = minimum_phi_value
        self.per_word_topics = per_word_topics
        self.callbacks = callbacks
        self.dtype = dtype
        self.hierarchy_levels = len(kwargs.items())
        self.docs_dir_path = docs_dir_path

    def get_tokens(self):
        """
        tokenize text
        """
        tokens = split_on_space(self.corpus)
        print(tokens)

    def get_first_set_models(self):
        """
        Instantiates the same model through a range of the number of topics.
        Returns a list of LDAmodels
        """
        models = []
        for topic_num in range(2, 
                self.seed_model_max_topics,self.seed_model_step):

            model = LdaModel(self.corpus, topic_num, self.id2word,
                    self.distributed, self.chunksize, self.passes, 
                    self.update_every, self.alpha, self.eta, self.decay, 
                    self.offset, self.eval_every, self.iterations, 
                    self.gamma_threshold, self.minimum_probability,
                    self.random_state, self.ns_conf, self.minimum_phi_value,
                    self.per_word_topics, self.callbacks, self.dtype)
            print(model)
            models.append(model)
        return models

    def evaluate_models_on_coherence(self):
        models = self.get_first_set_models()
        for model in models:
            cm = CoherenceModel(
                    model=model,
                    coherence='c_v',
                    texts=common_texts)
            coherence = cm.get_coherence()
            print(coherence)

    def get_doc_paths(self):
        """
        return all files in a given directory
        """
        return [x for x in Path(
            self.docs_dir_path).glob("**/*") if x.is_file()]

    def yield_raw_text(self):
        """
        yield text for documents
        """
        for path in self.get_doc_paths():
            with path.open() as fh:
                text = fh.read()
            yield text


def main():
    kwargs = {
            "seed_model_max_topics":20,
            "seed_model_step":1,
            "level_one_max_splits": 5,
            "level_two_max_splits": 5
            }
    docs_dir_path = "20news_home/20news-bydate-test/alt.atheism"

    ilda = iLDA(common_corpus, 
            num_topics=5, 
            id2word=common_dictionary,
            docs_dir_path = docs_dir_path,
            **kwargs)
    ilda.evaluate_models_on_coherence()
    paths = ilda.get_doc_paths()
    text_iter = ilda.yield_raw_text()
    for text in text_iter:
        print(text)
    #ilda.get_tokens()

if __name__ == "__main__":
    main()
