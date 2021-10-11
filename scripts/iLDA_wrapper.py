"""
will define a wrapper around the LDA Model of gensim for a template of a text
processing pipeline.
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import split_on_space
from gensim.parsing.preprocessing import preprocess_documents

class iLDA(LdaModel):
    """
    inherits all attributes from LdaModel.
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None,
            distributed=False, chunksize=2000, passes=1, update_every=1,
            alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10,
            iterations=50, gamma_threshold=0.001, minimum_probability=0.01,
            random_state=None, ns_conf=None, minimum_phi_value=0.01,
            per_word_topics=False, callbacks=None, dtype=np.float32,
            hierarchy_levels=3, tokens=None, 
            model_eval_info=None, original_lda_model=None, **kwargs):
        
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
        self.tokens = tokens
        
        if model_eval_info is None:
            self.model_eval_info = {
                    "level_one": 
                    {
                        "models":[],
                        "num_topics_coherence":[],
                        "optimal_model": None
                }
                    }

    def pick_optimal_model(self):
        """
        Algorithm to hueristically pick an optimal number of topics. Want to
        Bias towards smaller numbers
        """
        changes = []

    def vis_topics_coherences(self, level):
        """
        Visualizes coherence vs number of topics.
        """
        num_tops_cos = self.model_eval_info[level]["num_topics_coherence"]
        if num_tops_cos !=[]:
            num_topics = [num_top_co[0] for num_top_co in num_tops_cos]
            coherence = [num_top_co[1] for num_top_co in num_tops_cos]
        
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(num_topics, coherence)
            coherence = [round(co, 3) for co in coherence]
            for xy in zip(num_topics, coherence):
                ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
            ax.grid()
            plt.savefig("test.png")
        else:
            # TO-DO: change this to better type of error?
            raise ValueError("There are no topics or coherence values to\n"
            "Visualize! You have to run train_models and\n"
            "train_coherence first!")

    def train_coherence(self, level):
        """
        returns a list of tuples with the number of topics and the coherence
        This allows for easy evaluation of models.
        """
        models = self.model_eval_info[level]["models"]
        for num_topic, model in zip(
            range(2, self.seed_model_max_topics+1), 
            models):
            cm = CoherenceModel(
                    model=model,
                    coherence='c_v',
                    texts=self.tokens)
            coherence = cm.get_coherence()
            self.model_eval_info[level]["num_topics_coherence"].append(
                    (num_topic, coherence))

    def train_models(self, level):
        """
        Instantiates the same model through a range of the number of topics.
        Returns a list of LDAmodels
        """
        for topic_num in range(2, 
                self.seed_model_max_topics+1,self.seed_model_step):

            model = LdaModel(self.corpus, topic_num, self.id2word,
                    self.distributed, self.chunksize, self.passes, 
                    self.update_every, self.alpha, self.eta, self.decay, 
                    self.offset, self.eval_every, self.iterations, 
                    self.gamma_threshold, self.minimum_probability,
                    self.random_state, self.ns_conf, self.minimum_phi_value,
                    self.per_word_topics, self.callbacks, self.dtype)
            self.model_eval_info[level]["models"].append(model)

# Preprocessing functions
def get_tokens(doc_raw_text):
    """
    yields text per document
    """
    doc_tokens = split_on_space(doc_raw_text)
    return doc_tokens

def get_raw_text(doc_path):
    """
    yield text for documents
    """
    with doc_path.open() as fh:
        text = fh.read()
        return text

def get_doc_paths(docs_dir_path):
    """
    return all files in a given directory
    """
    return [x for x in Path(
        docs_dir_path).glob("**/*") if x.is_file()]

def main():
    # first retrieve text, tokens, dictionary, and corpus
    texts = []
    docs_dir_path = "20news_home/20news-bydate-test/alt.atheism"
    doc_paths = get_doc_paths(docs_dir_path)
    for doc_path in doc_paths:
        raw_text = get_raw_text(doc_path)
        print(raw_text)
        texts.append(raw_text)
        #tokens = get_tokens(raw_text)
        #texts_tokens.append(tokens)
    texts_tokens = preprocess_documents(texts)
    print(texts_tokens)
    dct = Dictionary(texts_tokens)
    corpus = [dct.doc2bow(text) for text in texts_tokens]

    kwargs = {
            "seed_model_max_topics":20,
            "seed_model_step":1,
            "level_one_max_splits": 5,
            "level_two_max_splits": 5
            } 
    
    ilda = iLDA(corpus, 
            num_topics=5, 
            id2word=dct,
            tokens=texts_tokens,
            **kwargs)
    ilda.train_models(level="level_one")
    ilda.train_coherence(level="level_one")
    ilda.vis_topics_coherences(level="level_one")

    #print(ilda.get_models_coherence())
    #print(ilda.get_first_set_models())

if __name__ == "__main__":
    main()
