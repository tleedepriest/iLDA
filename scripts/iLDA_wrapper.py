"""
will define a wrapper around the LDA Model of gensim for a template of a text
processing pipeline.
"""
import re
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_texts, datapath

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import split_on_space, preprocess_string, strip_tags, strip_punctuation, strip_multiple_whitespaces, strip_short, remove_stopwords
from gensim.parsing.preprocessing import preprocess_string
from gensim.models import Phrases
#from gensim.parsing.preprocessing import preprocess_documents
from kneed import KneeLocator
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
            sub_dict = {"models":[],
                        "num_topics_coherence":[],
                        "optimal_model": None}

            self.model_eval_info = {"level_one": sub_dict,
                                    "level_two": sub_dict,
                                    "level_three": sub_dict}

    def get_num_topics(self, level):
        num_tops_cos = self.model_eval_info[level]["num_topics_coherence"]
        return [num_top_co[0] for num_top_co in num_tops_cos]

    def get_coherences(self, level):
        num_tops_cos = self.model_eval_info[level]["num_topics_coherence"]
        return [num_top_co[1] for num_top_co in num_tops_cos]

    def find_optimal_model(self, level):
        """
        Returns The optimal model based on the elbow of the coherence
        vs. the num_topics curve. Sets optimal_model key in mode_eval_info
        to the optimal model found.
        """
        models = self.model_eval_info[level]["models"]
        (optimal_topics,
         optimal_model_index) = self.find_optimal_topics(level)
        optimal_model = models[optimal_model_index]
        self.model_eval_info[level]["optimal_model"] = optimal_model
        return optimal_model

    def find_optimal_topics(self, level):
        """
        Returns the value of the optimal topics and the index
        of the value in the list of num_topics. We return this so that
        we can get the optimal model from the list of models using the
        same index as the index of the number of optimal topics.
        """
        num_topics = self.get_num_topics(level)
        coherence = self.get_coherences(level)
        if num_topics!=[]:
            kneedle = KneeLocator(num_topics, coherence, curve='concave')
            optimal_topics = int(kneedle.knee)
            return kneedle.knee,  num_topics.index(optimal_topics)

        # TO-DO: change this to better type of error?
        raise ValueError("There are no topics or coherence values\n"
                         "! You have to run train_models and\n"
                         "train_coherence first!")

    def vis_topics_coherences(self, level):
        """
        Visualizes coherence vs number of topics.
        """
        num_topics = self.get_num_topics(level)
        coherence = self.get_coherences(level)
        if num_topics!=[]:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(num_topics, coherence)
            optimal_topics, _ = self.find_optimal_topics(level)
            coherence = [round(co, 3) for co in coherence]
            for xy in zip(num_topics, coherence):
                ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
            ax.grid()
            plt.axvline(optimal_topics)
            plt.savefig("test.png")
            return None
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
            range(2, self.seed_model_max_topics+1, self.seed_model_step),
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
        trains models. sets models key value to list of models.
        """
        if level == "level_one":
            topic_nums = [x for x in
                          range(2,
                                self.seed_model_max_topics+1,
                                self.seed_model_step)]

        for topic_num in topic_nums:

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

def remove_emails(text):
    """
    This removes values in enclosed < >
    """
    return re.sub(r"\S*@\S*\s?", "", text)

def remove_in_article(text):
    return re.sub(r"In article.+writes:", "", text)

def remove_names(text):
    return re.sub(r"[A-Z][a-z]{1, 15}\s[A-Z][a-z]{1, 15}", "", text)

def remove_char(text, char):
    return re.sub(r"{char}", "", text)

def get_doc_paths(docs_dir_path):
    """
    return all files in a given directory
    """
    return [x for x in Path(
        docs_dir_path).glob("**/*") if x.is_file()]

def format_topics_sentences(ldamodel, corpus, texts, doc_paths):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    doc_paths = pd.Series(doc_paths)
    sent_topics_df = pd.concat(
        [doc_paths, sent_topics_df, contents], axis=1)
    return sent_topics_df


def main():
    # first retrieve text, tokens, dictionary, and corpus
    texts_tokens = []
    raw_texts = []
    custom_filters = [lambda x: x.lower(), strip_tags, strip_punctuation, strip_multiple_whitespaces, remove_stopwords, strip_short]
    docs_dir_path = "20news_home"
    doc_paths = get_doc_paths(docs_dir_path)
    for doc_path in doc_paths:
        raw_text = get_raw_text(doc_path)
        raw_texts.append(raw_text)
        raw_text = remove_emails(raw_text)
        raw_text = remove_in_article(raw_text)
        raw_text = remove_names(raw_text)
        raw_text = remove_char(raw_text, ">")
        tokens = preprocess_string(raw_text, filters=custom_filters)
        texts_tokens.append(tokens)

    bigrams = Phrases(texts_tokens)
    texts_tokens = [bigrams[tokens] for tokens in texts_tokens]
    dct = Dictionary(texts_tokens)
    corpus = [dct.doc2bow(text) for text in texts_tokens]

    kwargs = {
            "seed_model_max_topics":40,
            "seed_model_step":2,
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
    optimal_model = ilda.find_optimal_model(level="level_one")
    temp_file = datapath("optimal_model")
    optimal_model.save(temp_file)
    num_topics = optimal_model.get_topics().shape[0]
    for top in range(0, num_topics):
        topic_terms = optimal_model.get_topic_terms(top)
        topic_inx = [topic_term[0] for topic_term in topic_terms]
        words = [dct[id_] for id_ in topic_inx]
        print(words)

    df = format_topics_sentences(optimal_model, corpus, raw_texts, doc_paths)
    df.to_csv("test.csv")
    # next step in the pipeline is to slice the df into rows by dominant
    # topic value and to train submodels for each of the
    # slices of rows.
    # for each value in dominant topic, train a submodel that splits the
    # group into at most five topics.

    #print(ilda.get_models_coherence())
    #print(ilda.get_first_set_models())

if __name__ == "__main__":
    main()
