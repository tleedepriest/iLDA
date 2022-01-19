"""
will define a wrapper around the LDA Model of gensim for a template of a text
processing pipeline.
"""
import re
import numpy as np
from numpy.linalg import norm
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

from utils import remove_emails, remove_names, \
    remove_arrow, remove_in_article
# from gensim.parsing.preprocessing import preprocess_documents
# from kneed import KneeLocator


class iLDA:
    """
    inherits all attributes from LdaModel.
    """

    def __init__(self, docs_dir=None,
                 string_filters=None,
                 hierarchy_levels=3,
                 model_eval_info=None,
                 original_lda_model=None,
                 **kwargs):


        # number of levels to the hierarchy.

        self.__dict__.update(kwargs)
        self.hierarchy_levels = len(kwargs.items())
        self.docs_dir = docs_dir
        self.doc_paths = [x for x in Path(
            self.docs_dir).glob("**/*") if x.is_file()]
        self.string_filters = string_filters
        self.tokens = None
        self.corpus = None
        self.id2word = None
        if model_eval_info is None:
            sub_dict = {"models":[],
                        "num_topics_coherence":[],
                        "optimal_model": None}

            self.model_eval_info = {"level_one": sub_dict,
                                    "level_two": sub_dict,
                                    "level_three": sub_dict}

    def get_num_topics(self, level):
        """
        Parameters
        --------------
        level: str
            'level_one', 'level_two', 'level_three'

        Returns
        -------------
            :List[float]
            A list of topic numbers for each model, ASC order
        """
        num_tops_cos = self.model_eval_info[level]["num_topics_coherence"]
        return [num_top_co[0] for num_top_co in num_tops_cos]

    def get_coherences(self, level):
        """
        Parameters
        --------------
        level: str
            'level_one', 'level_two', 'level_three'

        Returns
        -------------
            :List[float]
            A list of coherence values in the same order as ascending
            topic_number order.
        """
        num_tops_cos = self.model_eval_info[level]["num_topics_coherence"]
        return [num_top_co[1] for num_top_co in num_tops_cos]

    # TODO: change name to set_optimal_model. Have it only set a value
    # and return nothing.
    def find_optimal_model(self, level):
        """
        Sets "optimal_model value in model_eval_info dictionary"
        """
        models = self.model_eval_info[level]["models"]
        (optimal_topics,
         optimal_model_index) = self.find_optimal_topics(level)
        optimal_model = models[optimal_model_index]
        optimal_model = LdaModel.load(optimal_model)
        self.model_eval_info[level]["optimal_model"] = optimal_model
        return optimal_model

#   def find_optimal_topics(self, level):
#       """
#       Returns the value of the optimal topics and the index
#       of the value in the list of num_topics. We return this so that
#       we can get the optimal model from the list of models using the
#       same index as the index of the number of optimal topics.
#       """
#       num_topics = self.get_num_topics(level)
#       coherence = self.get_coherences(level)
#       if num_topics!=[]:
#           kneedle = KneeLocator(num_topics, coherence, curve='concave')
#           optimal_topics = int(kneedle.knee)
#           return kneedle.knee,  num_topics.index(optimal_topics)

       # TO-DO: change this to better type of error?
#        raise ValueError("There are no topics or coherence values\n"
#                         "! You have to run train_models and\n"
#                         "train_coherence first!")

    def find_optimal_topics(self, level):
        """
        Parameters
        -------------
        level: str
            'level_one', 'level_two', 'level_three'
        Returns
        -------------
        optimal topics: int
            mumber of topics in optimal_model. int > 2

        optimal_topics_index: int
        The index in the num_topics list, which is sorted in
        ascending order.Index can be used to get optimal model
        in models list
        """
        num_topics = self.get_num_topics(level)
        coherence = self.get_coherences(level)
        # perpendicular distance each data point is from
        # a line connecting first and last datapoint.
        distances = []
        if num_topics!=[]:
            first_coherence = coherence[0]
            first_topic = num_topics[0]
            last_coherence = coherence[-1]
            last_topic = num_topics[-1]
            # will use points to calc vector
            p1 = np.asarray((first_coherence, first_topic))
            p2 = np.asarray((last_coherence, last_topic))
            # calculate perpendicular distance from data
            # points in graph to vector between first
            # and last point.
            for co, top in zip(coherence, num_topics):
                p3 = np.asarray((co, top))
                # take cross product between line and
                # line to point in consideration, and normalize,
                # to get perpendicular distance datapoint from
                # graph.
                d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
                distances.append(d)

        max_distance = max(distances)
        max_distance_index = distances.index(max_distance)
        optimal_topics = num_topics[max_distance_index]
        optimal_topics_index =  num_topics.index(optimal_topics)
        return optimal_topics, optimal_topics_index
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
            model = LdaModel.load(model)
            cm = CoherenceModel(
                    model=model,
                    corpus=self.corpus,
                    texts=self.tokens,
                    coherence='c_v')
            coherence = cm.get_coherence()
            self.model_eval_info[level]["num_topics_coherence"].append(
                    (num_topic, coherence))
            del model

    def train_models(self, level):
        """
        trains models. sets models key value to list of models.
        """
        if level == "level_one":
            topic_nums = [x for x in
                          range(2,
                                self.seed_model_max_topics+1,
                                self.seed_model_step)]

        elif level == "level_two":
            topic_nums = [x for x in
                          range(2, self.level_one_max_topics+1)]

            optimal_seed_model = self.model_eval_info["level_one"]["optimal_model"]

        for num_topic in topic_nums:
            model = LdaModel(corpus=self.corpus,
                             num_topics=num_topic,
                             id2word=self.id2word)
            temp_filepath = datapath(f"model_{level}_{num_topic}")
            model.save(temp_filepath)
            self.model_eval_info[level]["models"].append(temp_filepath)
            del model

    def set_attributes(self, with_bigrams):
        """
        calls other private methods to set all needed attributes
        for the class. This should be called first after instantiation
        """
        self._set_tokens()

        if with_bigrams:
            self._convert_tokens_to_bigrams()

        self._set_id2word()
        self._set_corpus()

    def _set_tokens(self):
        """
        """
        all_tokens = []
        for doc_path in self.doc_paths:
            clean_text = self._get_raw_text(doc_path)
            tokens = self._clean_text_get_tokens(clean_text)
            all_tokens.append(tokens)
        self.tokens = all_tokens

    def _convert_tokens_to_bigrams(self):
        """
        should be called after set_tokens
        """
        bigrams = Phrases(self.tokens)
        bigram_tokens = [
            bigrams[doc_tokens] for doc_tokens in self.tokens]
        self.tokens = bigram_tokens

    def _set_corpus(self):
        """
        set corpus
        """
        self.corpus = [
            self.id2word.doc2bow(doc_token) for doc_token
            in self.tokens]

    def _set_id2word(self):
        """
        set id2word
        """
        dct = Dictionary(self.tokens)
        self.id2word = dct

    def _clean_text_get_tokens(self, raw_text):
        """
        Parameters
        -------------
        raw_text: str
        filters: List[function]

        Returns
        --------------
        tokens: List[str]
        """
        tokens = preprocess_string(raw_text, filters=self.string_filters)
        return tokens

    def _get_raw_text(self, doc_path):
        """
        yield text for documents
        """
        with doc_path.open() as fh:
            text = fh.read()
            return text

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
    string_filters = [remove_emails,
                      remove_in_article,
                      remove_names,
                      remove_arrow,
                      lambda x: x.lower(),
                      strip_tags,
                      strip_punctuation,
                      strip_multiple_whitespaces,
                      remove_stopwords,
                      strip_short]

    docs_dir_path = "20news_home"

    # will likely get rid of this, just wanted to explore **kwardg
    # with setting attributes.
    kwargs = {"seed_model_max_topics":10,
              "seed_model_step":2,
              "level_two_max_topics": 10
            }

    ilda = iLDA(docs_dir="20news_home/20news-bydate-test/sci.med",
                string_filters=string_filters,
                **kwargs)

    ilda.set_attributes(with_bigrams=True)
    ilda.train_models(level="level_one")
    ilda.train_coherence(level="level_one")
    ilda.vis_topics_coherences(level="level_one")
    optimal_model = ilda.find_optimal_model(level="level_one")

    #num_topics = optimal_model.get_topics().shape[0]
    #for top in range(0, num_topics):
    #    topic_terms = optimal_model.get_topic_terms(top)
    #    topic_inx = [topic_term[0] for topic_term in topic_terms]
    #    words = [dct[id_] for id_ in topic_inx]
    #    print(words)

    #df = format_topics_sentences(optimal_model, corpus, raw_texts, doc_paths)
    #df.to_csv("test.csv")
    # next step in the pipeline is to slice the df into rows by dominant
    # topic value and to train submodels for each of the
    # slices of rows.
    # for each value in dominant topic, train a submodel that splits the
    # group into at most five topics.

    #print(ilda.get_models_coherence())
    #print(ilda.get_first_set_models())

if __name__ == "__main__":
    main()
