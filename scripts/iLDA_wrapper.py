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
    High Level Wrapper around the LDA model from gensim.
    """

    def __init__(self, docs_dir=None,
                 doc_paths=None,
                 string_filters=None,
                 topic_range=None,
                 hierarchy_levels=3,
                 model_eval_info=None,
                 original_lda_model=None,
                 **kwargs):


        # number of levels to the hierarchy.

        self.__dict__.update(kwargs)
        self.hierarchy_levels = len(kwargs.items())
        self.docs_dir = docs_dir
        # support directory path or list of document paths
        if self.docs_dir is None:
            self.doc_paths = doc_paths
        else:
            self.doc_paths = [x for x in Path(
                self.docs_dir).glob("**/*") if x.is_file()]
        self.string_filters = string_filters
        self.topic_range = topic_range

        self.tokens = None
        self.corpus = None
        self.id2word = None
        self.model_eval_info = {"models":[],
                                "num_topics_coherence":[],
                                "optimal_model": None}

#            self.model_eval_info = {"level_one": sub_dict,
#                                    "level_two": sub_dict,
#                                    "level_three": sub_dict}


    # TODO: change name to set_optimal_model. Have it only set a value
    # and return nothing.
    def find_optimal_model(self):
        """
        Parameters
        --------------

        Notes:
        Sets 'optimal_model' value in model_eval_info dictionary"
        """
        models = self.model_eval_info["models"]
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

    def find_optimal_topics(self):
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


    def vis_topics_coherences(self):
        """
        Visualizes coherence vs number of topics.
        """
        num_topics = self.get_num_topics()
        coherence = self.get_coherences()
        if num_topics!=[]:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(num_topics, coherence)
            optimal_topics, _ = self.find_optimal_topics()
            coherence = [round(co, 3) for co in coherence]
            for xy in zip(num_topics, coherence):
                ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
            ax.grid()
            plt.axvline(optimal_topics)
            plt.savefig(f"{optimal_topics}.png")
            return None
        # TO-DO: change this to better type of error?
        raise ValueError("There are no topics or coherence values to\n"
                         "Visualize! You have to run train_models and\n"
                         "train_coherence first!")

    def get_num_topics(self):
        """
        Parameters
        --------------
        Returns
        -------------
            :List[float]
            A list of topic numbers for each model, ASC order
        """
        num_tops_cos = self.model_eval_info["num_topics_coherence"]
        return [num_top_co[0] for num_top_co in num_tops_cos]

    def get_coherences(self):
        """
        Returns
        -------------
            :List[float]
            A list of coherence values in the same order as ascending
            topic_number order.
        """
        num_tops_cos = self.model_eval_info["num_topics_coherence"]
        return [num_top_co[1] for num_top_co in num_tops_cos]


    def train_models_over_range(self, topic_range):
        """
        trains multiple models to find the optimal model.
        Saves the model locally in temp folder and stores the path.
        """
        for num_topic in topic_range:
            model = LdaModel(corpus=self.corpus,
                             num_topics=num_topic,
                             id2word=self.id2word)
            temp_filepath = datapath(f"model_{num_topic}")
            model.save(temp_filepath)
            self.model_eval_info["models"].append(temp_filepath)
            del model

    def train_models(self):
        """
        Convience method that calls train_models_over_range.
        """
        self.train_models_over_range(self.topic_range)

    def train_coherence(self):
        """
        returns a list of tuples with the number of topics and the
        coherence
        This allows for easy evaluation of models.
        """
        models = self.model_eval_info["models"]
        for num_topic, model in zip(
            self.topic_range,
            models):
            model = LdaModel.load(model)
            cm = CoherenceModel(
                    model=model,
                    corpus=self.corpus,
                    texts=self.tokens,
                    coherence='c_v')
            coherence = cm.get_coherence()
            self.model_eval_info["num_topics_coherence"].append(
                    (num_topic, coherence))
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
        trigrams = Phrases(bigram_tokens)
        trigram_tokens = [
            trigrams[doc_token] for doc_token in bigram_tokens]
        print(trigram_tokens)
        self.tokens = trigram_tokens

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

    def format_topics_sentences(self, ldamodel, corpus, tokens, doc_paths):
        data_dict = {'dom_topic':[],
                     'perc_contribution': [],
                     'topic_keywords':[]}
        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    data_dict['dom_topic'].append(int(topic_num))
                    data_dict['perc_contribution'].append(
                        round(prop_topic,4))
                    data_dict['topic_keywords'].append(topic_keywords)
                else:
                    break
        tokens = ['  '.join(tok) for tok in tokens]
        data_dict['tokens'] = tokens
        data_dict['doc_path'] = doc_paths
        sent_topics_df = pd.DataFrame.from_dict(data_dict)
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

    ilda = iLDA(docs_dir="20news_home/",
                string_filters=string_filters,
                topic_range = range(2, 40, 2))

    ilda.set_attributes(with_bigrams=True)
    ilda.train_models()
    ilda.train_coherence()
    ilda.vis_topics_coherences()
    optimal_topics, _ = ilda.find_optimal_topics()
    optimal_model = ilda.find_optimal_model()

    dominant_topic_per_doc = ilda.format_topics_sentences(
        optimal_model,
        ilda.corpus,
        ilda.tokens,
        ilda.doc_paths)

    dominant_topic_per_doc.to_csv("test.csv", index=False)

    for topic_num in range(optimal_topics):
        sub_df = dominant_topic_per_doc[dominant_topic_per_doc["dom_topic"] == topic_num]
        print(sub_df.columns)
        # TODO: replace the 0 with doc_path in format_topic_sentences
        # function
        sub_docs = sub_df["doc_path"].tolist()
        ilda = iLDA(doc_paths=sub_docs,
                    string_filters=string_filters,
                    topic_range = range(2, 10),
                    **kwargs)

        ilda.set_attributes(with_bigrams=True)
        ilda.train_models()
        ilda.train_coherence()
        ilda.vis_topics_coherences()
        optimal_topics, _ = ilda.find_optimal_topics()
        optimal_model = ilda.find_optimal_model()

        dominant_topic_per_doc_sub = ilda.format_topics_sentences(
            optimal_model,
            ilda.corpus,
            ilda.tokens,
            ilda.doc_paths)
        dominant_topic_per_doc_sub.to_csv(f"level_two_{topic_num}.csv", index=False)

if __name__ == "__main__":
    main()
