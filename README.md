# iLDA - iterative LDA

###### An unsupervised algorithm to automatically construct a hierarchy of topics.

iLDA creates a hierarchy of topics that can be mapped back to individual documents. The algorithm trains an LDA model with a number of topics n, assigns the dominant topic for each document, groups the documents by dominant topic (n groups), and then **trains an LDA model on each subset of documents**. The process repeats itself to create a hierarchy of documents. The level of the hierarchy corresponds to the number of times this process repeats.

- iLDA is a wrapper around the LDA model used in the python library [Gensim](https://radimrehurek.com/gensim/) and **provides convenience methods for:**
  - **preprocessing text** including tokenization and other filters, the option to use bigrams or trigrams, and creating the word dictionary needed to train LDA.
  - **training a range of models**, and utilizing the 'elbow method' to **automatically** pick the **best model**
  - visualizing the coherence scores of the range of models trained.
  - providing a data frame of the tokens, document paths, dominant topic for each document, the dominant topic key words, and the percent contribution of the document topics

### TO-DO

------------------------

- create a helper method/ helper class that performs the iteration. Currently, the process is simply in the main method of iLDA_wrapper.py
- Merge the results of the iteration into a single dataframe
- visualize the merged results using plotly. Plotly provides a nice way to visualize hierarchies.
- Make available as a general python package.
- Create an example notebook for the 20newsgroup dataset.

### References

-----------------------

Some of the code was adapted from this [post](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)

