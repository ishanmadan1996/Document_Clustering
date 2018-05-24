from sklearn.feature_extraction.text import TfidfVectorizer
from IPython import get_ipython
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
import time


#Read movie titles
titles = open('C:\\Users\Ishant\Desktop\Scraped_Data_Unfound\\title.txt').read().split('\n')
print(len(titles))

titles = titles[:1370]

# Read in the reviews from extracted text
reviews = open('C:\\Users\Ishant\Desktop\Scraped_Data_Unfound\\reviews.txt').read().split('\n\nBreak Here')
print(len(reviews))
reviews = reviews[:1370]


# strips html formatting and converts to unicode
reviews_clean = []
for text in reviews:
    text = BeautifulSoup(text, 'html.parser').getText()
    reviews_clean.append(text)
reviews = reviews_clean

# generates index or rank  for each item in the corpora
ranks = []
for i in range(1, len(titles)+1):
    ranks.append(i)


# load nltk's English stopwords as variable called 'stopwords'
# use nltk.download() to install the corpus first
# Stop Words are words which do not contain important significance to be used in Search Queries
stopwords = nltk.corpus.stopwords.words('english')
# load nltk's SnowballStemmer as variabled 'stemmer'
stemmer = SnowballStemmer("english")
print(len(stopwords))
print(stopwords)



# here I have defined a tokenizer and stemmer which returns the set of stems in the text that it is passed
# Punkt Sentence Tokenizer, sent means sentence
def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


words_stemmed = tokenize_and_stem("A parent’s job is never done. Even if they are 102 and going on 103,")
print(words_stemmed)

words_only = tokenize_only("A parent’s job is never done. Even if they are 102 and going on 103")
print(words_only)

# Below I use my stemming/tokenizing and tokenizing functions to iterate over the list of reviews to create two vocabularies: one stemmed and one only tokenized


totalvocab_stemmed = []
totalvocab_tokenized = []
for i in reviews:
    allwords_stemmed = tokenize_and_stem(i)  # for each item in 'reviews', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)


print(len(totalvocab_stemmed))
print(len(totalvocab_tokenized))

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed) #create a pandas frame for total vocab, which will later be used to find most frequent terms.
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
print(vocab_frame.head())



words_frame = pd.DataFrame({'WORD': words_only}, index = words_stemmed)
print('there are ' + str(words_frame.shape[0]) + ' items in words_frame')
print(words_frame)


# Generate TF-IDF matrix
#
# max_df: this is the maximum frequency within the documents a given feature can have to be used in the tfi-idf matrix. If the term is in greater than 80% of the documents it probably cares little meanining (in the context of film synopses)
#
# min_idf: this could be an integer (e.g. 5) and the term would have to be in at least 5 of the documents to be considered. Here I pass 0.2; the term must be in at least 20% of the document. I found that if I allowed a lower min_df I ended up basing clustering on names--for example "Michael" or "Tom" are names found in several of the movies and the synopses use these names frequently, but the names carry no real meaning.
#
# ngram_range: this just means I'll look at unigrams, bigrams and trigrams


# Note that the result of this block takes a while to show


#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

#get_ipython().magic(u'time tfidf_matrix = tfidf_vectorizer.fit_transform(synopses) #fit the vectorizer to synopses')
tfidf_matrix = tfidf_vectorizer.fit_transform(reviews)
# (100, 563) means the matrix has 100 rows or sentences  and 563 columns or terms
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()
print(len(terms))

from sklearn.cluster import KMeans

num_clusters = 5

km = KMeans(n_clusters=num_clusters)
start = time.time()
km.fit(tfidf_matrix)
print('Trained in %.1f seconds' % (time.time() - start))

clusters = km.labels_.tolist()
print(clusters)

from sklearn.externals import joblib

#uncomment the below to save your model
# since I've already run my model I am loading from the pickle

#joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

# Here, I create a dictionary of titles, ranks, the reviews , the cluster assignment, and the genre [genre and reviews were scraped from wogma].
# I convert this dictionary to a Pandas DataFrame for easy access. I'm a huge fan of Pandas and recommend taking a look at some of its awesome functionality which I'll use below, but not describe in a ton of detail.



films = { 'title': titles, 'rank': ranks, 'reviews': reviews, 'cluster': clusters}

frame = pd.DataFrame(films, index = [clusters] , columns = ['rank', 'title', 'cluster'])

print(frame) # here the ranking is still 0 to 99

frame['cluster'].value_counts() #number of films per cluster (clusters from 0 to 4)


# In[44]:

grouped = frame['rank'].groupby(frame['cluster']) # groupby cluster for aggregation purposes

grouped.mean() # average rank (1 to 100) per cluster

# Here is  indexing and sorting on each cluster to identify which are the top n (I chose n=6) words that are nearest to the cluster centroid. This gives a good sense of the main topic of the cluster.

from __future__ import print_function

print("Top terms per cluster:")

# sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print()  # add whitespace
    print()  # add whitespace

    print("Cluster %d titles:" % i, end='')
    for title in frame.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()  # add whitespace
    print()  # add whitespace

# Cosine similarity is measured against the tf-idf matrix and can be used to generate a measure of similarity between each document and the other documents in the corpus (each synopsis among the synopses). cosine similarity 1 means the same document, 0 means totally different ones. dist is defined as 1 - the cosine similarity of each document.  Subtracting it from 1 provides cosine distance which I will use for plotting on a euclidean (2-dimensional) plane.
# Note that with dist it is possible to evaluate the similarity of any two or more synopses.

# In[46]:

from sklearn.metrics.pairwise import cosine_similarity

similarity_distance = 1 - cosine_similarity(tfidf_matrix)
print(type(similarity_distance))
print(similarity_distance.shape)



import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)


pos = mds.fit_transform(similarity_distance)
print(pos.shape)
print(pos)

xs, ys = pos[:, 0], pos[:, 1]
print(type(xs))
print(xs)

# Visualizing document clusters
# In this section, I demonstrate how you can visualize the document clustering output using matplotlib and mpld3 (a matplotlib wrapper for D3.js).
# First I define some dictionaries for going from cluster number to color and to cluster name. I based the cluster names off the words that were closest to each cluster centroid.


# set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

# set up cluster names using a dict
cluster_names = {0: 'love, lively, movie',
                 1: 'story, dialogue, times',
                 2: 'watch, cinema, however',
                 3: 'just, story, only',
                 4: 'does, story, just'}

# Next, I plot the labeled observations (films, film titles) colored by cluster using matplotlib. I won't get into too much detail about the matplotlib plot, but I tried to provide some helpful commenting.

# In[49]:

# some ipython magic to show the matplotlib plots inline
get_ipython().magic(u'matplotlib inline')

# create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=titles))

print(df[1:10])
# group by cluster
# this generate {name:group(which is a dataframe)}
groups = df.groupby('label')
print(groups.groups)

# set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
# ms: marker size
for name, group in groups:
    print("*******")
    print("group name " + str(name))
    print(group)
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=20,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',  # changes apply to the x-axis
                   which='both',  # both major and minor ticks are affected
                   bottom='off',  # ticks along the bottom edge are off
                   top='off',  # ticks along the top edge are off
                   labelbottom='off')
    ax.tick_params(axis='y',  # changes apply to the y-axis
                   which='both',  # both major and minor ticks are affected
                   left='off',  # ticks along the bottom edge are off
                   top='off',  # ticks along the top edge are off
                   labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
# for i in range(len(df)):
#     ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=10)


plt.savefig('C:\\Users\Ishant\Desktop\Scraped_Data_Unfound\\clusters.png')
plt.show()  # show the plot