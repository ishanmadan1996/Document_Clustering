This is my revision of the tutorial at http://brandonrose.org/clustering
Data: All movies totalling 1370 extracted from www.wogma.com (extracted reviews and movie titles)

Goal: Put 1370 movies into 5 clusters by text-mining their reviews and plotted results is in 'Results' Directory

Broad generalised steps:-
1)Read data: read titles, reviews, rankings into two arrays
2)Tokenize and stem: break paragraphs into sentences, then to words, stem the words (without removing stopwords) - each review essentially becomes a bag of stemmed words.
3)Generate tf-idf matrix: each row is a term (unigram, bigram, trigram...generated from the bag of words in 2.), each column is a review.
4)Generate clusters: based on the tf-idf matrix, 5 (or any number) clusters are generated using k-means. The top key terms are selected for each cluster.
5)Calculate similarity: generate the cosine similarity matrix using the tf-idf matrix (100x100), then generate the distance matrix (1 - similarity matrix), so each pair of review has a distance number between 0 and 1.
6)Plot clusters: use multidimensional scaling (MDS) to convert distance matrix to a 2-dimensional array, each review has (x, y) that represents their relative location based on the distance matrix. Plot the 100 points with their (x, y) using matplotlib.

Exact detailed steps taken by me:-
1) I've extracted all movie titles and their reviews from wogma.com.
2) I made 2 lists containing titles and reviews.
3) From the reviews I made a vocabulary of all stemmed and tokenised words.
4) Then I made a tfidf matrix using the reviews list.
5) I then performed k means clustering on the tfidf matrix, and ran it a couple of times till I got an optimal training time (18s). Then I saved that model in a pkl file.
6) Then, using the vocabulary built in step 3, I found top 6-7 terms which were closest to each corresponding cluster centroid. The movie titles or names within each cluster can also be shown.
7) Finally I plotted all clusters, wherein the cluster names in the graph were represented by the top terms of each cluster found in previous step. The data points in each cluster were named as the titles of the movie.