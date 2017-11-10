from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

import numpy as np

class Vectorizer:
    '''
    vectorize text data
    '''
    def __init__(self, max_df=0.5, max_features=10000, min_df=2, use_idf=True):
        self.vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features, min_df=min_df,
                                          stop_words='english', use_idf=use_idf)

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X)


class DimensionalityReduction:
    '''
    do dimension Reduction on original data vectors
    '''
    def __init__(self, n_components=100):
        # TODO: use make_pipeline to reduce dimension of data
        #.      use SVD in the pipeline to do dimensionality reduction
        #.      use Normalizer to normalize output
        self.svd = None
        self.normalizer = None
        self.DR = None

    def fit_transform(self, X):
        # TODO: fit and transform data

class KM:
    '''
    do clustering on dataset
    '''
    def __init__(self, n_clusters, svd, vectorizer, max_iter=100, n_init=1):
        self.n_clusters = n_clusters
        self.svd = svd
        self.vectorizer = vectorizer
        # CHANGE HERE TO USE KMEANS
        self.km = None
        
    def fit(self, X):
        # fit your data to do clustering

    def print(self):
        # print top topics
        original_space_centroids = self.svd.inverse_transform(self.km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
        terms = self.vectorizer.get_feature_names()
        for i in range(self.n_clusters):
            print('topics: {}'.format(i), end=' ')
            for ind in order_centroids[i, :10]:
                print('{}'.format(terms[ind]), end=' ')
            print()

if __name__ == '__main__':
    with open('../data/20newsgroup.json', 'r') as f:
        # load dataset from json file
        data_string = f.read()
        dataset = json.loads(data_string)

        # get text data and labels
        data = dataset['text']
        labels = dataset['label']
        unique_labels = np.unique(labels)
        
        # You will need to do clustering both for words and documents, be careful about whether to use X or X^T
        # You will also need to summarize clustering results
        # What follows is a rough structure, you will need to call them in the right setup

        # vectorize text data
        vectorizer = Vectorizer()
        X = vectorizer.fit_transform(data)

        # do dimensionality reduction
        dr = DimensionalityReduction()
        X = dr.fit_transform(X)

        # do clustering
        km = KM(n_clusters=len(unique_labels), svd=dr.svd, vectorizer=vectorizer.vectorizer)
        km.fit(X)
        km.print()
