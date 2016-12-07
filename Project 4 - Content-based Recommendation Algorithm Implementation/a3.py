"""
Online Social Network Analysis - Project 4

Here I have implemented a content-based recommendation algorithm.
It will use the list of genres for a movie as the content.
The data comes from the MovieLens project: http://grouplens.org/datasets/movielens/
"""

from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile


def download_data():
    """ 
    Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ 
    This function tokenizes the data passed after converting to lower case.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Appends a new column to the movies DataFrame with header 'tokens'.
    This contains a list of strings, one per token, extracted
    from the 'genre' field of each movie. 

    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.
    """
    data = []
    tlist = []
    
    #get the genres in a list
    for i in range(len(movies.index)):
        data.append(movies.get_value(index=i, col='genres', takeable=False))
    
    #tokenize each genre in above list
    for d in data:
        tlist.append(tokenize_string(d))
    
    #Append a new column to the movies DataFrame with header 'tokens' and add the token list to it
    movies = movies.assign(tokens=' ')
    movies['tokens'] = tlist
    
    return movies
    pass


def featurize(movies):
    """
    Appends a new column to the movies DataFrame with header 'features'.
    Each row contains a csr_matrix of shape (1, num_features). 
    Each entry in this matrix contains the tf-idf value of the term.

    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i
    
    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    #Append a new column to the movies DataFrame with header 'features'
    movies = movies.assign(features=' ')
        
    #the list of terms
    terms_list = []
    for t in movies['tokens']:
        terms_list.append(t)
    
    #term frequency - tf dict - tf(i,d)
    tf = {}   
    for i in range(len(terms_list)):
        temp = {}
        for each_term in terms_list[i]:
            temp[each_term] = terms_list[i].count(each_term)
        tf[i] = temp
    
    #document frequency - df dict - df(i)
    df = {}
    temp_list = []
    for i in range(len(terms_list)):
        for t in terms_list[i]:
            temp_list.append(t)
    for each_term in temp_list:
        df[each_term] = temp_list.count(each_term)
    
    #max_k tf(k, d)
    max_k = {}
    for i in tf.keys():
        max_k[i] = max(tf.get(i).values())
    
    #N - the number of documents (movies)
    N = len(movies['movieId'])
    
    #Vocab
    pos = 0
    vocab = defaultdict(lambda: 0)
    for s in sorted(df):
        if s not in vocab:
            vocab[s]= pos
            pos += 1
    
    #Calc tfidf and create CSR matrix for each row in movies DataFrame
    tmp = []
    for i in tf:
        data = []
        row = []
        column=  []
        for j in tf[i]:
            data.append((tf.get(i).get(j))/max_k[i] * math.log10(N/df.get(j)))
            column.append(vocab[j])
            row.append(0)
        tmp.append(csr_matrix((data,(row,column)), shape=(1,len(vocab)),dtype='float64'))
    
    movies['features'] = tmp
    return movies, vocab
    pass


def train_test_split(ratings):
    """
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Computes the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    
    #compute the numerator of cosine 
    numerator = np.dot(a,b.T)
    
    #compute the denominator of cosine
    denominator = np.linalg.norm(a.toarray(), ord=None, axis=None, keepdims=False) * np.linalg.norm(b.toarray(), ord=None, axis=None, keepdims=False)
    
    return (numerator/denominator).toarray()[0][0]
    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, now predict the ratings for each
    row in ratings_test.
    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.
    If there are no other movies with positive cosine similarity to use in the
    prediction, the mean rating of the target user in ratings_train is used as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    movieIds = []
    userIds = []
    numpyarray_list = []
    
    movieIds = ratings_test['movieId'].values.tolist()
    
    userIds = ratings_test['userId'].values.tolist()
    
    for (user, movie) in zip(userIds, movieIds):
        user_movies = ratings_train[ratings_train['userId']==user]['movieId'].values.tolist() 
        
        #compute the input for cosine - matrix a 
        a = (movies[movies.movieId==movie]['features'].values)[0]
        
        ratings = ratings_train[ratings_train['userId']==user]['rating'].values.tolist()
        total_weight = 0
        weighted = 0
        total_ratings = 0
        mean = 0
        #calculate the mean
        for each_movie in range(len(user_movies)):
            total_ratings += ratings[each_movie]
        mean = total_ratings/len(user_movies)
        
        #predicting the ratings
        for each_movie in range(len(user_movies)):
            
            #compute the input for cosine - matrix b
            b = (movies[movies.movieId==user_movies[each_movie]]['features'].values)[0]
            
            #calculate cosine similarity
            weighted_value = cosine_sim(a,b)
            if(weighted_value)>0:
                total_weight += weighted_value
                weighted += (weighted_value * ratings[each_movie])
        if(total_weight)>0:
            predicted_ones = weighted/total_weight
        else:
            predicted_ones = mean
        
        numpyarray_list.append(predicted_ones)        
     
    return np.array(numpyarray_list)
    pass


def mean_absolute_error(predictions, ratings_test):
    """
    Returns the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
