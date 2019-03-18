import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import sys

sys.path.append('D:/data_science/movie_recomend')
warnings.filterwarnings('ignore')

def data_clean():
    df_rate = pd.read_csv('D:/data_science/movie_recomend/ratings.csv', sep=',')
    df_movie = pd.read_csv('D:/data_science/movie_recomend/movies.csv',sep=',')
    df = pd.merge(df_rate, df_movie, on='movieId')
    #print(df.describe())
    rating = pd.DataFrame(df.groupby('title')['rating'].mean())
    rating['number_of_rating'] = df.groupby('title')['rating'].count()
    movielist = df['title'].unique().tolist()
    userIdlist = df['userId'].unique().tolist()
    row = df.userId.astype('category', categories=userIdlist).cat.codes
    col = df.title.astype('category', categories=movielist).cat.codes
    ratinglist = df.rating.tolist()
    movie_matrix = csr_matrix((ratinglist, (row, col)), shape=(len(userIdlist), len(movielist)))
    movie_matrix = pd.SparseDataFrame([ pd.SparseSeries(movie_matrix[i].toarray().ravel(), fill_value=0)
                              for i in np.arange(movie_matrix.shape[0]) ],
                       index=userIdlist, columns=movielist, default_fill_value=0)
    rating.sort_values(by='rating',ascending=False,inplace=True)
    movie_target = movie_matrix['Air Force One (1997)']
    movie_coef = movie_matrix.corrwith(movie_target)

if __name__ == '__main__':
    data_clean()