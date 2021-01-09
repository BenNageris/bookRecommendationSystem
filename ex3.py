import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import heapq

from sklearn.metrics.pairwise import pairwise_distances

DEFAULT_SEPERATOR = ","
SIM = "cosine"


def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
    return arr


def _load_data(file_path, seperator=DEFAULT_SEPERATOR):
    df = pd.read_csv(file_path, sep=seperator, encoding = "ISO-8859-1")  # , names=r_cols, encoding='latin-1')
    return df


def _get_data_matrix(ratings):
    max_user_id = ratings.user_id.max()
    max_book_id = ratings.book_id.max()
    data_matrix = np.empty((max_user_id, max_book_id))
    data_matrix[:] = np.nan
    for line in ratings.itertuples():
        user = line[1] - 1
        book = line[2] - 1
        rating = line[3]
        data_matrix[user, book] = rating
    return data_matrix


def build_CF_prediction_matrix(sim):
    """
    The function gets as parameter similarity metric and returns prediction matrix
    :param sim: string
    :return: matrix
    """
    ratings = _load_data(r"..\ratings.csv")
    # TODO: need to check with osnat about number of books or max_id for matrix
    n_users = ratings.user_id.unique().shape[0]
    n_books = ratings.book_id.unique().shape[0]

    # create ranking table - that table is sparse
    data_matrix = _get_data_matrix(ratings)
    # calc mean
    mean_user_rating = np.nanmean(data_matrix, axis=1).reshape(-1, 1)

    ratings_diff = (data_matrix - mean_user_rating)
    # replace nan -> 0
    ratings_diff[np.isnan(ratings_diff)] = 0

    # calculate user x user similarity matrix
    user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)
    print(user_similarity.shape)

    # For each user (i.e., for each row) keep only k most similar users, set the rest to 0.
    # Note that the user has the highest similarity to themselves.
    # TODO: decide size of k
    k = 10
    user_similarity = np.array([keep_top_k(np.array(arr), k) for arr in user_similarity])
    pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
    return pred


def get_CF_recommendation(user_id, k):
    ratings = _load_data(r"..\ratings.csv")
    data_matrix = _get_data_matrix(ratings)
    pred = build_CF_prediction_matrix(SIM)
    predicted_ratings_row = pred[user_id]
    data_matrix_row = data_matrix[user_id]
    predicted_ratings_unrated = predicted_ratings_row[np.isnan(data_matrix_row)]
    idx = np.argsort(-predicted_ratings_unrated)
    sim_scores = idx[0:k]
    books = _load_data(r"..\books.csv")
    return books.iloc[sim_scores]


# get_CF_recommendation(1, 10)

def _idx_to_user_id(idx, user_id_unique=None, indices_user_id_unique=None):
    if user_id_unique is None:
        print("You have to pass at least one of user_id_unique or indices_user_id_unique")
        return None
    if indices_user_id_unique is None:
        indices_user_id_unique_ = user_id_unique.argsort()
    else:
        indices_user_id_unique_ = indices_user_id_unique
    return user_id_unique[indices_user_id_unique_[idx]]


def _user_id_to_idx(idx, user_id_unique=None, indices_user_id_unique=None):
    if user_id_unique is None and indices_user_id_unique == None:
        print("You have to pass at least one of user_id_unique or indices_user_id_unique")
        return None
    if indices_user_id_unique is not None:
        return indices_user_id_unique[idx]
    return user_id_unique.argsort()[idx]


print(get_CF_recommendation(1, 10))
