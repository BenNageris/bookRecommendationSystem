import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import heapq
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

DEFAULT_SEPERATOR = ","
SIM = "cosine"


def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
    return arr


def _load_data(file_path, seperator=DEFAULT_SEPERATOR):
    df = pd.read_csv(file_path, sep=seperator, encoding="ISO-8859-1")  # , names=r_cols, encoding='latin-1')
    return df


# collabrative filtering

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


# print(get_CF_recommendation(1, 10))

# Contact based filtering
def build_contact_sim_metrix():
    def _get_books_tag_name(books):
        books_tags = pd.read_csv(r"..\books_tags.csv", encoding="ISO-8859-1", sep=",")
        tags = pd.read_csv(r"..\tags.csv", encoding="ISO-8859-1", sep=",")
        books_tags['tag_id'] = books_tags['tag_id'].astype('int')
        tags['tag_id'] = tags['tag_id'].astype('int')
        books_tags_2 = books_tags.merge(tags, on="tag_id")
        books_with_tags_names = books_tags_2.groupby(["goodreads_book_id"])['tag_name'].apply(' '.join).reset_index()
        books['goodreads_book_id'] = books['goodreads_book_id'].astype('int')
        return books.merge(books_with_tags_names, on="goodreads_book_id", how="left")
        # return books.merge(books_with_tags_names, on="goodreads_book_id")

    def _clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            # Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    def create_soup(x):
        tagged_name = "" if x['tag_name'] is np.nan else x['tag_name']
        return x['title'] + tagged_name + ' ' + x['authors'] + ' ' + x['language_code']

    books = pd.read_csv(r"..\books.csv", encoding="ISO-8859-1", sep=DEFAULT_SEPERATOR)

    books_with_tags_names = _get_books_tag_name(books)
    # Apply clean_data function to your features.
    features = ['authors', 'language_code']
    for feature in features:
        books_with_tags_names[feature] = books_with_tags_names[feature].apply(_clean_data)
    # print(books[['title', 'authors']].head(3))
    books_with_tags_names['soup'] = books_with_tags_names.apply(create_soup, axis=1)
    from sklearn.feature_extraction.text import CountVectorizer

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(books_with_tags_names['soup'])
    print(count_matrix.shape)

    # Add 8
    # Compute the Cosine Similarity matrix based on the count_matrix
    # Reset index of your main DataFrame and construct reverse mapping as before
    metadata = books_with_tags_names.reset_index()
    indices = pd.Series(metadata.index, index=metadata['title'])
    return indices, count_matrix


def get_contact_recommendation(book_name, k):
    indices, count_matrix = build_contact_sim_metrix()
    # Get the index of the movie that matches the title
    idx = indices[book_name]

    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim2[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies (the first is the movie we asked)
    sim_scores = sim_scores[1:k]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    books = _load_data(r"..\books.csv")
    return books['title'].iloc[movie_indices]


print(get_contact_recommendation("Twilight: The Complete Illustrated Movie Companion", 10))
