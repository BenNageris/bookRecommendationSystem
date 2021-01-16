import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

RATINGS_PATH = r"../ratings.csv"
BOOKS_PATH = r"../books.csv"
TAGS_PATH = r"../tags.csv"
BOOKS_TAGS_PATH = r"../books_tags.csv"

SIMILARITY_FUNCTION = "cosine"

DEFAULT_SEPERATOR = ","
DEFAULT_ENCODING = "ISO-8859-1"


# function gets csv file path, seperator and encoding and returns its dataframe
def _load_data(file_path, seperator=DEFAULT_SEPERATOR, encoding=DEFAULT_ENCODING):
    df = pd.read_csv(file_path, sep=seperator, encoding=encoding)
    return df


# utils from TIRGUL colab
def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
    return arr


#  Takes as a parameter book title and outputs most similar books
def get_recommendations(predicted_ratings_row, data_matrix_row, items, k=5):
    predicted_ratings_row[~np.isnan(data_matrix_row)] = 0
    idx = np.argsort(-predicted_ratings_row)
    sim_scores = idx[0:k]
    items = items.set_index('book_id')
    return items['title'].iloc[sim_scores]

# collabrotive filtering
def build_CF_prediction_matrix(sim):
    """
    The function gets as a parameter similarity function and returns user based prediction matrix
    :param sim: "cosine" or "euclidean" or "jaccard"
    :return: matrix
    """
    global ITEMS, DATA_MATRIX
    ratings = _load_data(RATINGS_PATH)
    ITEMS = _load_data(BOOKS_PATH)
    n_users = ratings.user_id.astype(int).max()
    n_items = ratings.book_id.astype(int).max()
    DATA_MATRIX = np.empty((n_users, n_items))
    DATA_MATRIX[:] = np.nan
    for line in ratings.itertuples():
        user = line[1] - 1
        book = line[2] - 1
        rating = line[3]
        DATA_MATRIX[user, book] = rating
    mean_user_rating = np.nanmean(DATA_MATRIX, axis=1).reshape(-1, 1)
    ratings_diff = (DATA_MATRIX - mean_user_rating)
    ratings_diff[np.isnan(ratings_diff)] = 0
    # calculate user x user similarity matrix
    user_similarity = 1 - pairwise_distances(ratings_diff, metric=sim)
    user_similarity = np.array([keep_top_k(np.array(arr), k=10) for arr in user_similarity])
    return mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T

def get_CF_recommendation(user_id, k):
    """
    The function gets user_id and k and returns top most recommended books for user based on user-based recommendation
    :param user_id: int
    :param k: int
    :return: dataframe
    """
    global DATA_MATRIX, ITEMS
    pred_matrix = build_CF_prediction_matrix(SIMILARITY_FUNCTION)
    predicted_ratings_row = pred_matrix[user_id - 1]
    data_matrix_row = DATA_MATRIX[user_id - 1]

    return get_recommendations(predicted_ratings_row, data_matrix_row, ITEMS, k)


# contact based filtering
def build_contact_sim_metrix():
    """
    The function returns contact based similarity matrix
    :return: matrix
    """
    global COUNT_MATRIX, INDICES, BOOKS_WITH_TAG_NAME

    # Contact based filtering
    def _get_books_tag_name(books):
        global BOOKS_WITH_TAG_NAME
        books_tags = _load_data(BOOKS_TAGS_PATH)
        tags = _load_data(TAGS_PATH)
        books_tags['tag_id'] = books_tags['tag_id'].astype('int')
        tags['tag_id'] = tags['tag_id'].astype('int')
        books_tags_2 = books_tags.merge(tags, on="tag_id")
        books_with_tags_names = books_tags_2.groupby(["goodreads_book_id"])['tag_name'].apply(' '.join).reset_index()
        books['goodreads_book_id'] = books['goodreads_book_id'].astype('int')
        BOOKS_WITH_TAG_NAME = books.merge(books_with_tags_names, on="goodreads_book_id", how="left")
        return BOOKS_WITH_TAG_NAME

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

    books = _load_data(BOOKS_PATH)

    books_with_tags_names = _get_books_tag_name(books)
    # Apply clean_data function to your features.
    features = ['authors', 'language_code']
    for feature in features:
        books_with_tags_names[feature] = books_with_tags_names[feature].apply(_clean_data)
    books_with_tags_names['soup'] = books_with_tags_names.apply(create_soup, axis=1)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(books_with_tags_names['soup'])
    # Add 8
    # Compute the Cosine Similarity matrix based on the count_matrix
    return cosine_similarity(count_matrix, count_matrix)


def get_contact_recommendation(book_name, k):
    """
    The function gets book name and k and returns top most recommended books based on book
    :param book_name: string
    :param k: int
    :return: dataframe
    """
    global BOOKS_WITH_TAG_NAME
    cosine_sim2 = build_contact_sim_metrix()
    # Get the index of the book that matches the title
    metadata = BOOKS_WITH_TAG_NAME.reset_index()
    indices = pd.Series(metadata.index, index=metadata['original_title'])

    idx = indices[book_name]
    if idx.count() > 1:
        print("There are fre books with original title:{}, assuming you asked for similarity for the first one".format(
            book_name))
        idx = idx[0]
    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim2[idx]))
    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the K most similar books (the first is the books we asked)
    sim_scores = sim_scores[1:k + 1]
    book_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar books
    books = _load_data(BOOKS_PATH)
    return books['title'].iloc[book_indices]


if __name__ == "__main__":
    # PART B
    print(get_CF_recommendation(1, 10))
    print(get_CF_recommendation(511, 10))

    # PART C
    book_name = "Twilight"
    k = 10
    print("The {} best recommendations for the book {} are:".format(book_name, k))
    print(get_contact_recommendation(book_name, k))

    # PART D