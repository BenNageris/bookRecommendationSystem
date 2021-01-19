import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import math

RATINGS_PATH = r"..\ratings.csv"
TAGS_PATH = r"..\tags.csv"
BOOKS_TAGS_PATH = r"..\books_tags.csv"
TEST_PATH = r"..\test.csv"
BOOKS_PATH = r"..\books.csv"
USERS_PATH = r"..\users.csv"

SIMILARITY_FUNCTION = "cosine"

DEFAULT_SEPERATOR = ","
DEFAULT_ENCODING = "ISO-8859-1"

ITEMS = None
USERS = None
PRED_MATRIX = None
COSINE_SIM2 = None


def _load_data(file_path, seperator=DEFAULT_SEPERATOR, encoding=DEFAULT_ENCODING):
    """
    The function gets csv file path, seperator and encoding and returns its dataframe
    :param file_path: string
    :param seperator: string
    :param encoding: string
    :return:
    """
    df = pd.read_csv(file_path, sep=seperator, encoding=encoding)
    return df


def _normalize_items(items):
    """
    The function gets items (books) dataframe as a parameter and returns it with padded index (without the missing book_ids)
    :param items: dataframe
    :return: dataframe
    """
    tmp_items = pd.DataFrame(index=range(items.book_id.max()), columns=items.columns)
    for index, item in items.iterrows():
        tmp_items.at[item.book_id - 1] = item
    return tmp_items


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
    items2 = items.set_index('book_id')
    return items2["title"].iloc[sim_scores]
    # return items[items["book_id"].isin(sim_scores)]["title"]


# non-personalized

# Create rating_vote_count
def create_rating_vote_count(metadata_rating):
    """
    Creating a csv file for the rating file with vote count and vote average
    """
    metadata_rating_new = metadata_rating.groupby(["book_id"]).count()
    metadata_rating_new["rating"] = metadata_rating.groupby(["book_id"]).mean()["rating"]
    metadata_rating_new.columns = ["vote_count", "vote_average"]
    return metadata_rating_new


def get_rating_vote_count_per_location(metadata):
    """
    Get a dataframe for the location and rating combined table with vote count and vote average
    """
    met1 = metadata.groupby(["location", "book_id"])["user_id"].count().reset_index()
    met2 = metadata.groupby(["location", "book_id"])['rating'].mean().reset_index()
    met = pd.merge(met1, met2, on=["location", "book_id"])
    met.columns = ["location", "book_id", "vote_count", "vote_average"]
    return met


def create_rating_vote_count_per_age(metadata):
    """
    Creating a csv file for the age and rating combined table with vote count and vote average
    """
    met1 = metadata.groupby(["age", "book_id"])["user_id"].count().reset_index()
    met2 = metadata.groupby(["age", "book_id"])['rating'].mean().reset_index()
    met = pd.merge(met1, met2, on=["age", "book_id"])
    met.columns = ["age", "book_id", "vote_count", "vote_average"]
    return met


def weighted_rating(x, m, C):
    """
    Function that computes the weighted rating of each book
    """
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v / (v + m) * R) + (m / (m + v) * C)


def normalized_age(x):
    """
    As requested in the exercise, any normalized between X1 to (X+1)0 change to (X+1)0.
    for example:
    34 -> 40, 21 -> 30, 50 -> 50, and so on..
    """
    if type(x) is int:
        a = x
    else:
        a = x['age']
    if a % 10 == 0:
        return a
    return a + (10 - a % 10)


def calc_top_k(k, metadata):
    """
    calculate weighted_rating and print top-k books
    """
    # Calculate mean of vote average column
    C = metadata['vote_average'].mean()

    # Calculate the minimum number of votes required to be in the chart, m
    m = metadata['vote_count'].quantile(0.90)
    q_books = metadata.copy().loc[metadata['vote_count'] >= m]

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_books['score'] = q_books.apply(weighted_rating, axis=1, args=(m, C))

    # Sort movies based on score calculated above
    q_books = q_books.sort_values('score', ascending=False)

    # Print the top 10 books
    return q_books[['book_id', 'title', 'vote_count', 'vote_average', 'score']].head(k)


def get_simply_recommendation(k):
    """
    calculate weighted_rating for each book and print top-k books
    """
    ratings = _load_data(RATINGS_PATH)
    # Load Books Rating Metadata
    metadata_books = _load_data(BOOKS_PATH)
    metadata = create_rating_vote_count(ratings)
    metadata = pd.merge(metadata, metadata_books, on="book_id")
    return calc_top_k(k, metadata)[["book_id", "title"]].set_index("book_id")


def get_simply_place_recommendation(place, k):
    """
    calculate weighted_rating for each book by the location of the rating users and print top-k books
    """
    # Load Books Rating Metadata
    metadata_books = _load_data(BOOKS_PATH)
    metadata_rating = _load_data(RATINGS_PATH)
    metadata_user = _load_data(USERS_PATH)
    metadata = pd.merge(metadata_rating, metadata_user, on="user_id")

    metadata = get_rating_vote_count_per_location(metadata)
    metadata = pd.merge(metadata, metadata_books, on="book_id")

    metadata_per_place = metadata.copy().loc[metadata['location'] == place]

    return calc_top_k(k, metadata_per_place)[["book_id", "title", "score"]].set_index("book_id")


def get_simply_age_recommendation(age, k):
    """
    calculate weighted_rating for each book by the age of the rating users and print top-k books
    """
    age = normalized_age(age)
    # Load Books Rating Metadata
    metadata_books = _load_data(BOOKS_PATH)
    metadata_rating = _load_data(RATINGS_PATH)
    metadata_user = _load_data(USERS_PATH)
    metadata = pd.merge(metadata_rating, metadata_user, on="user_id")
    metadata['age'] = metadata.apply(normalized_age, axis=1)
    metadata = create_rating_vote_count_per_age(metadata)

    metadata = pd.merge(metadata, metadata_books, on="book_id")

    metadata_per_age = metadata.copy().loc[metadata['age'] == age]

    return calc_top_k(k, metadata_per_age)[["book_id", "title", "score"]].set_index("book_id")


# collabrotive filtering
def build_CF_prediction_matrix(sim):
    """
    The function gets as a parameter similarity function and returns user based prediction matrix
    :param sim: "cosine" or "euclidean" or "jaccard"
    :return: matrix
    """
    global USERS, RATINGS, ITEMS, DATA_MATRIX
    n_users = USERS.user_id.max()
    n_items = ITEMS.book_id.max()
    DATA_MATRIX = np.empty((n_users, n_items))
    DATA_MATRIX[:] = np.nan
    for line in RATINGS.itertuples():
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
    PRED_MATRIX = mean_user_rating + user_similarity.dot(ratings_diff) / np.array(
        [np.abs(user_similarity).sum(axis=1)]).T
    return PRED_MATRIX


def get_CF_recommendation(user_id, k):
    """
    The function gets user_id and k and returns top most recommended books for user based on user-based recommendation
    :param user_id: int
    :param k: int
    :return: dataframe
    """
    global DATA_MATRIX, ITEMS, PRED_MATRIX
    if PRED_MATRIX is None:
        PRED_MATRIX = build_CF_prediction_matrix(SIMILARITY_FUNCTION)
    predicted_ratings_row = PRED_MATRIX[user_id - 1]
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
        BOOKS_WITH_TAG_NAME = books.merge(books_with_tags_names, on="goodreads_book_id",
                                          how="left")
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
    global BOOKS_WITH_TAG_NAME, COSINE_SIM2
    if COSINE_SIM2 is None:
        COSINE_SIM2 = build_contact_sim_metrix()
    # Get the index of the book that matches the title
    metadata = BOOKS_WITH_TAG_NAME.reset_index()
    indices = pd.Series(metadata.index, index=metadata['original_title'])

    idx = indices[book_name]
    if type(idx) == pd.core.series.Series and idx.count() > 1:
        print("There are {} books with original title:{}, assuming you asked for similarity for the first one".format(
            idx.count(), book_name))
        idx = idx[0]
    # Get the pairwise similarity scores of all books with that book
    sim_scores = list(enumerate(COSINE_SIM2[idx]))
    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the K most similar books (the first is the books we asked)
    sim_scores = sim_scores[1:k + 1]
    book_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar books
    books = _load_data(BOOKS_PATH)
    return books['title'].iloc[book_indices]


def precision_k(k):
    """
    The function takes as a parameter integer and checks precision rate for k recommendations
    :param k: int
    :return: float
    """
    test = _load_data(TEST_PATH)
    test_only_4_and_5 = test.loc[test['rating'].isin([4, 5])]
    unique_user_ids = test_only_4_and_5.user_id.unique()
    user_ids_to_check = np.array([], dtype=int)
    for user_id in unique_user_ids:
        if test_only_4_and_5[test_only_4_and_5.user_id == user_id].book_id.count() >= k:
            user_ids_to_check = np.append(user_ids_to_check, user_id)
    total = 0
    # foreach user_id in user_ids_to_check
    for user_id in user_ids_to_check:
        k_recommendations = get_CF_recommendation(user_id, k)
        real_ratings_of_user_id = test_only_4_and_5[test_only_4_and_5.user_id == user_id].book_id
        num_of_hits = len(np.intersect1d(k_recommendations.index, real_ratings_of_user_id))
        total = total + (float(num_of_hits) / k)
    return total / len(user_ids_to_check)


def ARHR(k):
    tests = _load_data(TEST_PATH)
    test_only_4_and_5 = tests[tests['rating'].isin([4, 5])]
    user_ids_to_check = np.array([], dtype=int)
    for user_id in test_only_4_and_5['user_id'].unique():
        if test_only_4_and_5[test_only_4_and_5.user_id == user_id].book_id.count() >= k:
            user_ids_to_check = np.append(user_ids_to_check, user_id)
    test_only_4_and_5 = test_only_4_and_5.loc[test_only_4_and_5['user_id'].isin(user_ids_to_check)]
    total_sum = 0
    unique_tests_4_5_user_ids = test_only_4_and_5['user_id'].unique()
    for user in unique_tests_4_5_user_ids:
        recommendations = get_CF_recommendation(user, k)
        for item_index in range(k):
            recommend_idx = recommendations.index[item_index]
            if recommend_idx in list(test_only_4_and_5[test_only_4_and_5['user_id'] == user]['book_id']):
                total_sum += 1 / (item_index + 1)
    return total_sum / len(unique_tests_4_5_user_ids)


def rmse():
    """
    The function gets k and returns the RMSE rate
    :return: float
    """
    pred_matrix = build_CF_prediction_matrix(SIMILARITY_FUNCTION)
    test_ratings = _load_data(TEST_PATH)
    sum = 0
    cnt = 0
    for _, row in test_ratings.iterrows():
        user_id = row['user_id'].astype(int)
        book_id = row['book_id'].astype(int)
        rating = row['rating'].astype(int)
        diff = rating - pred_matrix[user_id - 1, book_id - 1]
        sum = sum + math.pow(diff, 2)
        cnt = cnt + 1
    return math.sqrt(sum / cnt)


if __name__ == "__main__":
    # loads to increase efficiency
    RATINGS = _load_data(RATINGS_PATH)
    ITEMS = _load_data(BOOKS_PATH)
    USERS = _load_data(USERS_PATH)
    ITEMS = _normalize_items(ITEMS)

    # PART A
    K = 10
    print(get_simply_recommendation(K))
    print("\n\n")
    K = 10
    place = "Ohio"
    print(get_simply_place_recommendation(place, K))
    print("\n\n")
    K = 10
    age = 28
    print(get_simply_age_recommendation(age, K))
    print("\n\n")

    # PART B
    # print(get_CF_recommendation(1, 10))

    # PART C
    book_name = "Twilight"
    k = 10
    print("The {} best recommendations for the book {} are:".format(book_name, k))
    print(get_contact_recommendation(book_name, k))

    # PART D
    print(precision_k(10))
    print(ARHR(10))
    print(rmse())
