import pandas as pd

RATING_VOTE_FILE = "dataset\\ordered_rating.csv"
LOCATION_VOTE_FILE = "dataset\\ordered_location.csv"
AGE_VOTE_FILE = "dataset\\ordered_age.csv"


# Create rating_vote_count
def create_rating_vote_count(metadata_rating):
    """
    Creating a csv file for the rating file with vote count and vote average
    """
    metadata_rating_new = metadata_rating.groupby(["book_id"]).count()
    metadata_rating_new["rating"] = metadata_rating.groupby(["book_id"]).mean()["rating"]
    metadata_rating_new.columns = ["vote_count", "vote_average"]
    metadata_rating_new.to_csv(RATING_VOTE_FILE, index=True, index_label="book_id")
    del metadata_rating_new


def create_rating_vote_count_per_location(metadata):
    """
    Creating a csv file for the location and rating combined table with vote count and vote average
    """
    met1 = metadata.groupby(["location", "book_id"])["user_id"].count().reset_index()
    met2 = metadata.groupby(["location", "book_id"])['rating'].mean().reset_index()
    met = pd.merge(met1, met2, on=["location", "book_id"])
    met.columns = ["location", "book_id", "vote_count", "vote_average"]
    met.to_csv(LOCATION_VOTE_FILE, index=False)
    del met1, met2, met


def create_rating_vote_count_per_age(metadata):
    """
    Creating a csv file for the age and rating combined table with vote count and vote average
    """
    met1 = metadata.groupby(["age", "book_id"])["user_id"].count().reset_index()
    met2 = metadata.groupby(["age", "book_id"])['rating'].mean().reset_index()
    met = pd.merge(met1, met2, on=["age", "book_id"])
    met.columns = ["age", "book_id", "vote_count", "vote_average"]
    met.to_csv(AGE_VOTE_FILE, index=False)
    del met1, met2, met


def weighted_rating(x, m, C):
    """
    Function that computes the weighted rating of each book
    """
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


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


def get_simply_recommendation(k):
    """
    calculate weighted_rating for each book and print top-k books
    """
    # Load Books Rating Metadata
    metadata_books = pd.read_csv('dataset\\books.csv', low_memory=False, encoding="ISO-8859-1")
    metadata = pd.read_csv(RATING_VOTE_FILE, low_memory=False, encoding="ISO-8859-1")
    metadata = pd.merge(metadata, metadata_books, on="book_id")

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
    print(q_books[['book_id', 'title', 'vote_count', 'vote_average', 'score']].head(k))

    del metadata, metadata_books, q_books


def get_simply_place_recommendation(place, k):
    """
    calculate weighted_rating for each book by the location of the rating users and print top-k books
    """
    # Load Books Rating Metadata
    metadata_books = pd.read_csv('dataset\\books.csv', low_memory=False, encoding="ISO-8859-1")
    metadata = pd.read_csv(LOCATION_VOTE_FILE, low_memory=False, encoding="ISO-8859-1")
    metadata = pd.merge(metadata, metadata_books, on="book_id")

    metadata_per_place = metadata.copy().loc[metadata['location'] == place]
    # Calculate mean of vote average column
    C = metadata_per_place['vote_average'].mean()

    # Calculate the minimum number of votes required to be in the chart, m
    m = metadata_per_place['vote_count'].quantile(0.90)
    q_books = metadata_per_place.copy().loc[metadata_per_place['vote_count'] >= m]

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_books['score'] = q_books.apply(weighted_rating, axis=1, args=(m, C))

    # Sort movies based on score calculated above
    q_books = q_books.sort_values('score', ascending=False)

    # Print the top 10 books
    print(q_books[['book_id', 'title', 'vote_count', 'vote_average', 'score']].head(k))

    del metadata, metadata_books, metadata_per_place, q_books


def get_simply_age_recommendation(age, k):
    """
    calculate weighted_rating for each book by the age of the rating users and print top-k books
    """
    age = normalized_age(age)
    # Load Books Rating Metadata
    metadata_books = pd.read_csv('dataset\\books.csv', low_memory=False, encoding="ISO-8859-1")
    metadata = pd.read_csv(AGE_VOTE_FILE, low_memory=False, encoding="ISO-8859-1")
    metadata = pd.merge(metadata, metadata_books, on="book_id")

    metadata_per_age = metadata.copy().loc[metadata['age'] == age]
    # Calculate mean of vote average column
    C = metadata_per_age['vote_average'].mean()

    # Calculate the minimum number of votes required to be in the chart, m
    m = metadata_per_age['vote_count'].quantile(0.90)
    q_books = metadata_per_age.copy().loc[metadata_per_age['vote_count'] >= m]

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_books['score'] = q_books.apply(weighted_rating, axis=1, args=(m, C))

    # Sort movies based on score calculated above
    q_books = q_books.sort_values('score', ascending=False)

    # Print the top 10 books
    print(q_books[['book_id', 'title', 'vote_count', 'vote_average', 'score']].head(k))

    del metadata, metadata_books, metadata_per_age, q_books


def non_personalized_main():
    """
    Make all 3 missions (2 to 4) in part A of the exercise
    """
    # part 2)
    metadata_rating = pd.read_csv('dataset\\ratings.csv', low_memory=False, encoding="ISO-8859-1")
    create_rating_vote_count(metadata_rating)
    del metadata_rating
    K = 10
    get_simply_recommendation(K)

    # part 3)
    metadata_rating = pd.read_csv('dataset\\ratings.csv', low_memory=False, encoding="ISO-8859-1")
    metadata_user = pd.read_csv('dataset\\users.csv', low_memory=False, encoding="ISO-8859-1")
    metadata = pd.merge(metadata_rating, metadata_user, on="user_id")
    create_rating_vote_count_per_location(metadata)
    del metadata_rating, metadata_user, metadata
    K = 10
    place = "Ohio"
    get_simply_place_recommendation(place, K)

    # part 4)
    metadata_rating = pd.read_csv('dataset\\ratings.csv', low_memory=False, encoding="ISO-8859-1")
    metadata_user = pd.read_csv('dataset\\users.csv', low_memory=False, encoding="ISO-8859-1")
    metadata = pd.merge(metadata_rating, metadata_user, on="user_id")
    metadata['age'] = metadata.apply(normalized_age, axis=1)
    create_rating_vote_count_per_age(metadata)
    del metadata_rating, metadata_user, metadata
    K = 10
    age = 28
    get_simply_age_recommendation(age, K)


non_personalized_main()
