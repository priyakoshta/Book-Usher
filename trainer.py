from scipy.sparse import csr_matrix
import pickle
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def main():

    books = pd.read_csv('resources/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM','imageUrlL']
    users = pd.read_csv('resources/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
    users.columns = ['userID', 'Location', 'Age']
    ratings = pd.read_csv('resources/BX-Book-Ratings.csv', sep=';',error_bad_lines=False, encoding="latin-1")
    ratings.columns = ['userID', 'ISBN', 'bookRating']

    combine_book_rating = pd.merge(ratings, books, on='ISBN')
    columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    combine_book_rating.head()

    combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])
    book_ratingCount = (combine_book_rating.
        groupby(by=['bookTitle'])['bookRating'].
        count().
        reset_index().
        rename(columns={'bookRating': 'totalRatingCount'})
    [['bookTitle', 'totalRatingCount']]
        )
    book_ratingCount.head()

    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='bookTitle',
                                                             right_on='bookTitle', how='left')
    rating_with_totalRatingCount.head()

    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    print(book_ratingCount['totalRatingCount'].describe())
    print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

    popularity_threshold = 50
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    rating_popular_book.head()

    combined = rating_popular_book.merge(users, left_on='userID', right_on='userID', how='left')

    us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
    us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)
    us_canada_user_rating.head()

    if not us_canada_user_rating[us_canada_user_rating.duplicated(['userID', 'bookTitle'])].empty:
        initial_rows = us_canada_user_rating.shape[0]

        print('Initial dataframe shape {0}'.format(us_canada_user_rating.shape))
        us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
        current_rows = us_canada_user_rating.shape[0]
        print('New dataframe shape {0}'.format(us_canada_user_rating.shape))
        print('Removed {0} rows'.format(initial_rows - current_rows))


    us_canada_user_rating_pivot = us_canada_user_rating.pivot(index='bookTitle', columns='userID',
                                                              values='bookRating').fillna(0)

    pickle.dump(us_canada_user_rating_pivot, open('resources/user_rating_pivot.csv', 'wb'))

    us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(us_canada_user_rating_matrix)
    filename = 'model/finalized_model.sav'
    pickle.dump(model_knn, open(filename, 'wb'))

if __name__ == '__main__':
    main()
