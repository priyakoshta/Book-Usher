import pickle
import config

def recommend():
    filename = 'model/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    us_canada_user_rating_pivot = pickle.load(open('resources/user_rating_pivot.csv','rb'))

    us_canada_book_title = us_canada_user_rating_pivot.index
    us_canada_book_list = list(us_canada_book_title)
    query_index = us_canada_book_list.index(config.bookName)
    distances, indices = loaded_model.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1),
                                                 n_neighbors=6)

    for i in range(0, len(distances.flatten())):
        if i == 0:
           print('Recommendations for {0}:\n'.format(us_canada_user_rating_pivot.index[query_index]))
        else:
            print('{0}: {1}, with distance of {2}:'.format(i, us_canada_user_rating_pivot.index[indices.flatten()[i]],
                                                      distances.flatten()[i]))


def main():
    recommend()

if __name__ == '__main__':
    main()


