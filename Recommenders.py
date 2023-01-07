# This module contains the Recommenders class and all the functions that are used to build the recommendation system.
import numpy as np
import pandas as pd


# Class for Popularity based Recommender System model
class PopularityRecommender:
    def __init__(self, train_data):
        self.train_data = train_data
        # Get a count of user_ids for each unique song as recommendation score
        train_data_grouped = train_data.groupby(['song']).agg({'user_id': 'count'}).reset_index()
        train_data_grouped.rename(columns={'user_id': 'score'}, inplace=True)
        # Sort the songs based upon recommendation score
        self.popularity_recommendations = train_data_grouped.sort_values(['score', 'song'], ascending=[0, 1])
        # Generate a recommendation rank based upon score
        self.popularity_recommendations['rank_pop'] = self.popularity_recommendations['score'].rank(ascending=0, method='first')

    def recommend(self, user_id, nb_of_recommendations):
        """
        Use the popularity based recommender system model to make recommendations.
        Returns a DataFrame containing the recommendations ordered from the most likely to the less likely,
        and returns a set that contains the songs used to obtain the recommendation.
        """
        if nb_of_recommendations is None:
            return self.popularity_recommendations
        else:
            return self.popularity_recommendations.head(nb_of_recommendations)


# Class for Item similarity based Recommender System model
class ItemSimilarityRecommender:
    def __init__(self, train_data):
        self.train_data = train_data
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None

    def get_user_songs(self, user):
        """
        Return a pandas DataFrame that contains the unique items (songs) corresponding to a given user
        """

        user_data = self.train_data[self.train_data['user_id'] == user]
        user_items = user_data[['song']].drop_duplicates()
        return user_items

    def get_song_users(self, item):
        """
        Get unique users for a given item (song)
        """

        item_data = self.train_data[self.train_data['song'] == item]
        item_users = set(item_data['user_id'].unique())
        return item_users

    def get_all_unique_songs(self):
        """
        Get unique items (songs) in the training data
        """

        return list(self.train_data['song'].unique())

    def construct_cooccurence_matrix(self, user_songs, all_songs):
        """
        Construct cooccurence matrix
        """

        # Get users for all songs in user_songs.
        user_songs_users = []
        for song in user_songs:
            user_songs_users.append(self.get_song_users(song))

        # Initialize the item cooccurence matrix of size : len(user_songs) X len(songs)
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)

        # Calculate similarity between user songs and all unique songs in the training data
        for i, song in enumerate(all_songs):
            # Calculate unique listeners (users) of song i
            song_i_data = self.train_data[self.train_data['song'] == song]
            users_i = set(song_i_data['user_id'].unique())

            for j in range(len(user_songs)):
                # Get unique listeners (users) of song (item) j
                users_j = user_songs_users[j]
                # Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)

                # Calculate cooccurence_matrix[i,j] as Jaccard Index
                if len(users_intersection) != 0:
                    # Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    cooccurence_matrix[j, i] = float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        return cooccurence_matrix

    def generate_top_recommendations(self, all_songs, user_songs, nb_of_recommendations):
        """
        Use the cooccurence matrix to make top recommendations
        """

        # Construct item cooccurence matrix of size : len(user_songs) X len(songs)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        # Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0) / float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        # Sort the indices of user_sim_scores based upon their value
        # Also maintain the corresponding score
        sort_index = sorted(((e, i) for i, e in enumerate(list(user_sim_scores))), reverse=True)

        # Create a dataframe from the following
        df = pd.DataFrame(columns=['song', 'score', 'rank_sim'])

        # Fill the dataframe with top nb_of_recommendations item based recommendations
        rank = 1
        for i in range(0, len(sort_index)):
            if nb_of_recommendations is None or rank <= nb_of_recommendations:
                if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs:
                    df.loc[len(df)] = [all_songs[sort_index[i][1]], sort_index[i][0], rank]
                    rank += 1
            else:
                break

        # Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df

    def recommend(self, user_id, nb_of_recommendations):
        """
        Use the item similarity based recommender system model to make recommendations
        Returns a DataFrame containing the recommendations ordered from the most likely to the less likely,
        and returns a set that contains the songs used to obtain the recommendation
        """

        user_songs = set(self.get_user_songs(user_id)['song'])
        df_rec = self.generate_top_recommendations(self.get_all_unique_songs(), list(user_songs), nb_of_recommendations)
        if nb_of_recommendations is None:
            return df_rec
        else:
            return df_rec.iloc[:nb_of_recommendations, :]


# Class for a Recommender System using the play count as ratings
class PlayCountRecommender:
    def __init__(self, train_data):
        # Create a smaller version of the dataset for testing purpose.
        # Create a matrix of users and play counts
        self.train_data = train_data
        self.user_play_count = self.train_data.pivot(index='user_id', columns='song', values='play_count').fillna(0)
        self.user_play_count_matrix = self.user_play_count.values
        # Number of users and items
        self.num_items = self.user_play_count_matrix.shape[1]
        # Compute the similarity matrix
        self.similarity = self.cosine_matrix()
        # Finding the index of the user
        self.tup_user = []
        i = 0
        for user in self.user_play_count.index:
            self.tup_user.append((user, i))
            i += 1
        self.tup_song = []
        i = 0
        for song in self.user_play_count.columns:
            self.tup_song.append((song, i))
            i += 1

    def get_user_songs(self, user):
        """
        Return a pandas DataFrame that contains the unique items (songs) corresponding to a given user
        """

        user_items = self.train_data[self.train_data['user_id'] == user]
        user_items = user_items.sort_values(by='play_count', ascending=False)
        return user_items

    def cosine_matrix(self):
        """
        Calculate the cosine similarity matrix for the songs
        """

        similarity = np.zeros((self.num_items, self.num_items))
        for i in range(self.num_items):
            for j in range(self.num_items):
                # Calculate dot product of item vectors
                dot_product = np.dot(self.user_play_count_matrix[:, i], self.user_play_count_matrix[:, j])
                # Calculate magnitudes of item vectors
                magnitude_i = np.sqrt(np.dot(self.user_play_count_matrix[:, i], self.user_play_count_matrix[:, i]))
                magnitude_j = np.sqrt(np.dot(self.user_play_count_matrix[:, j], self.user_play_count_matrix[:, j]))
                # Calculate cosine similarity
                similarity[i, j] = dot_product / (magnitude_i * magnitude_j)
        return similarity

    def recommend(self, user_id, nb_of_recommendations):
        """
        Use the play count based recommender system model to make recommendations
        Returns a DataFrame containing the recommendations ordered from the most likely to the less likely,
        and returns a set that contains the songs used to obtain the recommendation
        """

        # Finding the index of the song in the list of tuples
        user_index = 0
        for user in self.tup_user:
            if user[0] == user_id:
                user_index = user[1]
                break

        # Make recommendations for user 0
        user_ratings = self.user_play_count_matrix[user_index, :]
        item_scores = np.zeros(self.num_items)
        for i in range(self.num_items):
            if user_ratings[i] == 0:
                # Calculate score for item i
                score = 0
                for j in range(self.num_items):
                    if user_ratings[j] != 0:
                        score += self.similarity[i, j] * user_ratings[j]
                item_scores[i] = score

        # Sort item scores in descending order
        sorted_item_scores = np.argsort(-item_scores)

        # Return the recommendations in a dataframe
        user_songs = set(self.get_user_songs(user_id)['song'])
        df_rec = pd.DataFrame(columns=['song', 'score', 'rank_play_count'])
        rank = 1
        if nb_of_recommendations is None:
            for i in range(self.num_items - len(user_songs)):
                rec_song = self.tup_song[sorted_item_scores[i]][0]
                df_rec.loc[i] = [rec_song, item_scores[sorted_item_scores[i]], rank]
                rank += 1
            return df_rec
        else:
            for i in range(nb_of_recommendations):
                rec_song = self.tup_song[sorted_item_scores[i]][0]
                df_rec.loc[i] = [rec_song, item_scores[sorted_item_scores[i]], rank]
                rank += 1
            return df_rec


def generate_reco_dataset(full_df, train_data, user_ids, reco_pop, reco_sim, reco_play, sample_size, seed):
    """
    Very long to run !
    Compute the recommendation score for sample_size number of users using the 3 previous models.
    Return the results combined in a big DataFrame.
    """
    full = []
    for i, studied_id in enumerate(list(user_ids.sample(n=sample_size, random_state=seed)['user_id'])):
        all_scores1 = reco_pop.recommend(studied_id, None)
        all_scores2 = reco_sim.recommend(studied_id, None)
        all_scores3 = reco_play.recommend(studied_id, None)
        train_songs = set(train_data[train_data['user_id'] == studied_id].sort_values(by='play_count', ascending=False)['song'])

        test_songs = set(full_df[full_df['user_id'] == studied_id]['song']) - train_songs

        # Normalize the scores between 0 and 1
        all_scores1['score_pop'] = (all_scores1['score'] - all_scores1['score'].min()) / (
                    all_scores1['score'].max() - all_scores1['score'].min())
        all_scores2['score_sim'] = (all_scores2['score'] - all_scores2['score'].min()) / (
                    all_scores2['score'].max() - all_scores2['score'].min())
        all_scores3['score_play_count'] = (all_scores3['score'] - all_scores3['score'].min()) / (
                    all_scores3['score'].max() - all_scores3['score'].min())

        # Merge in one dataframe
        all_scores = pd.merge(all_scores1, all_scores2, on='song', how='outer')
        all_scores = pd.merge(all_scores, all_scores3, on='song', how='outer')
        all_scores = all_scores.fillna(0)

        # Add column correction
        all_scores['correct'] = 0
        for song in test_songs:
            all_scores.loc[all_scores['song'] == song, ['correct']] = 1
        all_scores.drop(columns=['score', 'score_x', 'score_y', 'rank_pop', 'rank_sim', 'rank_play_count'], axis=1,
                        inplace=True)
        all_scores['user_id'] = studied_id
        full += all_scores.values.tolist()
        print(f"[{i + 1}/{sample_size}]")

    # Remove useless column
    full = pd.DataFrame(full, columns=['song', 'score_pop', 'score_sim', 'score_play_count', 'correct', 'user_id'])
    full['score_tot'] = full['score_pop'] + full['score_sim'] + full['score_play_count']
    full.drop(full[full['score_tot'] == 0].index, inplace=True)
    full.drop(columns=['score_tot'], axis=1, inplace=True)
    return full


# Class of a recommender that combines the 3 previous models.
class MixedRecommenders:
    def __init__(self, model, df_test):
        self.model = model
        self.df_test = df_test

    def recommend(self, user_id, nb_of_recommendations):
        user_reco = self.df_test[self.df_test['user_id'] == user_id]
        if user_reco.empty:
            print(f"The user {user_id} is not included in the precalculated dataset.")
            return None

        predicted = self.model.predict(user_reco[['score_pop', 'score_sim', 'score_play_count']])
        # Creates a dataframe that combines all the information.
        res = pd.concat([user_reco['song'].reset_index(drop=True), pd.DataFrame(predicted, columns=['predicted'])],
                        axis=1, join='outer', ignore_index=False, sort=False)
        res = res.sort_values(['predicted', 'song'], ascending=[0, 1])
        res['rank_mixed'] = res['predicted'].rank(ascending=0, method='first')
        if nb_of_recommendations is None:
            return res
        else:
            return res.head(nb_of_recommendations)


class ManualModel:
    def __init__(self, weight):
        self.weight = weight

    def predict(self, features):
        return (features['score_pop'] * self.weight[0] + features['score_sim'] * self.weight[1] + features['score_play_count'] * self.weight[2]).to_numpy()


# Class for a Recommender System using SVD with surprise
# from surprise import Reader, Dataset, SVD
# from surprise.model_selection import cross_validate
# from recommenders.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k,
#                                                        precision_at_k,
#                                                        recall_at_k, get_top_k_items)
# from recommenders.models.surprise.surprise_utils import predict, compute_ranking_predictions
# Class for a Recommender System using SVD with surprise
class SurpriseRecommender:
    def __init__(self, df_triplets):
        self.df_triplets = df_triplets
        self.reader = None
        self.data = None
        self.trainset = None
        self.testset = None
        self.algo = None

    def get_songs(self, user_id):
        """ Get all the songs listened to by a user

        Args:
            user_id (_str_): the user_id for which we want to get the songs
        """
        return self.df_triplets[self.df_triplets.user_id == user_id].sort_values(by='play_count',
                                                                                 ascending=False) \
            .song.unique()

    def get_data(self):
        """ Create a surprise dataset
        """
        # self.reader = Reader(rating_scale=(0, self.df_triplets.play_count.max()))
        # self.data = Dataset.load_from_df(self.df_triplets[['user_id', 'song', 'play_count']], self.reader)
        return self.data

    def get_trainset(self):
        """ Create a surprise trainset
        """
        self.trainset = self.data.build_full_trainset()
        return self.trainset

    def get_recommendations(self, user_id, k=10):
        """ Get the top k recommendations for a user based on the number of times a song has been played

        Args:
            user_id (_type_): the user_id for which we want to make recommendations
            k (int, optional): number of recommendations - Defaults to 10.
        """

        self.get_data()
        songs = self.get_songs(user_id)
        # self.algo = SVD(n_epochs=30, n_factors=100)
        self.algo.fit(self.get_trainset())

        # Predict the play count for a user and song
        def predict_play_count(user_id, song):
            play_count = self.algo.predict(user_id, song).est
            return play_count

        # Create a function to return the top 10 recommendations for a user
        def generate_reco(user_id):
            # Generate the top 10 recommendations for the user
            top_10 = []
            for song in self.df_triplets.song.unique():
                if song in songs:
                    continue
                else:
                    top_10.append((song, predict_play_count(user_id, song)))
            top_10.sort(key=lambda x: x[1], reverse=True)
            top_10 = top_10[:10]
            # top_10 to dataframe
            top_10 = pd.DataFrame(top_10, columns=['song', 'estimated_play_count'])

            return top_10

        top_10 = generate_reco(user_id)

        # Print the top 10 recommendations for a user
        print('User ID: ', user_id)
        print('Songs listened to: ')
        print("------------------------------------------------------------------------------------")
        for song in songs:
            print(song)
        print("------------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------------")

        print('Top 10 recommendations: ')
        return top_10
