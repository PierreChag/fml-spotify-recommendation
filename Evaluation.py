import pandas as pd
from statsmodels.stats.proportion import proportion_confint


# Class to calculate precision and recall
class Evaluation:
    def __init__(self, user_ids, df_songs, random_seed):
        """
        Create an instance of the Evaluation class :
        - user_ids: a data frame with a column 'user_id' that contains all the users in the dataset.
        - df_songs: the dataset used to give recommendations.
        - random_seed: the seed to create the user sample and the randomization.
        """

        self.user_ids = user_ids
        self.df_songs = df_songs
        self.seed = random_seed

    def evaluate(self, recommender, sample_size: int, evaluation_size: int = 100):
        """
        Evaluates our suggestion model on real users from the dataset.
        - recommender: function with the shape suggester(user_id, nb_of_recommendations, random_seed) that returns a DataFrame
        of songs ordered by decreasing similarity and a set of songs used to obtain these recommendations.
        - sample_size: number of user to study.
        - nb_comparisons: the number of songs to evaluate in suggester's result.
        """

        if evaluation_size < 20:
            evaluation_size = 20
        df_result = pd.DataFrame([[0, 0]] * evaluation_size, columns=['correct', 'total'])
        for studied_id in list(self.user_ids.sample(n=sample_size, random_state=self.seed)['user_id']):
            recommendations, used_songs = recommender(studied_id, evaluation_size, self.seed)
            recommendations = list(recommendations['song'])
            user_songs = set(self.df_songs[self.df_songs['user_id'] == studied_id]['song']) - used_songs

            for j in range(evaluation_size):
                df_result['total'][j] += 1
                if recommendations[j] in user_songs:
                    df_result['correct'][j] += 1
                    user_songs.remove(recommendations[j])
                    if len(user_songs) == 0:
                        break
        df_result['accuracy'] = df_result['correct'] / df_result['total']

        # Plot the evolution of the accuracy
        ax = df_result['accuracy'].plot(
            title='Evolution of the accuracy in regards of the ranking in the recommendation',
            figsize=(10, 5))
        ax.set_xlabel("Rank")
        ax.set_ylabel("Accuracy")

        # Compute accuracy over the 1, 5, 10, 20 first suggestions
        correct = df_result['correct'][0]
        total = df_result['total'][0]
        print(
            f"Accuracy of    the first suggestion  :    {100 * correct / total:.2f}% {proportion_confint(count=correct, nobs=total)}")
        correct = df_result['correct'].head(5).sum()
        total = df_result['total'].head(5).sum()
        print(
            f"Accuracy of the  5 first suggestions :    {100 * correct / total:.2f}% {proportion_confint(count=correct, nobs=total)}")
        correct = df_result['correct'].head(10).sum()
        total = df_result['total'].head(10).sum()
        print(
            f"Accuracy of the 10 first suggestions :    {100 * correct / total:.2f}% {proportion_confint(count=correct, nobs=total)}")
        correct = df_result['correct'].head(20).sum()
        total = df_result['total'].head(20).sum()
        print(
            f"Accuracy of the 20 first suggestions :    {100 * correct / total:.2f}% {proportion_confint(count=correct, nobs=total)}")
