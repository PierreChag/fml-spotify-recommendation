# Machine Learning Project : Music Recommendation System using Spotify Database

### Authors :
* Elie Trigano
* Zachary Guenassia
* Thomas Taylor
* Pierre Chagnon


### Introduction :

We live in a time when the popularity of digital content streaming services continues to grow. This trend is even more evident in musical streaming. Major providers such as Spotify, Apple Music or SoundCloud claim to have tens of millions of active users and these numbers are constantly growing. Another advantage is the amount of content that is easily accessible. Commercial music libraries easily exceed 15 million songs, far exceeding the listening capacity of a single person. Hence, it is very difficult for users to find relevant content in such an inexhaustible amount of material. This problem is mainly solved by referral systems that only suggest content that users may find useful.

*Objective*: Use Spotify’s Data to build a music recommendation system using Collaborative and Content-based filtering and classifications algorithms. 


### Motivation and Problem Definition :

* What is the problem you are trying to solve? 

The problem that we are trying to solve is to recommend songs to Spotify users based on the songs they like. Indeed, the outcome of our project would be a recommendation system based on a song dataset scrapped from Spotify’s API. 
Spotify already classifies songs with different features such as acousticness, danceability, tempo and others. 
The business problem to be solved in this project would be to offer new content to Spotify users based on their musical tastes. Furthermore, it would allow Spotify to push songs that are likely to be saved in a user’s playlist, thus extending the lifetime of the customer and increasing the use of Spotify. 

* Why is the problem important?

We listen to a lot of music, most of the group on Spotify. Music surrounds us and is a pillar of the culture. It is a very interesting topic to bring machine learning algorithms and processes. As students, it is more motivating to implement academic knowledge that could be useful for personal and professional life. Applying such a key concept as a recommendation system applied to one of the biggest tech platforms is a unique opportunity. Also, we could replicate the learnings of this project to other datasets/industries in the future as recommendation systems are an important part of data science professional implementation. 

As said earlier, it is also an important topic for companies such as Spotify as a recommendation system extends the LTV of a customer for instance. The more data Spotify gets on the likeability of songs from their user base, the better recommendations they can give, and the longer the customer will stay on Spotify.

* What are a few potential applications?

Movie recommendation system, books, computers, items, food, brands, etc. 
Recommendation systems are applicable to any industry, company and service. The most famous example can be Netflix’s recommendation system, but every company uses a form of it like YouTube, Tinder, Amazon, Carrefour…
 

### Methodology :

* How do you plan to address the problem? 

To build our recommendation system, we will need a few steps:
Creating our dataset of n random songs through Spotify API 
EDA, feature selection/engineering, data cleaning…
Create fake but coherent users profiles with probability models to fill the target column
Split the dataset into train and test
Create different models to predict rather a user will like a song
Model evaluation (accuracy, precision and recall)

Our models will probably consist of: content-based recommendation, collective filtering, and some machine learning classification algorithms such as decision trees, random forests, or SVMs.

### Evaluation : 

* How will you evaluate your work?

To evaluate our work, we will have to compute the accuracy of our model. The more frequently our models suggest songs that are the training dataset of a user, the higher the accuracy will be. However, it is possible that our models are correct without their accuracy being very high since the songs suggested to the user may suit his taste but simply be unknown to him.

* What experiments do you plan to do? 

We will have to train models for different users and compare the recommended songs to the ones in their respective test sets. At first, we will train and test our models on homogeneous datasets (for example with mainly 1 style of music) until we are able to reach sufficient accuracy. Then we would use a dataset closer to reality, with diverse styles of music. And lastly, we could even test our models on our own playlists.

* What dataset will be used (existing datasets, or are you planning to create new ones)?

We will either use the Kaggle dataset or scrap the Spotify dataset using Spotify API to generate the main dataset of songs from which our models will pick songs to suggest.
We also need a dataset for each user to train our models. These datasets will correspond to their playlist. At first we will use playlists that contain songs in a similar style, then we will use more diverse playlists.

Since the project is still at an early stage, many things could change depending on the difficulties we will meet during its realization.

### References : 

[1] "A Survey of Music Recommendation Systems and Future Perspectives" -  June 2012, London, UK -  Conference: The 9th International Symposium on Computer Music Modeling and Retrieval - 
https://www.researchgate.net/publication/277714802_A_Survey_of_Music_Recommendation_Systems_and_Future_Perspectives?utm_source=twitter&rgutm_meta1=eHNsLUF4TnZlVjVQbnppbHlIODhTNm83YnNoUWswc0tmc0RSTndaSm9pTGk5RGh3bGVIY2tibTRRU3g5ZlZhcjZwTjM5cklCMVhuR2Q5Yi9WRHgzK1JyMkZaUjg%3D 

[2] “An Automated Music Recommendation System Based on Listener Preferences” - 2021 - Mukkamala. S.N.V. Jitendra -
https://www.sciencegate.app/document/10.3233/apc210182

[3] "How to build a simple song recommender system" -  April 2017 - Eric Le
https://towardsdatascience.com/how-to-build-a-simple-song-recommender-296fcbc8c85

[4] "Building a Music Recommendation Engine " - October 2021 - Tanmoy Ghosh
https://www.section.io/engineering-education/building-spotify-recommendation-engine/#generating-recommendations

[5] "Building A Music Recommendation System Like Spotify | Music Recommender System"  - March 2022 - Intellipaat Data Science & AI Course
https://www.youtube.com/watch?v=FV3IvHeuH_k

[6] "All You Need to Know About a Music Recommendation System with a Step-By-Step Guide to Creating It"  - February 2021 - Serhii Ripenko, Nadiia Hasiuk
https://www.eliftech.com/insights/all-you-need-to-know-about-a-music-recommendation-system-with-a-step-by-step-guide-to-creating-it/
