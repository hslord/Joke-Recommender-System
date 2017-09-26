
import graphlab
import pandas as pd
import numpy as np
#%cd /Users/allen/Desktop/Galvanize/dsi-recommender-case-study
df_ratings = pd.read_table('data/ratings.dat')
df_ratings['rating'] = (df_ratings['rating']+10)**2
df_jokes = pd.read_table('data/jokes.dat')
df_ratings.head()
ratings_gl = graphlab.SFrame(df_ratings)
recommender = graphlab.recommender.factorization_recommender.create(ratings_gl,
					user_id = 'user_id', item_id = 'joke_id', target = 'rating', solver = 'als', max_iterations = 50, num_factors=9)

df = pd.read_csv('data/test_ratings.csv')
points = graphlab.SFrame({'user_id': df['user_id'].values, 'joke_id': df['joke_id'].values})
p = recommender.predict(points)
predictions = p

df['rating'] = np.array(predictions)
df.to_csv('predictions2.csv')
