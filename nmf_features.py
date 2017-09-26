import graphlab
import pandas as pd
import numpy as np
from collections import defaultdict
#%cd /Users/allen/Desktop/Galvanize/dsi-recommender-case-study
df_ratings = pd.read_table('data/ratings.dat')
df_ratings['rating'] = (df_ratings['rating']+10)**2
df_jokes = pd.read_table('data/jokes.dat')
df_jokes
ratings_gl = graphlab.SFrame(df_ratings)
recommender = graphlab.recommender.factorization_recommender.create(ratings_gl,
					user_id = 'user_id', item_id = 'joke_id', target = 'rating', solver = 'als', max_iterations = 50, num_factors=9, nmf = True)
jokes = recommender['coefficients']['joke_id']
categories = defaultdict(list)
np.argmax(jokes[jokes['joke_id']==1]['factors'])
for i in jokes['joke_id']:
	categories[np.argmax(jokes[jokes['joke_id']==i]['factors'])].append(i)
categories[0]
