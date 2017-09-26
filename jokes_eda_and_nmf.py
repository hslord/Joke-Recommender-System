#start EDA
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF, PCA
'''Read Files'''
jokes = pd.read_csv('/Users/hslord/galvanize/dsi-recommender-case-study/data/jokes.csv')


 #joke 151 seems to be missing, but rated 12.5k times
ratings = pd.read_table('/Users/hslord/galvanize/dsi-recommender-case-study/data/ratings.dat')

'''EDA'''
ratings.describe()
ratings.groupby('joke_id').count()

avg_user_rating = ratings[['user_id', 'rating']].groupby('user_id').mean().sort_values('rating')

max_user_rating = ratings[['user_id', 'rating']].groupby('user_id').max().sort_values('rating')

min_user_rating = ratings[['user_id', 'rating']].groupby('user_id').min().sort_values('rating')

joke_151 = ratings[ratings['joke_id'] == 151].sort_values('rating')

'''TFIDF of Jokes'''
jokes_arr = jokes.values[:,1]
lower_joke = []
for x in xrange(jokes_arr.shape[0]):
    lowers = jokes_arr[x].lower()
    no_punctuation = lowers.translate(None, string.punctuation)
    lower_joke.append(no_punctuation)

wordnet = WordNetLemmatizer()

def lem(arr):
    lemmatized = []
    for x in xrange(arr.shape[0]):
        tokenize_row = word_tokenize(arr[x])
        lem = [wordnet.lemmatize(word) for word in tokenize_row]
        lemmatized.append(' '.join(lem))
    return np.array(lemmatized)

jokes_lem = lem(np.array(lower_joke))

tfidfvect = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_vectorized = tfidfvect.fit_transform(jokes_lem)
tfidf_fit = tfidfvect.fit(jokes_lem)
words = tfidf_fit.get_feature_names()

'''NMF on TFIDF of Jokes'''
nmf_jokes = NMF(n_components=10, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf_vectorized)
nmf_data = nmf_jokes.transform(tfidf_vectorized)
joke_id = jokes['joke_id'].reshape(-1,1)
nmf_w_joke_id = np.concatenate((joke_id, nmf_data), axis=1)
nmf_df = pd.DataFrame(data=nmf_w_joke_id, columns=['joke_id', 'topic_0', 'topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5', 'topic_6', 'topic_7', 'topic_8', 'topic_9'])

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

print_top_words(nmf_jokes, words, 15)
