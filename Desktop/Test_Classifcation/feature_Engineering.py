import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras import optimizers



X_data = pickle.load(open('X_data.pkl', 'rb'))
y_data = pickle.load(open('y_data.pkl', 'rb'))

X_test = pickle.load(open('X_test.pkl', 'rb'))
y_test = pickle.load(open('y_test.pkl', 'rb'))

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
tfidf_vect_ngram.fit(X_data)

X_data_tfidf_ngram =  tfidf_vect_ngram.transform(X_data)

X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

svd_ngram = TruncatedSVD(n_components=300, random_state=42)
svd_ngram.fit(X_data_tfidf_ngram)

X_data_tfidf_ngram_svd = svd_ngram.transform(X_data_tfidf_ngram)
X_test_tfidf_ngram_svd = svd_ngram.transform(X_test_tfidf_ngram)

pickle.dump(X_data_tfidf_ngram_svd, open('X_data_tfidf_ngram_svd.pkl', 'wb'))
pickle.dump(X_test_tfidf_ngram_svd, open('X_test_tfidf_ngram_svd.pkl', 'wb'))
