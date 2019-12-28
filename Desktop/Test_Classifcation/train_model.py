import pickle
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


'''
y_data = pickle.load(open('y_data.pkl', 'rb'))
y_test = pickle.load(open('y_test.pkl', 'rb'))


X_data_tfidf_ngram_svd = pickle.load(open('X_data_tfidf_ngram_svd.pkl', 'rb'))
X_test_tfidf_ngram_svd = pickle.load(open('X_test_tfidf_ngram_svd.pkl', 'rb'))'''

y_data = pickle.load(open('y_data.pkl', 'rb'))
y_test = pickle.load(open('y_test_demo.pkl', 'rb'))


X_data_tfidf_ngram_svd = pickle.load(open('X_data_tfidf_ngram_svd.pkl', 'rb'))
X_test_tfidf_ngram_svd = pickle.load(open('X_test_tfidf_ngram_svd_demo.pkl', 'rb'))

from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)
encoder.classes_
def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):       
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    
    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)
        # val_predictions = classifier.predict(X_val)
        # test_predictions = classifier.predict(X_test)
        # val_predictions = val_predictions.argmax(axis=-1)
        # test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)
        # val_predictions = classifier.predict(X_val)
        # test_predictions = classifier.predict(X_test)
     
    '''print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))'''
    pickle.dump(classifier, open(filename, "wb"))
    #print(test_predictions)

print("Naive Bayes\n")
filename = "NB_model.sav"
train_model(BernoulliNB(), X_data_tfidf_ngram_svd, y_data, X_test_tfidf_ngram_svd, y_test, is_neuralnet=False)  
print("-------------------------------------------------------------------------------------------------------------------\n")  
print("Linear Classifier\n")
filename = "LC_model.sav"
train_model(LogisticRegression(), X_data_tfidf_ngram_svd, y_data, X_test_tfidf_ngram_svd, y_test, is_neuralnet=False)
print("-------------------------------------------------------------------------------------------------------------------\n")
print("Bagging Model\n")
filename = "RDF_model.sav"
train_model(RandomForestClassifier(), X_data_tfidf_ngram_svd, y_data, X_test_tfidf_ngram_svd, y_test, is_neuralnet=False)
print("-------------------------------------------------------------------------------------------------------------------\n")

def create_gru_model():
    input_layer = Input(shape=(300,))
    
    layer = Reshape((10, 30))(input_layer)
    layer = GRU(128, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    layer = Dense(128, activation='relu')(layer)
    
    output_layer = Dense(10, activation='softmax')(layer)
    
    classifier = Model(input_layer, output_layer)
    
    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

def create_dnn_model():
    input_layer = Input(shape=(300,))
    layer = Dense(1024, activation='relu')(input_layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    output_layer = Dense(10, activation='softmax')(layer)
    
    classifier =Model(input_layer, output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

classifier = create_dnn_model()
print("DNN:")
filename = "DNN_model.sav"
train_model(classifier=classifier, X_data=X_data_tfidf_ngram_svd, y_data=y_data_n, X_test=X_test_tfidf_ngram_svd, y_test=y_test_n, is_neuralnet=True, n_epochs=4)