from pyvi import ViTokenizer, ViPosTagger 
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
import gensim
import os 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'NLP')

def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-8") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)

                X.append(lines)
                y.append(path)

    return X, y

def get_data_from_console(input):
    X = []
    lines = input
    lines = gensim.utils.simple_preprocess(lines)
    lines = ' '.join(lines)
    lines = ViTokenizer.tokenize(lines)
    #print(lines)

    X.append(lines)

    return X


# X_data = pickle.load(open('X_data.pkl', 'rb'))
test_path = os.path.join(dir_path, 'demo')
print('Nhap van ban vao day:')
document = input()
X_test = get_data_from_console(document)
print("-------------------------------------------------------------------------------------------------------------------")
#print(X_test)

# pickle.dump(y_test, open('y_test_demo.pkl', 'wb'))

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
# tfidf_vect_ngram.fit(X_data)

# X_data_tfidf_ngram =  tfidf_vect_ngram.transform(X_data)
with open('tfidf_vect_obj', 'rb') as obj_file:
    tfidf_vect_ngram = pickle.load(obj_file)

X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)


svd_ngram = TruncatedSVD(n_components=300, random_state=42)
# svd_ngram.fit(X_data_tfidf_ngram)

with open('svd_vect_obj', 'rb') as obj_file:
    svd_ngram = pickle.load(obj_file)
    
X_test_tfidf_ngram_svd = svd_ngram.transform(X_test_tfidf_ngram)


pickle.dump(X_test_tfidf_ngram_svd, open('X_test_tfidf_ngram_svd_demo.pkl', 'wb'))



X_test_tfidf_ngram_svd = pickle.load(open('X_test_tfidf_ngram_svd_demo.pkl', 'rb'))

md = pickle.load(open('LC_model.sav', 'rb'))
print("Linear Classifier:")
print(md.predict(X_test_tfidf_ngram_svd))
print("-------------------------------------------------------------------------------------------------------------------")

print("Naive Bayes:")
md = pickle.load(open('NB_model.sav', 'rb'))
print(md.predict(X_test_tfidf_ngram_svd))
print("-------------------------------------------------------------------------------------------------------------------")

print("Bagging Model:")
md = pickle.load(open('RDF_model.sav', 'rb'))
print(md.predict(X_test_tfidf_ngram_svd))
print("-------------------------------------------------------------------------------------------------------------------")

print("DNN:")
md = pickle.load(open('DNN_model.sav', 'rb'))
nn = md.predict(X_test_tfidf_ngram_svd).argmax(axis=-1)
if nn==0 :
    print("Chinh tri Xa hoi")
elif nn == 1:
    print("Doi song")
elif nn==2:
    print("Khoa hoc")
elif nn == 3:
    print("Kinh doanh")
elif nn == 4:
    print("Phap luat")
elif nn == 5:
    print("Suc khoe")
elif nn == 6:
    print("The gioi")
elif nn == 7:
    print("The thao")
elif nn == 8:
    print("Van hoa")
else:
    print("Vi tinh")
print("-------------------------------------------------------------------------------------------------------------------")
