from pyvi import ViTokenizer, ViPosTagger 
from tqdm import tqdm
import numpy as np
import gensim
import os 
import pickle
dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'NLP')

def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-16") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)

                X.append(lines)
                y.append(path)

    return X, y

train_path = os.path.join(dir_path, 'Train_Full')
X_data, y_data = get_data(train_path)
pickle.dump(X_data, open('X_data.pkl', 'wb'))
pickle.dump(y_data, open('y_data.pkl', 'wb'))

test_path = os.path.join(dir_path, 'Test_Full')
X_test, y_test = get_data(test_path)

pickle.dump(X_test, open('X_test.pkl', 'wb'))
pickle.dump(y_test, open('y_test.pkl', 'wb'))