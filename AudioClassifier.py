import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import multiprocessing
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import os
import datetime

class AudioClassifier:
    def __init__(self, epochs=100, batch_size=32):
        self.epochs=epochs
        self.batch_size=batch_size
        self.ctime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def labelMapper(self, label):
        return label.replace({
            'Bill_Gates_Harvard': 'Bill Gates',
            'Mark_Zuckerberg_Harvard': 'Mark Zuckerberg',
            'Oprah_Winfrey_Harvard': 'Oprah Winfrey',
            'Sheryl_Sandberg_Addresses': 'Sheryl Sandberg',
            'J_K__Rowling_Speaks': 'JK Rowling',
            'Your_elusive_creative': 'Elizabeth Gilberts',
            'Raw_Video:_Barack': 'Barack Obama',
            'Ellen_at_Tulane': 'Ellen DeGeneres',
            'Malala_Yousafzai_UN': 'Malala Yousafzai',
            "Steve_Jobs'_2005": 'Steves Jobs',
            'Gloria_Steinem_discusses': 'Gloria Steinem',
            "Hillary_Clinton's_keynote": 'Hillary Clinton',
            'Jim_Carrey_at': 'Jim Carrey',
            "Penn's_2011_Commencement": 'Penn Jillette',
            'EKU_Class_of': 'Ellen Kuras',
        })
    
    def label_encoder(self, label):
        return label.replace({
            'Bill Gates': 0,
            'Mark Zuckerberg': 1,
            'Oprah Winfrey': 2,
            'Sheryl Sandberg': 3,
            'JK Rowling': 4,
            'Elizabeth Gilberts': 5,
            'Barack Obama': 6,
            'Ellen DeGeneres': 7,
            'Malala Yousafzai': 8,
            'Steves Jobs': 9,
            'Gloria Steinem': 10,
            'Hillary Clinton': 11,
            'Jim Carrey': 12,
            'Penn Jillette': 13,
            'Ellen Kuras': 14,
        })

    def label_decoder(self, label):
        lst = ["Bill Gates", "Mark Zuckerberg", "Oprah Winfrey", "Sheryl Sandberg", "JK Rowling", "Elizabeth Gilberts", "Barack Obama", "Ellen DeGeneres", "Malala Yousafzai", "Steves Jobs", "Gloria Steinem", "Hillary Clinton", "Jim Carrey", "Penn Jillette", "Ellen Kuras"]
        return lst[list(label)[0]]

    def split_dataset(self, x, y, test_size=0.2, random_state=42):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
        return x_train, x_test, y_train, y_test

    def create_model(self, optimizer='adam',activation='relu', dropout_rate=0.2, neurons=100):
        model = Sequential()
        model.add(Dense(neurons, input_dim=40, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(self.noutput, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def grid_search(self, x_train, y_train):
        self.noutput = len(np.unique(y_train))
        model = KerasClassifier(build_fn=self.create_model, verbose=1, epochs=self.epochs, batch_size=self.batch_size)

        # define the grid search parameters
        params = {
            'optimizer': ['adam', 'rmsprop'],
            'activation': ['relu', 'tanh'],
            'dropout_rate': [0.2, 0.3],
            'neurons': [100, 200]
        }

        grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=multiprocessing.cpu_count(), cv=3)
        grid_result = grid.fit(x_train, y_train)
        
        # summarize results
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        return grid_result

    def train_best_fit(self, grid_result, x_train, y_train):
        # save the model
        model = self.create_model(**grid_result.best_params_)
        model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size)
        return model 

    def evaluate_model(self, model, x_test, y_test):
        # evaluate the model
        scores = model.evaluate(x_test, y_test)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        return scores

    def predict_label(self, model, x_test):
        # predict the label
        y_pred = model.predict(x_test)
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def save_model(self, model):
        # save the model
        os.makedirs('models', exist_ok=True)
        model.save(f'models/model_{self.ctime}.h5')
        print('Model saved')

    def load_model(self, model_path):
        # load the model
        model = keras.models.load_model(model_path)
        return model

    def save_logs(self, grid_result):
        # save the logs
        df = pd.DataFrame(grid_result.cv_results_)
        os.makedirs('logs', exist_ok=True)
        df.to_csv(f'logs/logs_{self.ctime}.csv', index=False)
        print('Logs saved')

