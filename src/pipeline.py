from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from time import gmtime, strftime
import random


def balance_classes(df):
    certified = df[df['certified'] == 1]
    noncert = df[df['certified'] == 0]
    lencert = certified.shape[0]
    lennonc = noncert.shape[0]
    print(lencert, lennonc)
    certified = certified.sample(min(lencert, lennonc))
    noncert = noncert.sample(min(lencert, lennonc))

    return pd.concat([certified, noncert])

def train_evaluate(model_providers, data, split_criteria=None, feature_extractor=None, label=None, evaluation=None,
                   normalize=None, plots=None, sample=None, plot_enabled=True, balance=False):

    results = {}

    courses = ['all']
    if split_criteria:
        courses = data[split_criteria].unique()

    num_plot = 0

    for m, provider in enumerate(model_providers):
        model_name = provider.__class__.__name__
        results[model_name] = {}
        _models = []

        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +' Iniciando modelo '+model_name)
        for i, course_id in enumerate(courses):
            model = provider.provide()

            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +' Iniciando curso ' + course_id)

            if split_criteria:
                course = data[data[split_criteria] == course_id]
            else:
                course = data

            if sample and course.shape[0] > sample:
                course = course.sample(sample)
            elif provider.sample_size() and course.shape[0] > provider.sample_size():
                course = course.sample(provider.sample_size())

            print(course.shape)

            if feature_extractor:
                course = feature_extractor(course)

            train = course.sample(frac=0.95 if provider.require_balance() else 0.6, random_state=42)
            if provider.require_balance():
                train = balance_classes(train)

            test = course.drop(train.index)

            X, y = [], []
            if label:
                X_train, y_train = train.loc[:, train.columns != label], train[label]
                X_test, y_test = test.loc[:, test.columns != label], test[label]


            train_start = datetime.now()
            model.fit(X_train, y_train)
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +' Treinamento OK!')
            train_end = datetime.now()
            pred_test = model.predict(X_test)

            test_start = datetime.now()
            results[model_name][course_id] = evaluation(pred_test, y_test)
            test_end = datetime.now()

            results[model_name][course_id]['train_time'] = (train_end - train_start).microseconds
            results[model_name][course_id]['test_time'] = (test_end - test_start).microseconds
            results[model_name][course_id]['n_test'] = len(X_test)
            results[model_name][course_id]['n_train'] = len(X_train)

            _models.append(model)
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +' Resultados ' + str(results[model_name][course_id]))

            if plots:
                for p, plot in enumerate(plots):
                    num_plot += 1
                    if plot_enabled:
                        plt.subplot(len(courses) * len(model_providers), len(plots), num_plot)
                    plot(plt, y_test, pred_test, title= '['+model_name+'] '+course_id + " - Real x Pred")

    return results, plt, _models




def train_evaluate_by_course(model_providers, data, split_criteria=None, feature_extractor=None, label=None, evaluation=None,
                   normalize=None, plots=None, sample=None):

    grupo0 = ['2.01x',
              '3.091x',
              '6.002x',
              '7.00x',
              '8.02x',
              '8.MReV',
              'CB22x',
              'PH278x']

    grupo1 = ['14.73',
              '6.002',
              '6.00x',
              'CS50x',
              'ER22x',
              'PH207x']

    courses_train = ['CS50x', '6.00x', 'CB22x', '8.02x']

    results = {}

    courses = data[split_criteria].unique()

    num_plot = 0

    for m, provider in enumerate(model_providers):
        model_name = provider.__class__.__name__
        results[model_name] = {}
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +' Iniciando modelo '+model_name)

        data['usage'] = data.index.to_series().map(lambda id: 'train' if random.randint(0, 100) >= 50 else 'test')

        data_train = data[data['usage'] == 'train'].sample(frac=1)

        #print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Trainando para ' + courses_train[m])
        #data_train = data_train[data_train['course_id'] == courses_train[m]]

        #if courses_train[m] in grupo1:
        #    data_train = data_train.dropna(subset=['grade'])

        data_test = data[data['usage'] == 'test'].sample(frac=1)

        data_train = data_train.drop(columns=['usage'])
        data_test = data_test.drop(columns=['usage'])

        if feature_extractor:
            data_train = feature_extractor(data_train)

        if sample and data_train.shape[0] > sample:
            data_train = data_train.sample(sample)
        elif provider.sample_size() and data_train.shape[0] > provider.sample_size():
            data_train = data_train.sample(provider.sample_size())
        print('shape train', data_train.shape, 'shape test', data_test.shape)
        X_train, y_train = [], []
        if label:
            X_train, y_train = data_train.loc[:, data_train.columns != label], data_train[label]

        if normalize:
            X_train = normalize(X_train)

        model = provider.provide()
        model.fit(X_train, y_train)
        print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) + ' Treinamento OK!')

        for i, course_id in enumerate(courses):
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +' Iniciando curso ' + course_id)
            course = data_test[data_test[split_criteria] == course_id]
            course = feature_extractor(course)

            #if course_id in grupo1:
            #    course = course.dropna(subset=['grade'])

            X_test, y_test = course.loc[:, course.columns != label], course[label]
            pred_test = model.predict(X_test)
            results[model_name][course_id] = evaluation(pred_test, y_test )
            #print(strftime("%Y-%m-%d %H:%M:%S", gmtime()) +' Resultados ' + str(results[model_name][course_id]+'\n'))

            if plots:
                for p, plot in enumerate(plots):
                    num_plot += 1
                    #plt.subplot(len(courses) * len(model_providers), len(plots), num_plot)
                    plot(plt, y_test, pred_test, title= '['+model_name+'] '+course_id + " - Real x Pred")

    return results, plt



def log(context, msg):
    print(strftime("["+context+"] %Y-%m-%d %H:%M:%S", gmtime()) + msg)


def evaluate_models_by_feature(model_providers, data, split_criteria=None, feature_extractor=None, label=None,
                               evaluation=None, plots=None):

    results = {}

    courses = data[split_criteria].unique()

    num_plot = 0

    for m, provider in enumerate(model_providers):
        model_name = provider.__class__.__name__
        results[model_name] = {}
        log(model_name, 'Preparando treinamento...')

        data['usage'] = data.index.to_series().map(lambda id: 'train' if random.randint(0, 100) > 40 else 'test')
        data_train = data[data['usage'] == 'train'].sample(frac=1)
        data_test = data[data['usage'] == 'test'].sample(frac=1)

        data_train = data_train.drop(columns=['usage'])
        data_test = data_test.drop(columns=['usage'])

        if feature_extractor:
            data_train = feature_extractor(data_train)

        if provider.sample_size() and data.shape[0] > provider.sample_size():
            data_train = data_train.sample(provider.sample_size())

        print('shape', data_train.shape)
        X_train, y_train = [], []
        if label:
            X_train, y_train = data_train.loc[:, data_train.columns != label], data_train[label]

        model = provider.provide()

        log(model_name, 'Iniciando treinamento...')
        model.fit(X_train, y_train)

        log(model_name, 'Treinamento finalizado!')

        for i, course_id in enumerate(courses):
            log(model_name+'/'+course_id, 'Preparando avaliação')
            course = data_test[data_test[split_criteria] == course_id]
            course = feature_extractor(course)
            X_test, y_test = course.loc[:, course.columns != label], course[label]

            log(model_name + '/' + course_id, 'avaliando...')
            pred_test = model.predict(X_test)
            results[model_name][course_id] = evaluation(pred_test, y_test )
            log(model_name+'/'+course_id, 'Finalizando.')

            log(model_name + '/' + course_id, 'Gerando gráficos.')
            if plots:
                for p, plot in enumerate(plots):
                    num_plot += 1
                    plt.subplot(len(courses) * len(model_providers), len(plots), num_plot)
                    plot(plt, y_test, pred_test, title= '['+model_name+'] '+course_id + " - Real x Pred")

            log(model_name + '/' + course_id, 'Avaliação finalizada.')

    return results, plt
