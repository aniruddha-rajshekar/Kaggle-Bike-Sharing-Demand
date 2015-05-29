import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

def loadData(datafile):
    return pd.read_csv(datafile)

def splitDatetime(data):
    sub = pd.DataFrame(data.datetime.str.split(' ').tolist(), columns = "date time".split())
    date = pd.DataFrame(sub.date.str.split('-').tolist(), columns="year month day".split())
    time = pd.DataFrame(sub.time.str.split(':').tolist(), columns = "hour minute second".split())
    data['year'] = date['year']
    data['month'] = date['month']
    data['day'] = date['day']
    data['hour'] = time['hour'].astype(int)
    return data

def normalizedata(train, test, features):
    #norm = preprocessing.Normalizer()

    test_tr = train.as_matrix()#train.reset_index().values
    test_tr = test_tr.astype(np.int64)

    # print test_tr.shape
    #
    # print test_tr.mean(axis=0).shape

    test_tr = (test_tr - test_tr.mean(axis=0))/test_tr.std(axis=0)

    train_train = pd.DataFrame(test_tr, columns=features)


    test_ts = test.as_matrix()#train.reset_index().values
    test_ts = test_ts.astype(np.int64)

    test_ts = (test_ts - test_ts.mean(axis=0))/test_ts.std(axis=0)

    test_test = pd.DataFrame(test_ts, columns=features)


    #test = (test - tr_mean)/tr_std#norm.transform(test)
    return train_train, test_test

def createRandomForest():
    est = RandomForestRegressor(n_estimators=1000)
    return est

def createExtraTree():
    est = ExtraTreesRegressor(n_estimators=1200)
    return est


def predict(est, train, test, features, target, filename):

    f = open(filename, 'wb')
    #with open(outPath + filename, 'wb') as f:
    f.write("datetime,count\n")

    for index, value in enumerate(list(est.predict(test[features]))):
        f.write("%s,%s\n" % (test['datetime'].loc[index], int(value)))


def main():

    train = loadData("train.csv")
    test = loadData("test.csv")


    train = splitDatetime(train)
    test = splitDatetime(test)

    features = [col for col in train.columns if col not in ['datetime', 'casual', 'registered', 'count']]


    #Extra Tree Regressor (Casual)
    target = 'casual'
    train1 = train
    test1 = test


    est = createExtraTree()
    train1[features], test1[features] = normalizedata(train1[features], test1[features], features)

    print 'Train'
    est.fit(train1[features], train1[target])
    print 'Predict'
    predict(est, train1, test1, features, target, filename='submissions-ET-casual.csv')
    for feat in features:
    	print feat
    print est.feature_importances_


    #Extra Tree Regressor (Registered)
    target = 'registered'
    train2 = train
    test2 = test

    est = createExtraTree()
    train2[features], test2[features] = normalizedata(train2[features], test2[features], features)

    print 'Train'
    est.fit(train2[features], train2[target])
    print 'Predict'
    predict(est, train2, test2, features, target, filename='submissions-ET-registered.csv')
    for feat in features:
    	print feat
    print est.feature_importances_

    #Random Forest (Casual)
    target = 'casual'
    train3 = train
    test3 = test

    est = createRandomForest()
    train3[features], test3[features] = normalizedata(train3[features], test3[features], features)

    print 'Train'
    est.fit(train3[features], train3[target])
    print 'Predict'
    predict(est, train3, test3, features, target, filename='submissions-RF-casual.csv')
    for feat in features:
    	print feat
    print est.feature_importances_


    #Random Forest (Registered)
    target = 'registered'
    train4 = train
    test4 = test

    est = createRandomForest()
    train4[features], test4[features] = normalizedata(train4[features], test4[features], features)

    print 'Train'
    est.fit(train4[features], train4[target])
    print 'Predict'
    predict(est, train4, test4, features, target, filename='submissions-RF-registered.csv')
    for feat in features:
    	print feat
    print est.feature_importances_


if __name__ == "__main__":
    print 'main'
    main()