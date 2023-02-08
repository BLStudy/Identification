import time
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier, Booster, early_stopping, log_evaluation
import math

headers = open("headers.csv").read().split(",")
users = open("users.txt").read().split("\n")

def standardizednp(trainDF: pd.DataFrame, testDF: pd.DataFrame, validateDF: pd.DataFrame) -> tuple:
    trainSize = trainDF.shape[0]
    testSize = testDF.shape[0]
    df = pd.concat([trainDF, testDF, validateDF])
    df['uid'] = pd.factorize(df['uid'])[0]
    df.drop('nid', axis=1, inplace=True)
    cols = df.columns
    cols = cols.delete(list(range(7)))
    ct = ColumnTransformer([
        ('StandardScaler', StandardScaler(), cols)
    ], remainder='passthrough')
    datas = ct.fit_transform(df)
    trainData = torch.tensor(datas[:trainSize])
    testData = torch.tensor(datas[trainSize:trainSize+testSize])
    validateData = torch.tensor(datas[trainSize+testSize:])
    trainData = torch.cat([trainData[:, -7:], trainData[:, :-7]], dim=1)
    testData = torch.cat([testData[:, -7:], testData[:, :-7]], dim=1)
    validateData = torch.cat([validateData[:, -7:], validateData[:, :-7]], dim=1)
    return trainData, testData, validateData

trainDF = []
validateDF = []
testDF = []
for user in tqdm(users):
    try:
        df = pd.read_csv('./users/train/' + str(user) + '.csv', header=None)
        df.columns = headers
        trainDF.append(df)
    except Exception:
        pass
    try:
        df = pd.read_csv('./users/validate/' + str(user) + '.csv', header=None)
        df.columns = headers
        validateDF.append(df)
    except Exception:
        pass
    try:
        df = pd.read_csv('./users/test/' + str(user) + '.csv', header=None)
        df.columns = headers
        testDF.append(df)
    except Exception:
        pass

trainDF = pd.concat(trainDFX)
validateDF = pd.concat(validateDFX)
testDF = pd.concat(testDFX)

trainDF.dropna(inplace=True)
testDF.dropna(inplace=True)
validateDF.dropna(inplace=True)

trainData, testData, validateData = standardizednp(trainDF, testDF, validateDF)

torch.save(trainData, './data/train.pt')
torch.save(testData, './data/test.pt')
torch.save(validateData, './data/validate.pt')
