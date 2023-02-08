print("Importing Libraries...")

import time
import torch
import pandas as pd
import numpy as np
import math
from tqdm import tqdm, trange
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier, Booster, early_stopping, log_evaluation
from joblib import dump, load

headers = open("headers.csv").read().split(",")
users = open("users.txt").read().split("\n")
groups = open("./layer2/groups.txt").read().split("\n")
groups = [list(map(int, group.split(","))) for group in groups]
groupsets = [set(group) for group in groups]

print("Importing Classifiers (1/3)...")
clfs1 = []
for i in tqdm(range(20)):
    clfs1.append(Booster(model_file='./primary/round' + str(i) + '.txt'))

print("Importing Classifiers (2/3)...")
clfs2 = []
for i in tqdm(range(20)):
    clfs2.append(Booster(model_file='./primary2/round' + str(i) + '.txt'))

print("Importing Classifiers (3/3)...")
clfs3 = []
for i in tqdm(range(len(groups))):
    clfs3.append(load('./layer3/round' + str(i) + '.joblib'))

def standardizednp(validateDF: pd.DataFrame) -> tuple:
    df = validateDF
    df['uid'] = pd.factorize(df['uid'])[0]
    df.drop('nid', axis=1, inplace=True)
    cols = df.columns
    cols = cols.delete(list(range(7)))
    ct = ColumnTransformer([
        ('StandardScaler', StandardScaler(), cols)
    ], remainder='passthrough')
    datas = ct.fit_transform(df)
    validateData = torch.tensor(datas)
    validateData = torch.cat([validateData[:, -7:], validateData[:, :-7]], dim=1)
    return validateData

def getClassifyData(data):
    dataX = data[:, 1:]
    dataY = data[:, 0]
    return dataX, dataY

print("Importing Data...")
testData = torch.load('testdata.pt')
testX, testY = getClassifyData(testData)

for round in range(20):
    print("Starting Round " + str(round+1) + "/20...")
    print("Importing Data...")
    validateDF = []
    for user in tqdm(users[2250*round:2250+2250*round]):
        for i in range(2):
            try:
                df = pd.read_csv('./users/' + str(user) + '/test/' + str(i) + '.csv', header=None)
                df.columns = headers
                validateDF.append(df)
            except Exception:
                pass
    validateDF = pd.concat(validateDF)
    validateDF.dropna(inplace=True)
    validateData = standardizednp(validateDF)
    validateX, validateY = getClassifyData(validateData)

    valid = 0
    total = 0
    cfsn = []
    mtrx1 = []
    mtrx2 = []

    def predictUser(i):
        preds1 = []
        for j in range(20):
            preds1.append(clfs1[j].predict(validateX[20*i:20+20*i]))
        pred1 = np.hstack(preds1).sum(axis=0)
        mtrx1.append(pred1)

        preds2 = []
        for j in range(20):
            preds2.append(clfs2[j].predict(validateX[20*i:20+20*i]))
        pred2 = np.hstack(preds2).sum(axis=0)
        mtrx2.append(pred2)

        pred = []
        for j in range(45000):
            model = j % 20
            pos = math.floor(j / 20)
            pred.append(pred1[j] + pred2[model*2250 + pos])
        pred = np.argmax(pred)

        for j in range(len(groups)):
            if (pred in groupsets[j]):
                preds3 = clfs3[j].predict(testX[50*i:50+50*i])
                return np.bincount(preds3).argmax()

        return pred

    print("Testing...")
    t = trange(2250, desc='0/0 Valid (0%)')
    for i in t:
        pred = predictUser(i)
        if (pred == i + 2250*round): valid += 1
        else: cfsn.append([i + 2250*round, pred])
        total += 1
        t.set_description(str(valid) + "/" + str(total) + " Valid (" + str((valid/total)*100) + "%)")

    file = open('./layer3/cfsn' + str(round) + '.txt','w')
    file.write("\n".join([str(item[0]) + ',' + str(item[1]) for item in cfsn]))
    file.close()

    file = open('./layer3/mtrxA' + str(round) + '.txt','w')
    file.write("\n".join([",".join(map(str,item)) for item in mtrx1]))
    file.close()

    file = open('./layer3/mtrxB' + str(round) + '.txt','w')
    file.write("\n".join([",".join(map(str,item)) for item in mtrx2]))
    file.close()
