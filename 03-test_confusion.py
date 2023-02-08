NOTES_TEST = 50
NUM_USERS = 45000
SLICE = 233

LAYER_1_SIZE = 10
LAYER_2_SIZE = 10

print("Importing Libraries...")
import time
import torch
import math
import numpy as np
from tqdm import tqdm
from lightgbm import LGBMClassifier, log_evaluation
from joblib import load

def getClassifyData(data):
    dataX = data[:, 1:]
    dataY = data[:, 0]
    return dataX, dataY

print("Importing Data...")
testData = torch.load('./data/validate.pt')
testX, testY = getClassifyData(testData)

print("Importing Models (1/2)...")
clfs1 = []
for i in tqdm(range(LAYER_1_SIZE)):
    clfs1.append(load('./models/layer1/model' + str(i) + '.pkl'))

print("Importing Models (2/2)...")
clfs2 = []
for i in tqdm(range(LAYER_2_SIZE)):
    clfs2.append(load('./models/layer2/model' + str(i) + '.pkl'))

mtrxA = []
mtrxB = []

def predictUser(i):
    preds1 = []
    for j in range(LAYER_1_SIZE):
        preds1.append(clfs1[j].predict_proba(testX[50*i:NOTES_TEST+50*i]))
    pred1 = np.hstack(preds1).sum(axis=0)
    mtrxA.append(pred1)

    preds2 = []
    for j in range(LAYER_2_SIZE):
        preds2.append(clfs2[j].predict_proba(testX[50*i:NOTES_TEST+50*i]))
    pred2 = np.hstack(preds2).sum(axis=0)
    mtrxB.append(pred2)

    pred = []
    for j in range(NUM_USERS):
        model2 = j % LAYER_2_SIZE
        pos2 = math.floor(j / LAYER_2_SIZE)
        users_per_round2 = NUM_USERS // LAYER_2_SIZE
        pred.append(pred1[j] + pred2[model2*users_per_round2 + pos2])
    return np.argmax(pred)

print("Testing Accuracy...")
valid = 0
total = 0
cfsn = []
t = tqdm(range(NUM_USERS), desc='0/0 Valid (0%)')
for i in t:
    pred = predictUser(i)
    if (pred == i): valid += 1
    else: cfsn.append([i, pred])
    total += 1
    t.set_description(str(valid) + "/" + str(total) + " Valid (" + str((valid/total)*100) + "%)")

file = open('./data/cfsn.txt','w')
file.write("\n".join([str(item[0]) + ',' + str(item[1]) for item in cfsn]))
file.close()

np.save('./data/mtrxA', mtrxA)
np.save('./data/mtrxB', mtrxB)
