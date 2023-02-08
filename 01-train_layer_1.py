NOTES_TRAIN = 150
NOTES_VALIDATE = 3
NOTES_TEST = 10
NUM_TREES = 200
NUM_ROUNDS = 10
NUM_USERS = 45000

print("Importing Libraries...")
import time
import torch
import numpy as np
from lightgbm import LGBMClassifier, log_evaluation
from joblib import dump

print("Importing Data...")
trainData = torch.load('./data/train.pt')
testData = torch.load('./data/test.pt')
validateData = torch.load('./data/validate.pt')

def getClassifyData(data):
    dataX = data[:, 1:]
    dataY = data[:, 0]
    return dataX, dataY

for round in range(NUM_ROUNDS):
    print("Starting Round " + str(round+1) + "/" + str(NUM_ROUNDS) + "...")

    print("Selecting Data...")
    trainFrame = []
    testFrame = []
    validateFrame = []
    users_per_round = NUM_USERS // NUM_ROUNDS
    for id in range(users_per_round*round, users_per_round*(round+1)):
        trainFrame.append(trainData[150*id:NOTES_TRAIN+150*id])
        testFrame.append(testData[50*id:NOTES_TEST+50*id])
        validateFrame.append(validateData[50*id:NOTES_VALIDATE+50*id])

    print("Processing Data...")
    trainX, trainY = getClassifyData(torch.cat(trainFrame))
    testX, testY = getClassifyData(torch.cat(testFrame))
    validateX, validateY = getClassifyData(torch.cat(validateFrame))

    print("Training Model " + str(round+1) + "/" + str(NUM_ROUNDS) + "...")
    clf = LGBMClassifier(boosting_type='goss', colsample_bytree=0.6933333333333332, learning_rate=0.1, \
        max_bin=63, max_depth=-1, min_child_weight=7, min_data_in_leaf=20, \
        min_split_gain=0.9473684210526315, n_estimators=NUM_TREES, \
        num_leaves=23, reg_alpha=0.7894736842105263, reg_lambda=0.894736842105263, \
        subsample=1, n_jobs=16, objective='multiclassova', device_type='gpu')
    start_time = time.time()
    clf.fit(trainX, trainY.long(),
           eval_set=[(validateX, validateY)],
           eval_metric='multi_error',
           callbacks=[log_evaluation()])
    print("Training Finished in %s Minutes" % ((time.time() - start_time) / 60))

    print("Testing Model " + str(round+1) + "/" + str(NUM_ROUNDS) + "...")
    testPreds = torch.tensor(clf.predict(testX))
    print('Test Accuracy (1 Sample):', torch.sum(testPreds.long() == testY.long()).item() / testY.shape[0])
    uids = {}
    for (pred, real) in zip(testPreds.int().numpy(), testY.int().numpy()):
        if (not real in uids): uids[real] = []
        uids[real].append(pred)
    for uid in uids:
        uids[uid] = np.bincount(uids[uid]).argmax()
    for uid in uids:
        uids[uid] = (uid == uids[uid])
    print('Test Accuracy (Test Set):', sum(uids.values()) / len(uids))

    print("Saving Model " + str(round+1) + "/" + str(NUM_ROUNDS) + "...")
    dump(clf, './models/layer1/model' + str(round) + '.pkl')
