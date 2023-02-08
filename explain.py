import time
import torch
import math
import numpy as np
from tqdm import tqdm
from lightgbm import plot_importance
import matplotlib.pyplot as plt
from joblib import load

clf = load('./models/layer2/model0.pkl')

feature_names = "cutDirection,colorType,noteLineLayer,lineIndex,scoringType,saberType,saberSpeed,timeDeviation,cutDirDeviation,cutDistanceToCenter,cutAngle,beforeCutRating,afterCutRating,saberDirx,saberDiry,saberDirz,cutPointx,cutPointy,cutPointz,cutNormalx,cutNormaly,cutNormalz,bhpxmin,bhpxmax,bhpxmean,bhpxmed,bhpxstd,bhpymin,bhpymax,bhpymean,bhpymed,bhpystd,bhpzmin,bhpzmax,bhpzmean,bhpzmed,bhpzstd,bhrxmin,bhrxmax,bhrxmean,bhrxmed,bhrxstd,bhrymin,bhrymax,bhrymean,bhrymed,bhrystd,bhrzmin,bhrzmax,bhrzmean,bhrzmed,bhrzstd,bhrwmin,bhrwmax,bhrwmean,bhrwmed,bhrwstd,blpxmin,blpxmax,blpxmean,blpxmed,blpxstd,blpymin,blpymax,blpymean,blpymed,blpystd,blpzmin,blpzmax,blpzmean,blpzmed,blpzstd,blrxmin,blrxmax,blrxmean,blrxmed,blrxstd,blrymin,blrymax,blrymean,blrymed,blrystd,blrzmin,blrzmax,blrzmean,blrzmed,blrzstd,blrwmin,blrwmax,blrwmean,blrwmed,blrwstd,brpxmin,brpxmax,brpxmean,brpxmed,brpxstd,brpymin,brpymax,brpymean,brpymed,brpystd,brpzmin,brpzmax,brpzmean,brpzmed,brpzstd,brrxmin,brrxmax,brrxmean,brrxmed,brrxstd,brrymin,brrymax,brrymean,brrymed,brrystd,brrzmin,brrzmax,brrzmean,brrzmed,brrzstd,brrwmin,brrwmax,brrwmean,brrwmed,brrwstd,ahpxmin,ahpxmax,ahpxmean,ahpxmed,ahpxstd,ahpymin,ahpymax,ahpymean,ahpymed,ahpystd,ahpzmin,ahpzmax,ahpzmean,ahpzmed,ahpzstd,ahrxmin,ahrxmax,ahrxmean,ahrxmed,ahrxstd,ahrymin,ahrymax,ahrymean,ahrymed,ahrystd,ahrzmin,ahrzmax,ahrzmean,ahrzmed,ahrzstd,ahrwmin,ahrwmax,ahrwmean,ahrwmed,ahrwstd,alpxmin,alpxmax,alpxmean,alpxmed,alpxstd,alpymin,alpymax,alpymean,alpymed,alpystd,alpzmin,alpzmax,alpzmean,alpzmed,alpzstd,alrxmin,alrxmax,alrxmean,alrxmed,alrxstd,alrymin,alrymax,alrymean,alrymed,alrystd,alrzmin,alrzmax,alrzmean,alrzmed,alrzstd,alrwmin,alrwmax,alrwmean,alrwmed,alrwstd,arpxmin,arpxmax,arpxmean,arpxmed,arpxstd,arpymin,arpymax,arpymean,arpymed,arpystd,arpzmin,arpzmax,arpzmean,arpzmed,arpzstd,arrxmin,arrxmax,arrxmean,arrxmed,arrxstd,arrymin,arrymax,arrymean,arrymed,arrystd,arrzmin,arrzmax,arrzmean,arrzmed,arrzstd,arrwmin,arrwmax,arrwmean,arrwmed,arrwstd".split(",")

for i in range(22, len(feature_names)):
    sname = feature_names[i]
    lname = ""
    if (sname[4:] == "std"): lname = "Stdev. of "
    if (sname[4:] == "min"): lname = "Min "
    if (sname[4:] == "max"): lname = "Max "
    if (sname[4:] == "mean"): lname = "Mean "
    if (sname[4:] == "med"): lname = "Median "

    if (sname[1:2] == "l"): lname = lname + "Left Hand "
    if (sname[1:2] == "r"): lname = lname + "Right Hand "
    if (sname[1:2] == "h"): lname = lname + "Headset "

    if (sname[2:3] == "p"): lname = lname + "\nPosition "
    if (sname[2:3] == "r"): lname = lname + "\nRotation "

    lname = lname + sname[3:4].upper()

    if (sname[0:1] == "a"): lname = lname + " After Cut"
    if (sname[0:1] == "b"): lname = lname + " Before Cut"

    feature_names[i] = lname


feature_importance = clf._Booster.feature_importance()
total_importance = sum(feature_importance)
relative_importance = [100*importance/total_importance for importance in feature_importance]
feature_importance = list(zip(feature_names, relative_importance))
sorted_importance = sorted(feature_importance, key=lambda x: x[1])
sorted_importance.reverse()

# for feature in :
names = [f[0] for f in sorted_importance[0:10]]
values = [f[1] for f in sorted_importance[0:10]]
colors = ['brown', 'brown', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'blue', 'blue']
names.reverse()
values.reverse()
colors.reverse()
labels = [str(round(pct, 2)) + '%' for pct in values]


fig, ax = plt.subplots()
ax.set_aspect(0.12)

b = plt.barh(names, values, color=colors)
plt.bar_label(b, labels=labels, label_type='center', color='white')
# plt.yticks(labelsize=10)
plt.tick_params(axis='y', which='major', labelsize=8.7)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='brown', lw=4),
                Line2D([0], [0], color='orange', lw=4),
                Line2D([0], [0], color='blue', lw=4)]
plt.legend(custom_lines, ['Height', 'Arm Length', 'Motion'], title="Explanation", title_fontproperties={'weight': 'bold'})
plt.xlabel('% of Splits Explained')
# plt.margins(y=20)
plt.tight_layout()
plt.savefig('features10.pdf')


# print(sorted_importance)

# ax = plot_importance(clf, max_num_features=10)
#
# labels = ax.get_yticklabels()
# labels = [label._text for label in labels]
# labels = [int(label.split("_")[1]) for label in labels]
# labels = [feature_names[label] for label in labels]
# ax.set_yticklabels(labels)
#
# plt.show()
