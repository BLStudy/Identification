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

colors = ['green']*22 + ['blue']*210

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

    if (sname[4:] == "min" or sname[4:] == "max"):
        if (sname[2:3] == "p"):
            colors[i] = 'orange'

    feature_names[i] = lname


feature_importance = clf._Booster.feature_importance(importance_type='gain')
total_importance = sum(feature_importance)
relative_importance = [100*importance/total_importance for importance in feature_importance]
feature_importance = list(zip(feature_names, relative_importance, colors))
sorted_importance = sorted(feature_importance, key=lambda x: x[1])
sorted_importance.reverse()

values = [f[1] for f in sorted_importance if f[1] > 0]
colors = [f[2] for f in sorted_importance if f[1] > 0]

plt.xlabel('Feature Importance Rank (#)')
plt.ylabel('% of Entropy Gain Explained')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='orange', lw=4),
                Line2D([0], [0], color='blue', lw=4),
                Line2D([0], [0], color='green', lw=4)]
plt.legend(custom_lines, ['Static Features', 'Motion Features', 'Context Features'], title="Feature Type", title_fontproperties={'weight': 'bold'})

print(sum(f[1] for f in sorted_importance if f[2] == 'green'))
exit()

plt.bar(range(len(values)), values, width=1.0, color=colors)
plt.tight_layout()
plt.savefig('features10.pdf')

plt.show()
