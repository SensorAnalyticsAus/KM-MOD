###############################################################################
#            KM-MOD Image Classifier for Security Cameras 
#                      Sensor Analytics Australia™ 2024
###############################################################################
import sys,os
if len(sys.argv) != 2:
    print('USAGE: -i|perc')
    print('-i: to run interactively')
    print('perc: select clusters lte <= percentile: (usually 20-80)')    
    sys.exit(1)
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import pickle
from config import ImgPath as prefixP

if not os.path.exists(prefixP):
    print('path',prefixP,'does not exist! fix prefixP')
    sys.exit(1)

fnames = np.loadtxt('fnames.txt',dtype=str)
print('fnames loaded:',len(fnames),'lines')
datamax = np.loadtxt('datamax.txt')

km = pickle.load(open('km_model.pkl', 'rb')) 
km.labels_
labels = km.labels_
cc = km.cluster_centers_

nC = len(cc)
print('clusters in this model:',nC)

membersC = [] #members in each cluster
mC = []
for i in range(0,nC):
    membersC.append(np.where(labels == i)[0])
    mC.append(len(membersC[i]))
mCs = sorted(mC)
print('cluster sizes in ascending order:')
for i in range(0,nC):
    print(mCs[i],' ',end='')
print('\n')

ffnames = [] # to store full path (which exceeds fnames array elem size)

f = open('ffnames.txt','w')

if sys.argv[1] == '-i':
    perc = float(input('Select clusters < = percentile (usually 20-80):'))
else:
    perc = float(sys.argv[1])
print('cluster size <=',perc,'percentile:',np.percentile(mC,perc))

for i in range(0,nC):
    if mC[i] <= np.percentile(mC,perc): # select clusters <=  perc 
        for j in membersC[i]:
            ffnames.append(os.path.join(prefixP,fnames[j]))
ffnames.sort()
for i in ffnames:
    f.write(i+'\n')
print(len(ffnames),'cluster imgfiles saved as ffnames.txt')

f.close()
