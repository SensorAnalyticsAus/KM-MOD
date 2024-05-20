###############################################################################
#            KM-MOD Image Classifier for Security Cameras 
#                      Sensor Analytics Australia™ 2024
###############################################################################
ImgPath='../../foscamR4/snap'

import sys
if len(sys.argv) < 5:
   print('USAGE: 0|1 numClusters YYYYMMDDHHMMSS YYYYMMDDHHMMSS')
   print('0:Elbow Analyses 1:Actual KMeans YYYYMMDDHHMMSS: from -> to date-')
   print('time range (HHMMSS field in 24-hour clock format)')
   sys.exit(1)
import numpy as np
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
from sautils3_3 import fileDt,calcEntropy,writeLog,imgcont
import matplotlib
import os
import cv2
from datetime import datetime

def p_img_read(name):
 p_img=cv2.imread(name)
 return p_img

#   pool = Pool()                       # Create a multiprocessing Pool
#   pool.map(p_img_read, img_ls)# process data_inputs iterable with pool

if not os.path.exists(ImgPath):
    print('fix image path in this code, it  does not exist!',ImgPath)
    sys.exit(1)
mname = 'km_model.pkl' # trained km model saved as

##########  Data Input and Normalization Block ##########
opt = int(sys.argv[1]) # 0 for elbow analyses 1 for actual km clustering
nC = int(sys.argv[2]) # number of clusters for elbow analyses or training 
st_d_t=datetime.strptime(sys.argv[3],'%Y%m%d%H%M%S').timestamp()
en_d_t=datetime.strptime(sys.argv[4],'%Y%m%d%H%M%S').timestamp()
if len(sys.argv[3]) !=14 or len(sys.argv[4]) !=14: 
    print('input datetime error')
    sys.exit(1)
if int(st_d_t) > int(en_d_t):
    print('input datetime error - inconsistent dates')
    sys.exit(1)
data = []
fnames = []
kn=knt=0
img_ls=os.listdir(ImgPath)
fileC=len(img_ls)
print('total images in {} : {}'.format(ImgPath,fileC))
for img in img_ls:
    fDt=datetime.strptime(fileDt(img),'%Y%m%d%H%M%S').timestamp()
    if int(fDt) >= int(st_d_t) and int(fDt) <= int(en_d_t):
       imgf=os.path.join(ImgPath,img)
       imgd=p_img_read(imgf)
       writeLog(img,'./fnames.txt') #save image filenames
       # Date_TS, Num_Con, Cont_Area, Entropy
       data.append([int(fileDt(img)),imgcont(imgd)[0],imgcont(imgd)[1],
                      calcEntropy(imgf)])
       fnames.append(img)
       kn +=1
    if(knt%100 == 0): 
        print('images processed: {} selected: {} '.format(knt,kn),
              end='\r',flush=True)
    knt +=1
print('Total images processed: {} selected: {} '.format(knt,kn))
data = np.array(data)
np.set_printoptions(suppress=True, formatter={'float_kind':'{:.1f}'.format})
datamax = data.max(axis=0) #later for scaling prediction vector
dataNormed = data/datamax
np.savetxt('datamax.txt',datamax, fmt='%.1f') # datamax saved for this run
np.savetxt('fnames.txt',fnames,delimiter=" ",fmt="%s") # fnames also saved 

######### Elbow Analyses and Display Block ###############
if opt == 0:
    matplotlib.use('TkAgg') # to avoid cv2 qt conflict
    inertias = []

    print('Getting ready to display')
    for i in range(1,int(sys.argv[2])):
        print('.',end='')
        km = KMeans(n_clusters=i)
        km.fit(dataNormed)
        inertias.append(km.inertia_)
    print('\n')

    print('press [q] inside the display chart to end')
    plt.plot(range(1,int(sys.argv[2])), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show(block=True) 
    sys.exit(0)
################### KMeans Block ################################
print('ready to train KMeans with:',nC,'clusters')
km = KMeans(n_clusters=nC,n_init=10) #to get rid of warning
km.fit(dataNormed)
pickle.dump(km, open(mname, 'wb')) # dump trained model
print('KMeans trained model saved as:',mname)
