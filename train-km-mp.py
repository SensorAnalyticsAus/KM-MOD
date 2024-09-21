###############################################################################
#            KM-MOD Image Classifier for Security Cameras 
#                      Sensor Analytics Australia™ 2024
###############################################################################
import sys
if len(sys.argv) < 6:
    print('USAGE: on|off  0|1 numClusters YYYYMMDDHHMMSS YYYYMMDDHHMMSS')
    print('on: run in interactive mode off: run in batch or cron')
    print('0:Elbow Analyses 1:Actual KMeans YYYYMMDDHHMMSS: from -> to date-')
    print('time range (HHMMSS field in 24-hour clock format)')
    sys.exit(1)
import numpy as np
from sklearn.cluster import KMeans
import pickle
import matplotlib.pyplot as plt
from sautils3_5 import fileDt,calcEntropy,writeLog,imgcont,save_list,\
                chunk,workpacks,mkdir_cleared,color,check_img,fileSel
from config import ImgPath,cSz,wdir,deBug
import matplotlib
import os
import cv2
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import glob,tqdm

def p_img_read(name):
 p_img=cv2.imread(name)
 return p_img
def img_load_proc(workpacket):
 tid = workpacket['id']
 fpath = workpacket['imgP']
 flist = workpacket['images']
 file_fnames  = workpacket['outPath_fnames']
 file_data  = workpacket['outPath_data']
 data = []
 fnames = []
 knt=len(flist)
 kn=0
 if deBug:
    print('Task id:',tid)
 for img in flist:
     imgf=os.path.join(fpath,img)
     if check_img(imgf):
        imgd=p_img_read(imgf)
     else: continue
     # Features: Date_TS, Num_Con, Cont_Area, Entropy
     data.append([int(fileDt(img)),imgcont(imgd)[0],imgcont(imgd)[1],
                  calcEntropy(imgf)])
     fnames.append(img)
     kn +=1
 if deBug:
     print('proc:{} Total images processed: {} selected: {} '
           .format(tid,knt,kn))
 with open(file_fnames,'wb') as fpkl:
     fpkl.write(pickle.dumps(fnames))
 with open(file_data,'wb') as fpkl:
     fpkl.write(pickle.dumps(data))
 wDir=os.path.dirname(file_fnames)
 ffstats=os.path.join(wDir,'proc_{}_stats.pkl'.format(tid))
 with open(ffstats,'wb') as fpkl:
     fpkl.write(pickle.dumps('{},{}'.format(knt,kn)))
 return None

##### Main #####
if __name__ == "__main__":

 if not os.path.exists(ImgPath):
     print('fix image path in this code, it  does not exist!',ImgPath)
     sys.exit(1)

 mname = 'km_model.pkl' # trained km model saved as
 print('total images in {} : {}'.format(ImgPath,len(os.listdir(ImgPath))))

 ##########  Data Input Block #################
 opt = int(sys.argv[2]) # 0 for elbow analyses 1 for actual km clustering
 nC = int(sys.argv[3]) # number of clusters for elbow analyses or training 
 st_d_t=datetime.strptime(sys.argv[4],'%Y%m%d%H%M%S').timestamp()
 en_d_t=datetime.strptime(sys.argv[5],'%Y%m%d%H%M%S').timestamp()
 if len(sys.argv[4]) !=14 or len(sys.argv[5]) !=14: 
    print('input datetime error')
    sys.exit(1)
 if int(st_d_t) > int(en_d_t):
    print('input datetime error - inconsistent dates')
    sys.exit(1)

 ############### MultiProcessing Block ###############################
 mkdir_cleared(wdir) # working dir to save serialised mproc outputs
 img_ls=fileSel(ImgPath,sys.argv[4],sys.argv[5])
 print('{} images selected for the date range'.format(len(img_ls)))
 img_ls_chunk=list(chunk(img_ls,cSz)) 
 workpckts=workpacks(ImgPath,img_ls_chunk,workDir=wdir)

 pool = Pool()
 if sys.argv[1] == 'off':
     tRun=True
     print('non-interactive mode:',tRun)
 else: 
     tRun=False
     print('non-interactive mode',tRun)
 for _ in tqdm.tqdm(pool.imap_unordered(img_load_proc,workpckts),
                    total=len(workpckts),colour='magenta',
                    disable=tRun):
   pass
 pool.close()
 pool.join()

 #################### Integrate mp output files in working dir #######
 print('Integrating output files in {}'.format(wdir))
 fnames_all = sorted(glob.glob(wdir+'/*fnames*.pkl'))
 data_all = sorted(glob.glob(wdir+'/*data*.pkl'))
 stats_all= sorted(glob.glob(wdir+'/*stats*.pkl'))

 fnames = []
 data = []
 for i,j in zip(fnames_all,data_all):
    fnames.extend(pickle.load(open(i,'rb')))
    data.extend(pickle.load(open(j,'rb')))
 knt_tot=knt_tot_sel=0
 for f in stats_all:
    s = pickle.load(open(f,'rb'))
    ss=s.split(',') 
    knt_tot += int(ss[0])
    knt_tot_sel  += int(ss[1])
 print('Images selected:{} Images processed:{}'
       .format(knt_tot,knt_tot_sel))

 ############# Saving Outputs ################################
 save_list(fnames,'fnames.txt')
 data = np.array(data)
 np.set_printoptions(suppress=True, formatter={'float_kind':'{:.1f}'.format})
 datamax = data.max(axis=0) #later for scaling prediction vector
 dataNormed = data/datamax
 np.savetxt('datamax.txt',datamax, fmt='%.1f') # datamax saved for this run
 np.savetxt('fnames.txt',fnames,delimiter=" ",fmt="%s") # fnames also saved 

 # Sanity Checks
 if knt_tot_sel < 1:
    print('No Images found exiting!')
    sys.exit(1)
 if knt_tot_sel < nC:
    print('Images found < number of cluster (nC):{} decreasing nC'
           .format(nC))
    nC = knt_tot_sel

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
