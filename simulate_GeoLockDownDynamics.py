import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import random
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import time
import os,re
import argparse
import importlib

def after_k(pos,D,k):
    for i in range(k):
        pos = D[pos]()
        
    return pos



def after_k_with_stop(pos,D,k,max_l,stop_state,transition_state):
    cnt  = 0
    for i in range(k):
        if pos == stop_state:
            cnt+=1
        if pos != stop_state:
            cnt = 0
        if cnt >= max_l:
            pos = transition_state
        else:
            pos = D[pos]()
        #print("time %d pos %s"%(i,pos))#random.choices(population=['a', 'b', 'c'],weights=[0.5, 0.3, 0.2])))
    return pos


def test_after_k_with_stop():
    k = 50
    trials = 10000
    start_tm = time.time()
    st = 'a'
    A = [after_k_with_stop(st,D,k,5,'c','b') for ii in range(trials)]
    #print([(x,A.count(x)/float(trials)) for x in D.keys()])
    print("total time %0.2f"%(time.time()-start_tm))



def step(init_distr,D):
    """
    """
    for k in init_distr.keys():
        init_distr[k] = D[init_distr[k]]() 
    return init_distr

def count_s(dist,labels):
    vals = list(dist.values())
    return [(l,vals.count(l)) for l in labels]
    
def simul(init_distr,D,num_of_iter,threshold = 14):
    distr = init_distr.copy()
    timer = {k: 0 for k in distr.keys()}
    for ii in range(num_of_iter):
        if list(distr.values()).count('c') == 0:
            return ii,count_s(distr,list(D.keys()))
        distr = step(distr,D)
        for k in distr.keys():
            if distr[k] == 'c':
                timer[k]+=1
                if timer[k] > threshold :
                    distr[k]  = 'b'
                    timer[k] = 0
            else:
                timer[k] = 0
            
    return num_of_iter,count_s(distr,list(D.keys()))


# In[145]:

def read_covid_simulate_config(config_file):
    conf = importlib.import_module(config_file.split(".")[0])
    return {"trans_mat" : conf.trans_mat, "high_risk_threshold" : conf.high_risk_threshold,"cities_dist":conf.cities_dist }

    




def create_histogram(T,num_of_bins,factor,dest_dir):
	##print("start create_histogram")
	fig, ax = plt.subplots()
	freq, bins, _ = ax.hist(T, num_of_bins, facecolor='skyblue')
	ax.set_xlabel('%s'%factor)
	ax.set_ylabel('Freq')                             
	plt.title('%s Histogram'%(factor))
	plt.savefig(os.path.join(dest_dir,factor+"_hist.png").replace(" ","_"))
	plt.clf()
	plt.close()


# In[148]:

def run_simulation(trans_mat,high_risk_threshold,cities_dist,dest,num_of_iter = 500,hist_num_samples=1000):
    st1 = time.time()
    BB = [simul(cities_dist,trans_mat,num_of_iter,high_risk_threshold) for ii in range(hist_num_samples)]
    AA = [xx[0] for xx in BB]

    print("running time is %0.2f"%(time.time() - st1))
    M= [ [y[0] for y in xx[1]] for xx in BB]
    N= [zz[1]  for zz in BB[0][1]]
    df1  = pd.DataFrame([ [y[1] for y in xx[1]] for xx in BB],columns = [zz[0]  for zz in BB[0][1]])
    df1.sum()
    
    create_histogram(AA,10,"simul",dest)

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Interface to simulation Cov19 mitigation Policy')
    parser.add_argument('--config_file', dest='config',  help='destination to the config_file, if not defined default_config_simulate.py will be used' )
    parser.add_argument('--Dest',dest='dest_dir',help ='<optional> Destination Dir')
    args = parser.parse_args()
    if args.dest_dir :
        dest_dir = args.dest_dir
        print("dest_dir %s"%args.dest_dir )
    else:
        dest_dir = "Dest"
    if args.config:
        config_file = args.config
    else:
        config_file = "default_config_simulate.py"
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    inputs = read_covid_simulate_config(config_file)
    inputs["dest"] = dest_dir
    run_simulation(**inputs)





