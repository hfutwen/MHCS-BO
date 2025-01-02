import sys
sys.path.append('E:\\experiment\\code\\')
from experiment.singleExperiment import experiment
from atomTemperature.atomTemperature import AT
from BO.BayesianOpt import bayesianOpt as bopt
from BO.BayesianOpt2 import bayesianOpt as bopt2
import os,time
import numpy as np



# 定义目标函数
def calcObj(k=0.5):
    while(True):
        dataDir = f'E:/其他/简仪采集卡/MultiChannel/bin/Debug/allData{exp.num}.csv'
        if os.path.exists(dataDir):
            time.sleep(0.2)
            at = AT(path=exp.path)
            data = np.loadtxt(fname=dataDir)
            T,a,v,vPre,t,y,data = at.temperatureFromTOF_vField(data=data,fs=1e4,t0=0.2,h=0.235,range=exp.range,fileName=f'fit{exp.num}',pltFlag=True)
            break
        else:
            time.sleep(0.1)

    exp.num += 1
    print('T={t:.2f}uK'.format(t=T))

    return np.abs(a**k/T)


def objFun_lattice_only(params):
    x=formatParams(params=params)
    exp.coolingType='lattice_only'
    exp.runExperiment(x)
    exp.moveRunFile()
    return calcObj()

def objFun_multiPGC(params):
    x=formatParams(params=params)
    exp.coolingType='multiPgc_magDelay'
    exp.runExperiment(x)
    exp.moveRunFile()
    return calcObj()

def formatParams(params):
    x=[]
    if isinstance(params, dict):
        for i in range(len(params)):
            x.append(params[f'x{i}'])
    elif isinstance(params, list) or isinstance(params, np.ndarray):
        x=params   
    return x

# 实验准备
path='E:\\experiment\\0atomAccelerator\\2024\\BO_lattice_SDS\\'
exp = experiment(path=path)
exp.range = range(275,350)
# exp.range = range(100,500)
exp.codeName = 'Raman_Rabi.py'
exp.moveCode()
current_file = os.path.basename(__file__)
exp.moveCode(current_file)

# # 参数搜索空间
# att2，deltaPGC, deltaT1-deltaT10， magDelay
# space = [(27, 31.5),(150, 240),
#          (0.1, 10),(0.1, 10),(0.1, 10),(0.1, 10),(0.1, 10),
#          (0.1, 10),(0.1, 10),(0.1, 10),(0.1, 10),(0.1, 10),(0.1, 20)]

#lattDelay,magDelay,tDa,Von
space = [(1, 30),(1, 30),(0.05, 1.1),(3.0, 6)]

n_init, N_iter = 10, 100
bo = bopt2(bounds=space, objectFun=objFun_lattice_only,n_init=n_init,n_iter=N_iter,converge_rate=0.2,path=exp.path)
# bo = bopt2(bounds=space, objectFun=objFun_multiPGC,n_init=n_init,n_iter=N_iter,converge_rate=0.2,path=exp.path)
# bo.runOpt_Native()
bo.runOpt(SDS_Flag=True)
bo.save()




