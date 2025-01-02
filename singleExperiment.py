import os, sys
import time
import numpy as np
import matplotlib.pyplot as plt
import shutil
import subprocess
import h5py
import pltconfv1

sys.path.append('E:\\experiment\\code\\')
from atomTemperature.atomTemperature import AT


class experiment:
    def __init__(self,path):
        self.num=1
        self.path=path
        self.range = range(50,700)
        self.codePath = 'E:\\experiment\\code\\experiment\\'
        self.codeName = None
        self.coolingType=None
        self.str_keys = []

        if not os.path.exists(self.path + 'data'):
            os.mkdir(self.path + 'data')

    def runExperiment(self,params=None, **keyWord):
        strParams=None
        para = None
        if params is None: # 如果为空
            pass
        else:
            para = np.round(params,2)
        str0 = 'call conda activate artiq5py35 \n'
        str1 = 'cd /d '+ self.codePath +'\n'
        str2 = 'call artiq_run ' + self.codePath + self.codeName 
        strParams = str0 + str1 + str2

        if self.coolingType == None:
            pass

        elif self.coolingType=='multiPgc_magDelay': 
            self.str_keys.extend(['att2','deltaPGC','deltaT1','deltaT2','deltaT3','deltaT4','deltaT5',
                         'deltaT6','deltaT7','deltaT8','deltaT9','deltaT10','magDelay'])
        

        elif self.coolingType=='lattice_only':
            self.str_keys.extend(['lattDelay','magDelay','tDa','Von'])


        for k, v in keyWord.items():
            if k == 'motAtLast':
                strParams += f' motAtLast={v}'

        if not len(self.str_keys)==0:
            temp = ''
            for i in range(len(self.str_keys)):
                temp += ' ' + self.str_keys[i]+'={}'
            strParams += temp.format(*para) +' \n'
            self.str_keys=[]

        with open(self.codePath + "run.bat", "w") as file:
            file.write(strParams)

        subprocess.call(self.codePath + 'run.bat', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        # subprocess.call(self.codePath + 'run.bat')
        time.sleep(0.1)
    
    def calcFilteredSigma(self):
        # 读取并拟合数据
        targetFile = 'E:/其他/简仪采集卡/MultiChannel/bin/Debug/allData{}.csv'
        targetFile = targetFile.format(self.num)
        data = np.loadtxt(targetFile)
        noise = np.mean(data[:100])
        y = data-noise
        window_size = 5
        cumsum = np.cumsum(np.insert(y, 0, 0))
        yPre = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        A = np.max(yPre)
        x = np.arange(len(yPre))
        x_interp = np.linspace(0,x[-1],2000)
        y_interp = np.interp(x_interp,x,yPre)
        ind = np.where(y_interp>A/2)
        ind1 = ind[0][0]
        ind2 = ind[0][-1]
        sigma = (ind2-ind1)/2
        if A<0.1:
            sigma=0.1

        if not os.path.exists(self.path + "data"):
            os.mkdir(self.path + "data")
        h5 = h5py.File(self.path + 'data\\data{}.h5'.format(self.num),mode='w')
        h5.create_dataset(data=y,name='y')
        h5.create_dataset(data=yPre,name='yPre')
        h5.create_dataset(data=data,name='data')
        h5.create_dataset(data=ind,name='ind')
        h5.create_dataset(data=sigma,name='sigma')
        h5.create_dataset(data=A,name='A')

        if not os.path.exists(self.path + "img"):
            os.mkdir(self.path + "img")
        plt.figure()
        plt.plot(y, '--')
        plt.plot(yPre,alpha=0.5,lw=3)
        plt.plot(x_interp[ind1],y_interp[ind1],'ro',alpha=0.5)
        plt.plot(x_interp[ind2],y_interp[ind2],'ro',alpha=0.5)
        plt.title('sigma={:.2f},A={:.2f}'.format(sigma,A))
        plt.savefig(self.path + 'img\\fit{}.png'.format(self.num))
        plt.close()

        return sigma,A

    
    def moveRunFile(self):
        if not os.path.exists(self.path + "runFiles"):
            os.mkdir(self.path + "runFiles")
        shutil.copy(self.codePath+"run.bat",self.path + "runFiles\\run{}.bat".format(self.num))

    def moveCode(self,name=None):    
        if not os.path.exists(self.path + "code\\"):
            os.mkdir(self.path + "code\\")

 
        if name==None:
            shutil.copy(self.codePath+'singleExperiment.py',self.path + "code\\" + 'singleExperiment.py')
            shutil.copy(self.codePath+ self.codeName,self.path + "code\\" + self.codeName)
        else:
            shutil.copy(name,self.path + "code\\" + name)

if __name__=='__main__':
    # # 人工优化的典型参数
    exp = experiment(path='E:\\experiment\\0atomAccelerator\\2024\\test\\')
    exp.num = 1
    # exp.range = range(100,500)
    exp.range = range(275,350)
    exp.codeName = 'Raman_Rabi.py'
    exp.coolingType = None    
    exp.runExperiment(motAtLast=0)
    exp.moveCode() 
    time.sleep(2.5)
    at = AT(path=exp.path)
    at.moveCode()
    dataPath = 'E:\\其他\\简仪采集卡\\MultiChannel\\bin\\Debug\\'
    data = np.loadtxt(dataPath + 'allData{}.csv'.format(exp.num))
    T,a,v,vPre,t,y,data = at.temperatureFromTOF_vField(data=data,fs=1e4,t0=0.2,h=0.235,range=exp.range,fileName=f'fit{exp.num}',pltFlag=True)
    print("T",T)



