import numpy as np
import os
from scipy.optimize import minimize
import h5py
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm
import shutil
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process.kernels import RBF
import os


class bayesianOpt:
    def __init__(self, bounds, objectFun, n_init=10, n_iter=100, converge_rate=0.2, path=None) -> None:
        self.bounds = bounds
        self.objectFun = objectFun
        self.n_init = n_init
        self.n_iter = n_iter
        kernel = RBF(length_scale=0.01,length_scale_bounds=(0.001,5))
        self.gpr = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=100,alpha=5e-3,optimizer='fmin_l_bfgs_b',normalize_y=True)
        self.path = path
        self.X_train=None
        self.y_train=None
        self.lb = np.array([bounds[i][0] for i in range(len(bounds))])
        self.ub = np.array([bounds[i][1] for i in range(len(bounds))])
        self.converge_rate = converge_rate
        self.scaler = MinMaxScaler(feature_range=(0,1))

        test_data = np.array([self.lb,self.ub])
        print('测试数据','\n',test_data)
        data = self.scaler.fit_transform(np.reshape(test_data,(2,len(self.bounds))))
        print('归一化测试','\n',test_data,'\n',data)

        if not os.path.exists(self.path + "code\\"):
            os.mkdir(self.path + "code\\")
        shutil.copy('E:\\experiment\\code\\BO\\BayesianOpt2.py',self.path + "code\\BayesianOpt2.py")

        self.combo_set = []
        self.r_List = []
        self.X_c_list = []

    def findLargeNB(self,X):
        #把新的组合、球心和半径添加进来
        N = len(self.X_train)
        for i in range(N):
            self.combo_set.append((i,N))
            x1 = self.X_train[i]
            x2 = X
            r = np.linalg.norm(x1 - x2)/2
            x_c = (x1+x2)/2
            self.r_List.append(r)
            self.X_c_list.append(x_c)

        # 剔除不满足要求的组合
        X_to_check = np.vstack((self.X_train,X))
        to_remove = []  # 用于存储需要删除的索引值
        for i in range(len(self.r_List)):
            next_flag=False
            for j in range(len(X_to_check)):
                if next_flag:
                    break
                # print('j,comb,panduan',j,combo_set[i],j in combo_set[i])
                if j in self.combo_set[i]:
                    continue
                x_ = X_to_check[j]
                x_c = self.X_c_list[i]
                d_ = np.linalg.norm(x_ - x_c)

                if d_<self.r_List[i]:
                    to_remove.append(i)
                    next_flag=True
        for index in sorted(to_remove, reverse=True):  # 倒序删除以避免索引错位
            self.r_List.pop(index)
            self.X_c_list.pop(index)   
            self.combo_set.pop(index)  

        # 找出直径最大的超球面，其球心即为最终的推荐点
        max_id = np.argmax(self.r_List)
        x_next = self.X_c_list[max_id]
        return x_next
    
    def findLargeNB_init(self):
        # 计算每两个数组之间的欧氏最小邻域
        # 算出每个组合对应超球面的球心和半径
        combo_set = []
        r_List=[]
        X_c_list=[]
        N = len(self.X_train)
        for i in range(N):
            for j in range(N):
                if j==i:
                    continue
                if ((i,j) in combo_set) or ((j,i) in combo_set):
                    continue
                combo_set.append((i,j))
                x1 = self.X_train[combo_set[i][0]]
                x2 = self.X_train[combo_set[i][1]]
                r = np.linalg.norm(x1 - x2)/2
                x_c = (x1+x2)/2
                r_List.append(r)
                X_c_list.append(x_c)
            
        print('所有组合',combo_set)
        to_remove = []  # 用于存储需要删除的索引值
        for i in range(len(r_List)):
            next_flag=False
            for j in range(N):
                if next_flag:
                    break
                if j in combo_set[i]:
                    continue
                x_check = self.X_train[j]
                x_c = X_c_list[i]
                d_ = np.linalg.norm(x_check - x_c)

                if d_<r_List[i]:
                    to_remove.append(i)
                    next_flag=True
        for index in sorted(to_remove, reverse=True):  #
            r_List.pop(index)
            X_c_list.pop(index)   
            combo_set.pop(index)  
        self.r_List += r_List
        self.X_c_list += X_c_list
        self.combo_set += combo_set

        #
        max_id = np.argmax(r_List)
        x_next = X_c_list[max_id]
        return x_next

    # 期望增强函数
    def expected_improvement(self, x, y_train):
        x = x.reshape(1, -1)  # 将 x 转换为二维数组
        mu, sigma = self.gpr.predict(x, return_std=True)
        f_best = np.max(y_train)
        Z = (mu - f_best) / sigma
        improvement = (mu - f_best) * norm.cdf(Z) + sigma * norm.pdf(Z)
        if sigma==0:
            improvement=0
        return improvement,mu,sigma  # 
    
    def predict(self, y_train, s, step=0, decay=0.01, acq_mu_Flag=False):
        """
        用于选择下一个采样点的预测函数。
    
        参数:
            y_train: array-like，训练数据的目标值。
            acq_mu_Flag: bool，控制期望改进方法的选择，默认为False。
    
        返回:
            x_next: array-like，下一个采样点。
            acq_value: float，采样点的获得值。
        """
        acquisition_ei = lambda x: -1*self.expected_improvement(x, y_train)[0]
        acquisition_mu = lambda x: -1*self.expected_improvement(x, y_train)[1]
        acquisition_hybrid = lambda x: -20*(1-decay)**step*self.expected_improvement(x, y_train)[0]-1*(1+decay)**step*self.expected_improvement(x, y_train)[1]


        acq_value=0
        x0=np.random.uniform(0, 1, size=(1000,len(self.bounds)))
        max_acq = -np.squeeze(acquisition_ei(x0[0]))
        x_max=x0[0]
        for i in range(len(x0)):
            acq_value += -np.squeeze(acquisition_ei(x0[0]))
            bounds = [(0, 1) for _ in range(len(self.bounds))]
            N_opt = 30
            if i<N_opt: # 为避免陷入局部最优，利用N_opt个随机初始值做N_opt次优化
                if not acq_mu_Flag:
                    res = minimize(fun=acquisition_ei, # acquisition_hybrid,
                                x0=x0[i],
                                bounds=bounds,
                                method="L-BFGS-B")
                else:
                    res = minimize(fun=acquisition_mu,
                                x0=x0[i],
                                bounds=bounds,
                                method="L-BFGS-B")
                
                if not res.success:
                    continue
                temp = -np.squeeze(res.fun)
                if temp>=max_acq:
                    x_max=res.x
                    max_acq = temp
        x_next = np.clip(x_max, 0, 1)  

        return x_next,acq_value
    
    
    def runOpt(self,SDS_Flag=True):
        # --初始化--
        X_ = np.zeros((self.n_init, len(self.bounds)))#  
        for j in range(len(self.bounds)):
            lower, upper = self.bounds[j]
            X_[:, j] =  [np.random.uniform(lower, upper) for _ in range(self.n_init)]
        self.y_train = np.array([self.objectFun(x) for x in X_]).reshape(-1,1)
        
        self.X_train = self.scaler.transform(X_)# X数据做归一化
        x_next2 = self.findLargeNB_init() # 

        # --迭代优化开始--
        acq_value_list=[]
        mu_flag=False
        n_Converge = int((1-self.converge_rate)*self.n_iter)
        best_value = -1e5
        for i in range(self.n_iter):
            # 对函数值做归一化
            s = MinMaxScaler(feature_range=(0,1))
            self.y_train = s.fit_transform(self.y_train)

            # 利用归一化的数据做高斯过程回归
            self.gpr.fit(self.X_train, self.y_train)
            
            # 预测荐点并存储
            x_next,acq_value = self.predict(s=s,step=i, y_train=self.y_train,acq_mu_Flag=mu_flag)
            x_next2 = self.findLargeNB(x_next)
    

            # 存储X值
            self.X_train = np.vstack((self.X_train, x_next))
            # 计算并存储y值
            self.y_train = s.inverse_transform(self.y_train) # 对函数值做反归一化
            x = self.scaler.inverse_transform(np.reshape(x_next,(-1,len(self.bounds))))
            
            y_next = self.objectFun(x.squeeze())
            self.y_train = np.vstack((self.y_train, y_next))


            if SDS_Flag:
                self.findLargeNB(x_next2) # 更新超球体数据，但不做预测
                self.X_train = np.vstack((self.X_train, x_next2))

                x2 = self.scaler.inverse_transform(np.reshape(x_next2,(-1,len(self.bounds))))
                y_next2 = self.objectFun(x2.squeeze())
                self.y_train = np.vstack((self.y_train, y_next2))

            acq_value_list.append(acq_value)
            acq_ACC = np.sum(acq_value_list[-10:])
            if len(acq_value_list)>10 and acq_ACC<10:
                mu_flag=True
            # 如果到了指定步数，开始采用均值收敛算法
            if i>n_Converge:
                mu_flag=True

            if y_next > best_value:
                best_value = y_next
                if SDS_Flag:
                    if y_next2 > best_value:
                        best_value = y_next

                RED = '\033[91m'
                END = '\033[0m'
                print(RED + f'{(i+1)*2+self.n_init}:best:{best_value}' + END)


        # 输出最优化结果
        best_idx = np.argmax(self.y_train)
        self.best_params = self.X_train[best_idx]
        self.best_value = self.y_train[best_idx]

        print("Best parameters:",self.best_params)
        print("Best value",self.best_value)

    def save(self):
        if not os.path.exists(self.path + 'bo\\'):
            os.mkdir(self.path + 'bo\\')

        h = h5py.File(name=self.path + 'bo\\boResult.h5',mode='w')
        h.create_dataset(data=self.X_train,name='X')
        h.create_dataset(data=self.y_train,name='y')
        h.create_dataset(data=self.best_params,name='best_X')
        h.create_dataset(data=self.best_value, name='best_y')
        h.close()


if __name__ == '__main__':
    # 定义目标函数 f
    def f1(params):
        x=params
        return -(x[0]**2 + x[1]**2 + x[2]**2)

    def f2(x0,x1,x2):
        return -(x0**2 + x1**2 + x2**2)
    
    def f3(x):
        return np.cos(2*np.pi*x)*np.exp(-0.5*np.abs(x))


    # 参数搜索空间
    space1 = [(-5., 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    space3 = [(-5., 5.0)]

    bo = bayesianOpt(bounds=space3, objectFun=f3,n_init=2,n_iter=5,converge_rate=0.2,
                     path='E:\\experiment\\')
    bo.runOpt(SDS_Flag=True)
    bo.save()


    
