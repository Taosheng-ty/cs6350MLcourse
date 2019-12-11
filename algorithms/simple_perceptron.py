# 
import sys
import os
sys.path.append("..")
from utils import ml_arg
from utils import get_data
from utils import Logger
import numpy as np
from datetime import datetime
from utils import calculate_metrics_prediction,calculate_metrics
class Perceptron:
    def train(self,X,label,X_val=None,label_val=None,param=None):        
#         directory=param.log_file+"current/"
#         if not os.path.exists(directory):
#             os.makedirs(directory)
        eta1=param.eta_1
#         eta2=param.eta_2
        maxium_epoch=param.maxium_epoch
        decay=param.decay
        eta_origin=eta1
        np.random.seed(2)

        epoch_train_acc=[]
        time_step=0.0
        train_acc_list=[]
        test_acc_list=[]
        val_acc_list=[]
        w_best=[]
        step_list=[]
        best_val=0.0
        time_step=0
        W=np.random.uniform(-0.01,0.01,X.shape[1])
        b=np.random.uniform(-0.01,0.01,(1))
        W_best=W
        b_best=b
        d=1
        running_W=W
        running_b=b
        for j in range(maxium_epoch+1):
            lr=eta_origin
                    #print("current lea",lr)
            if decay ==True:
                    d=1/(1+j)
            ind=np.arange(X.shape[0])
    #         print(ind)
            np.random.shuffle(ind)
            for k in range(X.shape[0]):
                k=ind[k]
                x=X[k,:]        
                l=label[k]
                running_W=0.99*running_W+W*0.01
                running_b=b*0.01+0.99*running_b
#                 if l*(x@W+b)<=0: # we will do update all the time which is to improve acc
                if l*(x@W+b)<=0 :  
                    lr=eta1
                    if hasattr(param,"pos_weight") and l==1:
                        lr=eta1*param.pos_weight
                    time_step=time_step+1
                    #print("learnign",lr)
                   # print(k,"here is th wrong one")
                    W=W+x*l*lr*d
                    #print(lr)
                    b=b+l*lr*d
            train_acc=calculate_metrics(X,label,W,b,param)
            if param.verbose==1:
                print("epoch: #",str(j)," current for training"+ param.metrics +" is",str(train_acc))
    # epoch_train_acc.append(train_acc)
            if j %param.valid_each==0:
                val_acc=calculate_metrics(X_val,label_val,running_W,running_b,param)
                if param.verbose==1:
                    print("########################/n epoch: #",str(j)," current validating "+ param.metrics+"is",str(val_acc)+"########################/n ")
                if val_acc>best_val:
                    best_val=val_acc
                    W_best=running_W
                    b_best=running_b              

        results={"W":W_best,"b": b_best , "metrics":best_val,"param":[W_best,b_best]  }
        return   results   
    def predict(self,x,label,results,param):
        prediction=calculate_metrics_prediction(x,label,results,param)
        return prediction

if __name__=="__main__":
        arg=ml_arg()
        argg=arg.parse_args()
        print(argg)
        data=get_data(argg)
        simple_perceptron(data.X_train,data.label_train,data.X_test,data.label_test,data.X_val,data.label_val,param=argg)