import os
import sys
sys.path.append("..")
from utils import Logger
from algorithms import *
from scipy.sparse import csr_matrix
import numpy as np

from argparse import Namespace
from utils import ml_arg
from utils import get_data
def SVM(X,label,X_val=None,label_val=None,X_test=None,label_test=None,param=None):
    
#     directory=param.log_file+"current"+str(datetime.now())+"/"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     sys.stdout=Logger(directory+"log.txt")
#     print(directory)
#     print(param)
    eta1=param.eta_1
    C=param.C
    maxium_epoch=param.maxium_epoch
    decay=param.decay
#     weights_return=param["weights_return
#     print(X.shape,X_test.shape)
    eta_origin=eta1
    np.random.seed(2)
   # label_expand=label[:,np.newaxis]

    epoch_train_acc=[]
    time_step=0.0
    train_acc_list=[]
    test_acc_list=[]
    val_acc_list=[]
    w_best=[]
    step_list=[]
    best_val=0.0
    time_step=0
    W=np.random.normal(0,0.001,X.shape[1])
    b=np.random.normal(0,0.001,(1))
    W_best=W
    b_best=b
    d=1
    for j in range(maxium_epoch):
        lr=eta_origin
                #print("current lea",lr)
        if decay ==True:
                d=1/(1+j)
        ind=np.arange(X.shape[0])
#         print(ind)
        np.random.shuffle(ind)
        for hh in range(X.shape[0]):
            k=ind[hh]    
            x=X[k,:]        
            l=label[k]
#             print(l*(x@W+b))
#             print(W.mean())
            if l*(x@W+b)<=1:
                    
                    lr=eta1
                    W=(1-lr)*W+x*l*lr*d*C
                    b=(1-lr)*b+l*lr*d*C
#                     print(b)
            else:
                    #print("neg")
                    W=(1-lr)*W
                    b=(1-lr)*b
#                 time_step=time_step+1
#                 #print("learnign",lr)
#                # print(k,"here is th wrong one")
#                 W=W+x*l*lr*d
#                 #print(lr)
#                 b=b+l*lr*d

        
        train_acc=calculate_metrics(X,label,W,b,param)
#         print("epoch: #",str(j)," current "+ param.metrics +" is",str(train_acc))
        epoch_train_acc.append(train_acc)
        if j %param.valid_each==0:
            
            val_acc=calculate_metrics(X_val,label_val,W,b,param)
#             print("########################/n epoch: #",str(j)," current validating "+ param.metrics+"is",str(val_acc)+"########################/n ")
            if val_acc>best_val:
                best_val=val_acc
                W_best=W
                b_best=b                
#                 np.savez(directory+"epoch #"+str(j)+param.metrics+"="+str(val_acc)+'.npz', w=W, b=b)
#                 print("found better one, checkpoint saved ")
            val_acc_list.append(val_acc)            
            
            
    #step_list=step_list.append(time_step)
#     test_acc=calculate_metrics(X_test,label_test,W_best,b_best,param)
#     print("the final test "+param.metrics+" is ",test_acc)
#     os.rename(directory,param.log_file+"best"+ param.metrics+"+"+str(best_val)+"  "+str(datetime.now()))
    results={"metrics":best_val,"W":W_best,"b":b_best}
    return results
if __name__=="__main__":
    arg=ml_arg()
#     argg=arg.parse_args(["--maxium_epoch",' 10',"-e1","1","-C"," 1","--metrics","acc","--norm"])
    argg=arg.parse_args(["--train_data", "/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_train", "--val_data","/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_test","--json_file","./setting.json","-e1",".1","-C"," .1","--maxium_epoch",' 20',"--norm","--metrics","acc"])
    data=get_data(argg)
    SVM(data.X_train,data.label_train,data.X_val,data.label_val,data.X_val,data.label_val,param=argg)