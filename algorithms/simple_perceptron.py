
import sys
import os
sys.path.append("..")
from utils import ml_arg
from utils import get_data
from utils import Logger
import numpy as np
from datetime import datetime
def simple_perceptron(X,label,X_test=None,label_test=None,X_val=None,label_val=None,param=None):
    
    
    directory=param.log_file+"current/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    sys.stdout=Logger(directory+"log.txt")
    print(directory)
    print(param)
    eta1=param.eta_1
    eta2=param.eta_2
    maxium_epoch=param.maxium_epoch
    decay=param.decay
#     weights_return=param["weights_return
    print(X.shape,X_test.shape)
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
    W=np.random.uniform(-0.01,0.01,X.shape[1])
    b=np.random.uniform(-0.01,0.01,(1))
    d=1
    for j in range(maxium_epoch):
        lr=eta_origin
                #print("current lea",lr)
        if decay ==True:
                d=1/(1+j)

        for k in range(X.shape[0]):

            x=X[k,:]        
            l=label[k]

            if l*(x@W+b)<=0:
                if l>0:
#                         print("pos")
                    lr=eta1
                else:
                    #print("neg")
                    lr=eta2
                time_step=time_step+1
                #print("learnign",lr)
               # print(k,"here is th wrong one")
                W=W+x*l*lr*d
                #print(lr)
                b=b+l*lr*d

        wrong_ind=np.where(label*(X@W+b)<=0)
        train_acc=1-len(wrong_ind[0])/X.shape[0]
        print("epoch: #",str(j)," current training accuracy is",str(train_acc))
        epoch_train_acc.append(train_acc)
        if j %param.valid_each==0:
            wrong_ind=np.where(label_val*(X_val@W+b)<=0)
            val_acc=1-len(wrong_ind[0])/X_val.shape[0]
            print("########################/n epoch: #",str(j)," current validate accuracy is",str(val_acc)+"########################/n ")
            if val_acc>best_val:
                best_val=val_acc
                np.savez(directory+"epoch #"+str(j)+" val_acc="+str(val_acc)+'.npz', w=W, b=b)
                print("found better one")
            val_acc_list.append(val_acc)            
            
            
    #step_list=step_list.append(time_step)
    wrong_ind=np.where(label*(X@W+b)<=0)
    train_acc=1-len(wrong_ind[0])/X.shape[0]
    step_list.append(time_step)
    print(train_acc)
    
    if X_test is not None:
        wrong_ind=np.where(label_test*(X_test@W+b)<=0)
        test_acc=1-len(wrong_ind[0])/X_test.shape[0]
        test_acc_list.append(test_acc)
    train_acc_list.append(train_acc)
    test_acc_list=np.array(test_acc_list)
    train_acc_list=np.array(train_acc_list)
    step_list=np.array(step_list)
    epoch_train_acc=np.array(epoch_train_acc)
    os.rename(directory,param.log_file+"best val accuracy="+str(best_val)+"  "+str(datetime.now()))
    if X_test is not None:
        
        return train_acc_list,test_acc_list,step_list,epoch_train_acc

    return train_acc_list
if __name__=="__main__":
        arg=ml_arg()
        argg=arg.parse_args()
        print(argg)
        data=get_data(argg)
        simple_perceptron(data.X_train,data.label_train,data.X_test,data.label_test,data.X_val,data.label_val,param=argg)