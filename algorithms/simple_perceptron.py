
import sys
import os
sys.path.append("..")
from utils import ml_arg
from utils import get_data
import numpy as np
def simple_perceptron(X,label,X_test=None,label_test=None,param={'eta':[0.01],"maxium_epoch":10,'decay':False,'weights_return':False,'margin':[0]}):
    #X=X.toarray()
   # eta=[0.1],maxium_epoch=10,decay=None,weights_return=False
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
    w_best=[]
    step_list=[]

    time_step=0
    W=np.random.uniform(-0.01,0.01,X.shape[1])
    b=np.random.uniform(-0.01,0.01,(1))
    for j in range(maxium_epoch):
        lr=eta_origin
                #print("current lea",lr)
        if decay ==True:
                d=eta_origin/(1+j)

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
                W=W+x*l*lr
                #print(lr)
                b=b+l*lr

        wrong_ind=np.where(label*(X@W+b)<=0)
        train_acc=1-len(wrong_ind[0])/X.shape[0]
        epoch_train_acc.append(train_acc)
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

    if X_test is not None:
        
        return train_acc_list,test_acc_list,step_list,epoch_train_acc

    return train_acc_list
if __name__=="__main__":
        arg=ml_arg()
        argg=arg.parse_args()
        print(argg)
        data=get_data(argg)
        simple_perceptron(data.X_train,data.label_train,data.X_test,data.label_test,param=argg)