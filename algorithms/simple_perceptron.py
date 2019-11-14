
import sys
import os
sys.path.append("..")
from utils import ml_arg
from utils import get_data
from utils import Logger
import numpy as np
from datetime import datetime
def calculate_metrics(x,label,W,b,param):
    if param.metrics=="acc":
        wrong_ind=np.where(label*(x@W+b)<=0)
        val_acc=1-len(wrong_ind[0])/x.shape[0]
        return val_acc
    if param.metrics=="F1_score":
        ind_actual=np.where(label>=0)
        ind_pred=np.where(x@W+b>=0)
        ind_actual_set=set(ind_actual[0])
        ind_pred_set=set(ind_pred[0])
        cross=list(ind_actual_set.intersection(ind_pred_set))
        tp=len(cross)
        fn=ind_actual[0].shape[0]
        fp=ind_pred[0].shape[0]
        if fp*fn*tp==0:
            return 0
        p=tp/(fp)
        r=tp/(fn)
        f1_score=2*p*r/(p+r)
        return f1_score
def calculate_metrics_prediction(x,label,results,param):
    W=results["W"]
    b=results["b"]
    prediction={}
    predict_label=np.ones_like(label)
    id_neg=np.where((x@W+b)<=0)
    predict_label[id_neg]=0
    prediction["prediction"]=predict_label
    if param.metrics=="acc":
        wrong_ind=np.where(label*(x@W+b)<=0)
        val_acc=1-len(wrong_ind[0])/x.shape[0]
        prediction["metrics"]=val_acc
    if param.metrics=="F1_score":
        ind_actual=np.where(label>=0)
        ind_pred=np.where(x@W+b>=0)
        ind_actual_set=set(ind_actual[0])
        ind_pred_set=set(ind_pred[0])
        cross=list(ind_actual_set.intersection(ind_pred_set))
        tp=len(cross)
        fn=ind_actual[0].shape[0]
        fp=ind_pred[0].shape[0]
        f1_score=0
        if fp*fn!=0:
            p=tp/(fp)
            r=tp/(fn)
            f1_score=2*p*r/(p+r)
        prediction["metrics"]=f1_score
    return prediction
def calculate_F1(ind_actual,ind_pred):
#         print(ind_actual,ind_pred)
        ind_actual_set=set(ind_actual)
        ind_pred_set=set(ind_pred)
#         print(ind_actual_set,ind_pred_set)
        cross=list(ind_actual_set.intersection(ind_pred_set))
        tp=len(cross)
        fn=ind_actual.shape[0]
        fp=ind_pred.shape[0]
        if fp*fn==0:
            print(fp,fn,"this is fp and fn")
            return 0
        p=tp/(fp)
        r=tp/(fn)
        f1_score=2*p*r/(p+r)
        return f1_score          
def simple_perceptron(X,label,X_test=None,label_test=None,X_val=None,label_val=None,param=None):
    
    
    directory=param.log_file+"current/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    sys.stdout=Logger(directory+"log.txt")
#     print(directory)
#     print(param)
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
    W_best=W
    b_best=b
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

        
        train_acc=calculate_metrics(X,label,W,b,param)
        print("epoch: #",str(j)," current "+ param.metrics +" is",str(train_acc))
        epoch_train_acc.append(train_acc)
        if j %param.valid_each==0:
            
            val_acc=calculate_metrics(X_val,label_val,W,b,param)
            print("########################/n epoch: #",str(j)," current validating "+ param.metrics+"is",str(val_acc)+"########################/n ")
            if val_acc>best_val:
                best_val=val_acc
                W_best=W
                b_best=b                
                np.savez(directory+"epoch #"+str(j)+param.metrics+"="+str(val_acc)+'.npz', w=W, b=b)
                print("found better one")
            val_acc_list.append(val_acc)            
            
            
    #step_list=step_list.append(time_step)
    test_acc=calculate_metrics(X_test,label_test,W_best,b_best,param)
    print("the final test "+param.metrics+" is ",test_acc)
    os.rename(directory,param.log_file+"best"+ param.metrics+"+"+str(best_val)+"  "+str(datetime.now()))
def average_perceptron(X,label,X_val=None,label_val=None,param=None):
    
    
    directory=param.log_file+"current/"
    if not os.path.exists(directory):
        os.makedirs(directory)

#     print(directory)
#     print(param)
    eta1=param.eta_1
    eta2=param.eta_2
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

        for k in range(X.shape[0]):

            x=X[k,:]        
            l=label[k]
            running_W=0.99*running_W+W*0.01
            running_b=b*0.01+0.99*running_b
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

        
        train_acc=calculate_metrics(X,label,W,b,param)
        print("epoch: #",str(j)," current for training"+ param.metrics +" is",str(train_acc))
# epoch_train_acc.append(train_acc)
        if j %param.valid_each==0:
            val_acc=calculate_metrics(X_val,label_val,running_W,running_b,param)
            print("########################/n epoch: #",str(j)," current validating "+ param.metrics+"is",str(val_acc)+"########################/n ")
            if val_acc>best_val:
                best_val=val_acc
                W_best=running_W
                b_best=running_b              
#                 np.savez(directory+"epoch #"+str(j)+param.metrics+"="+str(val_acc)+'.npz', w=W, b=b)
#                 print("found better one, when at epoch",str(j),)
#             val_acc_list.append(val_acc)  
    results={"W":W_best,"b": b_best , "metrics":best_val  }
    return   results   
                   
    #step_list=step_list.append(time_step)
#     test_acc=calculate_metrics(X_test,label_test,W_best,b_best,param)
#     print("the final test "+param.metrics+" is ",test_acc)
#     os.rename(directory,param.log_file+"best"+ param.metrics+"+"+str(best_val)+"  "+str(datetime.now()))
if __name__=="__main__":
        arg=ml_arg()
        argg=arg.parse_args()
        print(argg)
        data=get_data(argg)
        simple_perceptron(data.X_train,data.label_train,data.X_test,data.label_test,data.X_val,data.label_val,param=argg)