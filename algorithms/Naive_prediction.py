import os
import sys
sys.path.append("..")
from utils import Logger
from algorithms import *
from scipy.sparse import csr_matrix
import numpy as np
def Naive_prediction(X,label,results,param=None):
    Naive_param_np =results['posterior']
    prior =results['prior']
#     print(prior ,"prior ")
    n=np.arange(X.shape[1])
    nn=np.repeat(n,X.shape[0],0)
    X_1=X.astype(int)
#     print(Naive_param_np[0:4,0:10],"Naive_param_np")
#     print(X_1[0:10,0:10],"X_1")
    pos=Naive_param_np[X_1,n]
    neg=Naive_param_np[X_1+2,n]
#     print(pos[0:10,0:10],"pos")
    pos_prob=np.sum(np.log(pos),1)+np.sum(np.log(prior[0]))
    neg_prob=np.sum(np.log(neg),1)+np.sum(np.log(prior[1]))
    prediction=-1*np.ones(X.shape[0])
#     calculate_metrics_prediction()
    pos_ind=np.where(pos_prob>neg_prob)
    prediction[pos_ind]=1
    ratio=np.where(prediction==label)[0].shape[0]/X.shape[0]
    if param.metrics=="F1_score":
        ind_label=np.where(label==1)[0]
        ind_prediction=np.where(prediction==1)[0]
        ratio=calculate_F1(ind_label,ind_prediction)
#     print(prediction[:20],"prediction")
#     print(label[:20],"label")
    final_results={"metrics":ratio,"prediction":prediction}
    return final_results
    
    
    
    
def Naive_bayes(X,label,X_val=None,label_val=None,X_test=None,label_test=None,param=None,lambda_para=1):
    ind_p=np.where(label==1)
    ind_n=np.where(label==-1)
    prior_p=ind_p[0].shape[0]/X.shape[0]
    prior_n=ind_n[0].shape[0]/X.shape[0]
    
    Naive_param_np=np.zeros((4,X.shape[1]))
    for j in  range(X.shape[1]):
        ind_feature_pp=np.where(X[ind_p,j]==1)[0]
        ind_feature_pn=np.where(X[ind_p,j]==0)[0]
        ind_feature_np=np.where(X[ind_n,j]==1)[0]
        ind_feature_nn=np.where(X[ind_n,j]==0)[0]
        Naive_param_np[0,j]=(ind_feature_pn.shape[0]+param.smooth_term)/(ind_p[0].shape[0]+2)
        Naive_param_np[1,j]=(ind_feature_pp.shape[0]+param.smooth_term)/(ind_p[0].shape[0]+2)
        Naive_param_np[2,j]=(ind_feature_nn.shape[0]+param.smooth_term)/(ind_n[0].shape[0]+2)
        Naive_param_np[3,j]=(ind_feature_np.shape[0]+param.smooth_term)/(ind_n[0].shape[0]+2)
    directory=param.log_file+"current"+str(datetime.now())+"/"
#     print(Naive_param_np,"print(Naive_param_np)")
#     print(Naive_param_np.shape,"Naive_param_np.shape")
    if not os.path.exists(directory):
        os.makedirs(directory)
    sys.stdout=Logger(directory+"log.txt")
    results={}
#     results={"W":W_best,"b": b_best , "metrics":best_val  }
    results={"posterior":Naive_param_np ,"prior":[prior_p,prior_n],}
    final=Naive_prediction(X_val,label_val,results,param)
    results["metrics"]=final["metrics"]
#     print("the final validating "+param.metrics+" is ",ratio)
#     os.rename(directory,param.log_file+"best"+ param.metrics+"+"+str(ratio)+"  "+str(datetime.now()))
    return results

if __name__=="__main__":
    arg=ml_arg()
#     argg=arg.parse_args(["--maxium_epoch",' 10',"-e1","1","-C"," 1","--metrics","acc","--norm"])
    argg=arg.parse_args(["--train_data", "/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_train", "--val_data","/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_test","--json_file","./setting.json","--maxium_epoch",' 10'])
    data=get_data(argg)
    Naive_bayes(data.X_train,data.label_train,data.X_val,data.label_val,data.X_val,data.label_val,param=argg)