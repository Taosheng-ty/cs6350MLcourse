import sys
sys.path.append("..")
import os
import numpy as np
from utils import ml_arg
from utils import get_data
from utils import Logger
from argparse import Namespace
import numpy as np
from datetime import datetime
import math
from utils import calculate_F1
import pandas as pd
inf= math.inf
def cal_entropy(label):
        pois=np.where(label==1)
        edi=np.where(label==-1)
        tag_num=label.shape[0]
        if len(pois[0])*len(edi[0])*tag_num==0:
            return 0
        pois_num=len(pois[0])/tag_num   
        #print(pois_num)
        edi_num=len(edi[0])/tag_num
        prob=np.array([pois_num,edi_num])
        return np.sum(-prob*np.log2(prob),0)
def cal_entropy_gain(data,label,attribute,parse):
    entropy_orign=cal_entropy(label)
    entropy_after=0
    number_tatal=label.shape[0]

    for i in range(parse.n_interve):
        
        data_subset,label_subset=get_subset(data,label,attribute, i,parse)
        number_attr=label_subset.shape[0]
#         print(label_subset.shape)
        entropy=cal_entropy(label_subset)
#         print(entropy)
        if number_tatal==0:
            result=0
        else:
            result=entropy*number_attr/number_tatal
        entropy_after=entropy_after+result

    return entropy_orign-entropy_after
def Entroy_tree(data,label,parse):
    entropy_orign=cal_entropy(label)
    entropy_diffmax=0
    entropy_diffind=0
    tag_num=label.shape[0]
    attributes=data.shape[1]
    most_common=1 if label.mean()>0 else -1
    #print(most_common)
    for attribute in range(attributes):
        current_gain=cal_entropy_gain(data,label,attribute,parse)
        if current_gain>entropy_diffmax:
            entropy_diffmax=current_gain
            entropy_diffind=attribute 
    if entropy_diffmax==0:
        entropy_diffind=most_common
    return entropy_diffind,entropy_diffmax
def get_subset(data,label,entropy_diffind,attribute,parse):
        
    data_subset_ind =np.where(data[:,entropy_diffind]==attribute)
    
    return data[data_subset_ind],label[data_subset_ind]
def itera_tree(X,label,X_test=None,label_test=None,X_val=None,label_val=None,param=None,num_iter=0,parse=None,string_=[],tree_decision=list()):
    parse=param
    if parse.binary==1:
        parse.n_interve=2
    if hasattr(parse,"n_interve"):
        parse.n_interve=parse.n_interve    
    maxi_iter=parse.tree_depth
    if num_iter==0:
        tree_decision=list()
    num_iter1=num_iter+1
    string=string_[:]    
#     print(X.shape,"X.shape")
    entropy_diffind,entropy_diffmax=Entroy_tree(X,label,parse)
#     print(entropy_diffind,entropy_diffmax,"entropy_diffind,entropy_diffmax")
    tag_num=label.shape[0]
    most_common=1 if label.mean()>0 else -1
#     print(parse)
    if entropy_diffmax ==0:
        
        string=string+["label"]+[most_common]
        tree_decision.append(string)
        return tree_decision

    for i in range(parse.n_interve):
            string_1=string[:] 
            
            data_subset,label_subset=get_subset(X,label,entropy_diffind, i,parse)
#             print(i,"i",np.bincount(label_subset+1))
#             print(x,"x")
            tag_num1=label_subset.shape[0]

            if tag_num1==0:

                string_2=string_1+[entropy_diffind]+[i]+["label"]+[most_common]
                tree_decision.append(string_2)                
                continue
            if num_iter1>=parse.tree_depth:

                most_common_sub=1 if label_subset.mean()>0 else -1
                string_2=string_1+[entropy_diffind]+[i]+["label"]+[most_common_sub]
                tree_decision.append(string_2)
                continue
            string_2=string_1+[entropy_diffind]+[i]
            tree_decision=itera_tree(data_subset,label_subset,\
                                     string_=string_2,num_iter=num_iter1,tree_decision=tree_decision,param=parse)
     
    return    tree_decision
def return_label(value,parse):
    ra=np.linspace(-1,1,parse.n_interve)
    ra[0]=-inf
    ra[-1]=inf
    ind=np.where(value>ra)
#     print(ind[0][-1],"ind[0][-1]")
    return ind[0][-1]
    
def prediction(tree_dec_np,test,i=0,parse=None):
    
    attri=tree_dec_np[0,i]
    #print(i)
#     print(parse,"parse")
    #print(attri)
    if attri=="label":
        #print(tree_dec_np)
        #print(tree_dec_np[0,i+1])
        label=tree_dec_np[0,i+1]
        return label
    ind_test=i+1

    ind_match=np.where(tree_dec_np[:,i+1]==test[attri])
    
    tree_dec_np_prun=tree_dec_np[ind_match]

    label=prediction(tree_dec_np_prun,test,i=i+2,parse=parse)
    return label
def conver2numpy(tree_dec):
    
    
    #b = np.array([len(a),len(max(a,key = lambda x: len(x)))])
    g=len(max(tree_dec,key = lambda x: len(x)))+2
    b =  [[ None for y in range( g ) ] for x in range(len(tree_dec))]
    for i,j in enumerate(tree_dec):
        b[i][0:len(j)] = j
    return np.array(b)

def batch_predic(X,label,tree_dec_np_entro,parse):
    N=X.shape[0]
    array_pre=np.zeros(N)
    ind_actual=np.where(label==1)[0]
    ind_pred=[]
    predict=[]
    for i in range(N):
        test=X[i,:]
#         print(label[i],"label[i]")
        label_pre=prediction(tree_dec_np_entro,test,0,parse)
#         print(label_pre,"label_pre")
        predict.append(1 if label_pre==1 else -1)
        if label_pre==1:
#             print(label_pre)
#             print("right")
            ind_pred.append(i)
        else:
            pass
    ind_pred=np.array(ind_pred)
    predict=np.array(predict)
    results={}
#     print(predict,label)
    acc=np.where(predict==label)[0]
    acc=acc.shape[0]/predict.shape[0]
    if parse.metrics=="acc":
         results["metrics"]=acc
    if parse.metrics=="F1_score":
        results["metrics"]= calculate_F1(ind_actual,ind_pred)
    results["prediction"]=predict
#     print(predict,"prediction")
    return results



class Decision_tree:
    def train(self,X,label,X_val=None,label_val=None,num_iter=0,\
              X_test=None,label_test=None,param=None,string_=[],tree_decision=list()):
       
        re=itera_tree(X,label,param=param)
        tree_dec_np=conver2numpy(re)
        results=batch_predic(X_val,label_val,tree_dec_np,param)
        
        results={"metrics":results["metrics"],"param":tree_dec_np}
        return results
    def predict(self,X,label,results,param):
        results=batch_predic(X,label,results["param"],param)
#         parse=param
#         N=X.shape[0]
#         tree_dec_np_entro=results["param"]
#         array_pre=np.zeros(N)
#         ind_actual=np.where(label==1)[0]
#         ind_pred=[]
#         predict=[]
#         for i in range(N):
#             test=X[i,:]
#             label_pre=prediction(tree_dec_np_entro,test,0,param)

#             predict.append(1 if label_pre==1 else -1)
#             if label_pre==1:
#     #             print(label_pre)
#     #             print("right")
#                 ind_pred.append(i)
#             else:
#                 pass
#         ind_pred=np.array(ind_pred)
#         predict=np.array(predict)
#         results={}
#         acc=np.where(predict==label)[0]
#         acc=acc.shape[0]/predict.shape[0]
#         if parse.metrics=="acc":
#              results["metrics"]=acc
#         if parse.metrics=="F1_score":
#             results["metrics"]= calculate_F1(ind_actual,ind_pred)
#         results["prediction"]=predict
    #     print(predict,"prediction")
        return results


if __name__=="__main__":
        arg=ml_arg()
        argg=arg.parse_args()
        
        data=get_data(argg)
        directory=argg.log_file+"current/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        sys.stdout=Logger(directory+"log.txt")
        print(argg)
        re=itera_tree(data.X_train,data.label_train,parse=argg)
        tree_dec_np=conver2numpy(re)

#         print(tree_dec_np)
        results=batch_predic(data.X_val,data.label_val,tree_dec_np,argg)
        print(results,"val")
        data_xl=pd.DataFrame.from_dict(results["prediction"])
        data_xl.to_csv(directory+"log"+str(datetime.now())+".csv")
        param=argg
        results=batch_predic(data.X_train,data.label_train,tree_dec_np,argg)
        print(results,"train")
        results=batch_predic(data.X_test,data.label_test,tree_dec_np,argg)
        print(results,"test")
        os.rename(directory,param.log_file+"tree_depth "+str(param.tree_depth)+"n_interve"+str(param.n_interve)+"precision"+str(results["metrics"])+" metircs"+param.metrics+"    "+str(datetime.now()))
