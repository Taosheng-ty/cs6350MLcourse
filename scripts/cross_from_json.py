import sys
sys.path.append("../../cs6350MLcourse/")
from algorithms import *
from algorithms.Naive_prediction import *
from utils import *
import json
import numpy as np
import os
import pandas as pd
from utils import Logger
from argparse import Namespace
def get_cross_param(exp_settings):
    param_list=iteration(exp_settings,param=[])

    for j in param_list:
        name=exp_settings["validation"]
        cur_exp_settings=exp_settings.copy()
        for i in range(len(name)):
            cur_exp_settings[name[i]]=j[i]
        cur_exp_settings["current_param"]=j
        yield cur_exp_settings
        
        
def iteration(exp_settings,i=0,path=[],param=[]):
    name=exp_settings["validation"]
    current_name=name[i]
    param_sub=[]
#     print(current_name)
    for j in exp_settings[current_name]:
        if i==0:
            path=[]
        
        path_current=path.copy()
        path_current.append(j)
        j=i+1
        if j>=len(name):
            param.append(path_current)
            pass
            
        else:           
            iteration(exp_settings,j,path_current,param)
    return param        
arg=ml_arg()
argg=arg.parse_args()
exp_settings=json.load(open(argg.json_file))

algorithm=create_object(exp_settings['learning_algorithm'])
argg=Namespace(**exp_settings)   

data=get_data(argg)

directory=argg.log_file+"current_"+argg.learning_algorithm+str(datetime.now())+"/"
if not os.path.exists(directory):
    os.makedirs(directory)
sys.stdout=Logger(directory+"log.txt")
print(argg)
gene=data_loader(data.X_train,data.label_train,exp_settings['fold_num']) 
cross_param=get_cross_param(exp_settings)
best_tree=None
best_f1=0
best_cross_param=0
best_C_trdeoff=0
peformance={}
peformance["learning_rate1"]=[]
best_metrics=0

best_param_list=None   
cross_results={}
while True:
    try:
        current_param=next(cross_param)
        
        param_list=current_param["current_param"]
        argg=Namespace(**current_param)
    except:
        break

    per_ech=[]
    for m in range(exp_settings['fold_num']):
        train_x,train_label,val_x,val_label=next(gene)


        results_val=algorithm.train(train_x,train_label,val_x,val_label,param=argg)
       
        per_ech.append(results_val["metrics"])
    per_ech=np.array(per_ech) 
    mean=per_ech.mean()
    std=per_ech.std()
    cross_results[str(param_list)]=[mean,std]
    print("validating average "+argg.metrics+" as "+str(mean)+" when "+"validaton term are "+\
          str(exp_settings["validation"])+str(param_list[:]))
    if  mean>best_metrics:
        best_cross_param=argg
        best_metrics=mean
        best_param_list=param_list
        best_learner=results_val
        print("Found a better tree "+"with "+argg.metrics+" as "+str(best_metrics)+" when "+\
              str(exp_settings["validation"])+"is "+str(best_param_list))
        print("save it as checkpoint in the directory "+directory+argg.metrics+" as "+str(best_metrics)+" when "+str(exp_settings["validation"])+"is "+str(best_param_list)+'.npz')
        np.savez(directory+argg.metrics+" as "+str(best_metrics)+" when "+str(exp_settings["validation"])+"is "+str(best_param_list)+'.npz',np.array(best_learner["param"]))
                              

print("best_param" ,best_param_list,"for "+str(exp_settings["validation"]), "which we can get " ,argg.metrics+" as "+str(best_metrics)+"for validating")
final_results=pd.DataFrame.from_dict(cross_results)
final_results.to_csv(directory+"cross_validation_best metrics"+str(best_metrics)+".csv")



results_eval=algorithm.predict(data.X_train,data.label_train,best_learner,best_cross_param)
print("*******************************final training performance is ",results_eval["metrics"],"******************************")

results_eval=algorithm.predict(data.X_test,data.label_test,best_learner,best_cross_param)
print("*******************************final test performance is ",results_eval["metrics"],"******************************")

neg_label=exp_settings["neg_label"]

prediction=results_eval["prediction"]
if neg_label==0:
    ind_neg=np.where(prediction==-1)
    prediction[ind_neg]=0
if neg_label==-1:
    ind_neg=np.where(prediction==0)
    prediction[ind_neg]=-1
data_xl=pd.DataFrame.from_dict(prediction)
data_xl.to_csv(directory+"finl_prediction_best metrics"+str(best_metrics)+".csv")