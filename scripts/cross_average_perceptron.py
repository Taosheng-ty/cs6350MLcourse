import sys
sys.path.append("../../cs6350MLcourse/")
from algorithms import *
from algorithms.simple_perceptron import *
from utils import *
import json
import numpy as np
import os
import pandas as pd
arg=ml_arg()
argg=arg.parse_args()
# argg=arg.parse_args(["--tree_depth","3","--train_data", "datasets/data/data-splits/data.train","--val_data", "datasets/data/data-splits/data.eval.anon", "--test_data","datasets/data/data-splits/data.test","--json_file","./setting.json"])
print(argg)
data=get_data(argg)
setting=json.load(open(argg.json_file))
print(setting)
directory=argg.log_file+"current_average_perceptron"+str(datetime.now())+"/"
if not os.path.exists(directory):
    os.makedirs(directory)
sys.stdout=Logger(directory+"log.txt")
print(argg)
print("*********************************************** this is average perceptron*************************")
[lr1_mi,lr1_ma]=setting['learning_rate1'][0],setting['learning_rate1'][1]
[lr2_mi,lr2_ma]=setting['learning_rate2'][0],setting['learning_rate2'][1]
par_1=setting['learning_rate1_partition']
par_2=setting['learning_rate2_partition']
lr1_array=np.linspace(lr1_mi,lr1_ma,par_1)
lr2_array=np.linspace(lr2_mi,lr2_ma,par_2)
gene=data_loader(data.X_train,data.label_train,setting['fold_num']) 

argg.maxium_epoch=setting["max_epoch"]


best_tree=None
best_f1=0
best_lr1=0
best_lr2=0
peformance={}
peformance["learning_rate1"]=[]
for l in  np.nditer(lr2_array):
    j=10**l
    peformance["learning_rate2="+str(j)]=[]
for k in np.nditer(lr1_array):
    
    i=10**k
    peformance["learning_rate1"].append(i)

for k in np.nditer(lr1_array):
    
    i=10**k
#     peformance["learning_rate1"].append(i)
    for l in  np.nditer(lr2_array):
        j=10**l
#         peformance["learning_rate2="+str(j)]=[]
        per_ech=[]
        for k in range(setting['fold_num']):
            train_x,train_label,val_x,val_label=next(gene)
#             print(train_label.shape,val_label.shape,)
            argg.eta_1=i
            argg.eta_2=j
            results_val=average_perceptron(train_x,train_label,val_x,val_label,param=argg)
#             results_train=batch_predic(data.X_train,data.label_train,tree_dec_np,argg)
#             results_test=batch_predic(data.X_test,data.label_test,tree_dec_np,argg)          
            per_ech.append(results_val["metrics"])
#             print(argg.metrics+" as "+str(results_train["metrics"])+"for trianing when depth is "+str(i)+" and interve is "+str(j)+" at fold "+str(k))
            print(argg.metrics+" as "+str(results_val["metrics"])+" for validating when lr1 is "+str(i)+" and lr2 is "+str(j)+" at fold "+str(k))
        per_ech=np.array(per_ech) 
        mean=per_ech.mean()
        std=per_ech.std()
        print("validating average "+argg.metrics+" as "+str(mean)+" when lr1  is "+str(i)+" and lr2  is "+str(j))
        if  mean>=best_f1:
            best_lr1=i
            best_lr2=j
            best_f1=mean
            best_tree=results_val
            print("Found a better tree "+"with "+argg.metrics+" as "+str(best_f1)+" when lr1 is "+str(i)+" and lr2 is "+str(j))
            print("save it in the directory "+directory+argg.metrics+" as "+str(best_f1)+" when lr1 is "+str(i)+" and lr2 is "+str(j)+'.npz')
            np.savez(directory+argg.metrics+" as "+str(best_f1)+" when lr1 is "+str(i)+" and lr2 is "+str(j)+'.npz',w=results_val["W"],b=results_val["b"])
            results_eval=calculate_metrics_prediction(data.X_val,data.label_val,best_tree,argg)
            data_xl=pd.DataFrame.from_dict(results_eval["prediction"])
                              
            data_xl.to_csv(directory+"evalua_for_best_lr1_"+str(best_lr1)+"_and_lr2"+str(best_lr2)+argg.metrics+str(best_f1)+str(datetime.now())+".csv")
        peformance["learning_rate2="+str(j)].append(np.around([mean,std],decimals=3))
#         print(np.around([mean,std],decimals=3))
print("best_lr1 is",best_lr1," best_lr2" ,best_lr2, "which we can get " ,argg.metrics+" as "+str(best_f1)+"for validating")
argg.eta_1=best_lr1
argg.eta_2=best_lr2
results_test=calculate_metrics_prediction(data.X_test,data.label_test,best_tree,argg)
print("the final  performance on test data for ",argg.metrics+" is "+str(results_test["metrics"]))
results_test=calculate_metrics_prediction(data.X_train,data.label_train,best_tree,argg)
print("the final  performance on training data for ",argg.metrics+" is "+str(results_test["metrics"]))
print(peformance)
data_xl=pd.DataFrame.from_dict(peformance)
data_xl.to_csv(directory+"mean_std_for_corss_validate"+str(datetime.now())+".csv")
param=argg
results_eval=calculate_metrics_prediction(data.X_val,data.label_val,best_tree,argg)
data_xl=pd.DataFrame.from_dict(results_eval["prediction"])
data_xl.to_csv(directory+"evaluation_for_best_lr1_"+str(best_lr1)+"_and_lr2"+str(best_lr2)+argg.metrics+str(best_f1)+str(datetime.now())+".csv")
os.rename(directory,param.log_file+" cross_validate_lr1 "+str(setting['learning_rate1'])+"n_lr2"+str(setting['learning_rate2'])+"precision"+str(best_f1)+" metircs"+param.metrics+"    "+str(datetime.now()))