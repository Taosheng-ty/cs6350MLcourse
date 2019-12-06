import sys
sys.path.append("../../cs6350MLcourse/")
from algorithms import *
from algorithms.SVM import *
from utils import *
import json
import numpy as np
import os
import pandas as pd
from utils import Logger
arg=ml_arg()
argg=arg.parse_args()
# argg=arg.parse_args(["--tree_depth","3","--train_data", "datasets/data/data-splits/data.train","--val_data", "datasets/data/data-splits/data.eval.anon", "--test_data","datasets/data/data-splits/data.test","--json_file","./setting.json"])

#     arg=ml_arg()
# #     argg=arg.parse_args(["--maxium_epoch",' 10',"-e1","1","-C"," 1","--metrics","acc","--norm"])
#     argg=arg.parse_args(["--train_data", "/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_train", "--val_data","/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_test","--json_file","./setting.json","-e1",".1","-C"," .1","--maxium_epoch",' 20',"--norm","--metrics","acc"])
#     data=get_data(argg)
#     SVM(data.X_train,data.label_train,data.X_val,data.label_val,data.X_val,data.label_val,param=argg)

argg=arg.parse_args(["--train_data", "/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_train", "--val_data","/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_train","--test_data", "/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_test","--json_file","../../setting_SVM.json","-e1",".1","-C"," .1","--maxium_epoch",' 40',"--metrics","acc"])

# argg=arg.parse_args(["--train_data", "/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_madelon/madelon_data_train", "--val_data","/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_madelon/madelon_data_train","--test_data", "/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_madelon/madelon_data_test","--json_file","../../setting_SVM.json","-e1",".1","-C"," .1","--maxium_epoch",' 80',"--metrics","acc","--norm"])
directory=argg.log_file+"current"+str(datetime.now())+"/"
if not os.path.exists(directory):
    os.makedirs(directory)
sys.stdout=Logger(directory+"log.txt")
print(argg)
data=get_data(argg)
# print(data.X_train[:10,:10],"xdata")
setting=json.load(open(argg.json_file))
print(setting)
directory=argg.log_file+"current_SVM"+str(datetime.now())+"/"
if not os.path.exists(directory):
    os.makedirs(directory)
sys.stdout=Logger(directory+"log.txt")
print(argg)
print("*********************************************** this is SVM*************************")
# [lr1_mi,lr1_ma]=setting['learning_rate1'][0],setting['learning_rate1'][1]
# [lr2_mi,lr2_ma]=setting['learning_rate2'][0],setting['learning_rate2'][1]
lr=setting['learning_rate_SVM']
# par_1=setting['learning_rate1_partition']
# par_2=setting['learning_rate2_partition']
C_trdeoff=setting['C_trdeoff']

gene=data_loader(data.X_train,data.label_train,setting['fold_num']) 

argg.maxium_epoch=setting["max_epoch"]


best_tree=None
best_f1=0
best_lr1=0
best_C_trdeoff=0
peformance={}
peformance["learning_rate1"]=[]
for l in range(len(lr)):
    
    peformance["learning_rate1"].append(str(lr[l]))
for k in range(len(C_trdeoff)):
    

    peformance["C_tradeoff="+str(C_trdeoff[k])]=[]

for k in range(len(lr)):
    
#     peformance["learning_rate1"].append(i)
    for l in  range(len(C_trdeoff)):
#         peformance["learning_rate2="+str(j)]=[]
        per_ech=[]
        for m in range(setting['fold_num']):
            train_x,train_label,val_x,val_label=next(gene)
#             print(train_label.shape,val_label.shape,)
            argg.eta_1=lr[k]
            argg.C=C_trdeoff[l]
       
            results_val=SVM(train_x,train_label,val_x,val_label,param=argg)
#             results_train=batch_predic(data.X_train,data.label_train,tree_dec_np,argg)
#             results_test=batch_predic(data.X_test,data.label_test,tree_dec_np,argg)          
            per_ech.append(results_val["metrics"])
#             print(argg.metrics+" as "+str(results_train["metrics"])+"for trianing when depth is "+str(i)+" and interve is "+str(j)+" at fold "+str(k))
#             print(argg.metrics+" as "+str(results_val["metrics"])+" for validating when lr1 is "+str(lr[k])+" and C_tradeoff is "+str(C_trdeoff[l])+" at fold "+str(m))
        per_ech=np.array(per_ech) 
        mean=per_ech.mean()
        std=per_ech.std()
        print("validating average "+argg.metrics+" as "+str(mean)+" when lr1 is "+str(lr[k])+" and C_tradeoff is "+str(C_trdeoff[l]))
        if  mean>=best_f1:
            best_lr1=lr[k]
            best_C_trdeoff=C_trdeoff[l]
            best_f1=mean
            best_tree=results_val
            print("Found a better tree "+"with "+argg.metrics+" as "+str(best_f1)+" when lr1 is "+str(best_lr1)+" and C_tradeoff is "+str(best_C_trdeoff))
            print("save it as checkpoint in the directory "+directory+argg.metrics+" as "+str(best_f1)+" when lr1 is "+str(best_lr1)+" and C_tradeoff is "+str(best_C_trdeoff)+'.npz')
            np.savez(directory+argg.metrics+" as "+str(best_f1)+" when lr1 is "+str(best_lr1)+" and C_tradeoff is "+str(best_C_trdeoff)+'.npz',w=results_val["W"],b=results_val["b"])
#             results_eval=calculate_metrics_prediction(data.X_val,data.label_val,best_tree,argg)
#             data_xl=pd.DataFrame.from_dict(results_eval["prediction"])
                              
#         data_xl.to_csv(directory+"evalua_for_best_lr1_"+str(best_lr1)+"_and_C_tradeoff"+str(best_C_trdeoff)+argg.metrics+str(best_f1)+str(datetime.now())+".csv")
        peformance["C_tradeoff="+str(argg.C)].append(np.around([mean,std],decimals=3))
#         print(np.around([mean,std],decimals=3))
print("best_lr1 is",best_lr1," best_C_trdeoff" ,best_C_trdeoff, "which we can get " ,argg.metrics+" as "+str(best_f1)+"for validating")
argg.eta_1=best_lr1
argg.C_trdeoff=best_C_trdeoff
results_test=calculate_metrics_prediction(data.X_test,data.label_test,best_tree,argg)
print("the final  performance on test data for ",argg.metrics+" is "+str(results_test["metrics"]))
results_test=calculate_metrics_prediction(data.X_train,data.label_train,best_tree,argg)
# print("the final  performance on training data for ",argg.metrics+" is "+str(results_test["metrics"]))
# print(peformance)
data_xl=pd.DataFrame.from_dict(peformance)
data_xl.to_csv(directory+"mean_std_for_corss_validate"+str(datetime.now())+".csv")
# param=argg



results_eval=calculate_metrics_prediction(data.X_train,data.label_train,best_tree,argg)
print("*******************************final training performance is ",results_eval["metrics"],"******************************")

results_eval=calculate_metrics_prediction(data.X_test,data.label_test,best_tree,argg)
print("*******************************final test performance is ",results_eval["metrics"],"******************************")
# data_xl=pd.DataFrame.from_dict(results_eval["prediction"])
data_xl.to_csv(directory+"evaluation_for_best_lr1_"+str(best_lr1)+"_and_C_tradeoff"+str(best_C_trdeoff)+argg.metrics+str(best_f1)+str(datetime.now())+".csv")
os.rename(directory,argg.log_file+"SVM cross_validate_lr1 "+str(setting['learning_rate_SVM'])+"n_C_tradeoff"+str(setting['C_trdeoff'])+"precision"+str(best_f1)+" metircs"+argg.metrics+"    "+str(datetime.now()))