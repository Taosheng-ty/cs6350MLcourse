import sys
sys.path.append("../../cs6350MLcourse/")
from algorithms import *
from algorithms.decision_tree import *
from utils import *
import json
import os
arg=ml_arg()
argg=arg.parse_args()
# argg=arg.parse_args(["--tree_depth","3","--train_data", "datasets/data/data-splits/data.train","--val_data", "datasets/data/data-splits/data.eval.anon", "--test_data","datasets/data/data-splits/data.test","--json_file","./setting.json"])
print(argg)
data=get_data(argg)
print(data.X_train[:10,:10])
setting=json.load(open(argg.json_file))
print(setting)
directory=argg.log_file+"current_decition_tree"+str(datetime.now())+"/"
if not os.path.exists(directory):
    os.makedirs(directory)
sys.stdout=Logger(directory+"log.txt")
print(argg)
print("*********************************************** this desicion tree *************************")
[dep_mi,dep_ma]=setting['depth_limited_min_max'][0],setting['depth_limited_min_max'][1]
[interve_mi,interve_ma]=setting['interve_min_max'][0],setting['interve_min_max'][1]
gene=data_loader(data.X_train,data.label_train,setting['fold_num']) 
a1,b1,c1,d1=next(gene)
best_tree=None
best_f1=0
best_depth=0
best_interve=0
peformance={}
peformance["depth_limit"]=[]
for j in range(interve_mi,interve_ma+2,2):
    peformance["interve="+str(j)]=[]

for i in range(dep_mi,dep_ma+1):
    peformance["depth_limit"].append(i)
for i in range(dep_mi,dep_ma+1):

    for j in range(interve_mi,interve_ma+2,2):

        per_ech=[]
        for k in range(setting['fold_num']):
            train_x,train_label,val_x,val_label=next(gene)
#             print(train_label.shape,val_label.shape,)
            argg.tree_depth=i
            argg.n_interve=j
            re=itera_tree(train_x,train_label,parse=argg)
            tree_dec_np=conver2numpy(re)
            results_val=batch_predic(val_x,val_label,tree_dec_np,argg)
            results_train=batch_predic(train_x,train_label,tree_dec_np,argg)
#             results_train=batch_predic(data.X_train,data.label_train,tree_dec_np,argg)
#             results_test=batch_predic(data.X_test,data.label_test,tree_dec_np,argg)          
            per_ech.append(results_val["metrics"])
            print(argg.metrics+" as "+str(results_train["metrics"])+"for trianing when depth is "+str(i)+" and interve is "+str(j)+" at fold "+str(k))
            print(argg.metrics+" as "+str(results_val["metrics"])+" for validating when depth is "+str(i)+" and interve is "+str(j)+" at fold "+str(k))
        per_ech=np.array(per_ech) 
        mean=per_ech.mean()
        std=per_ech.std()
        print("validating average "+argg.metrics+" as "+str(mean)+" when depth is "+str(i)+" and interve is "+str(j))
        if  mean>=best_f1:
            best_depth=i
            best_interve=j
            best_f1=mean
            best_tree=tree_dec_np
            print("Found a better tree "+"with "+argg.metrics+" as "+str(best_f1)+" when depth is "+str(i)+" and interve is "+str(j))
            print("save it in the directory "+directory+argg.metrics+" as "+str(best_f1)+" when depth is "+str(i)+" and interve is "+str(j)+'.npz')
            np.savez(directory+argg.metrics+" as "+str(best_f1)+" when depth is "+str(i)+" and interve is "+str(j)+'.npz',best_tree=tree_dec_np)
            results_eval=batch_predic(data.X_val,data.label_val,best_tree,argg)
            data_xl=pd.DataFrame.from_dict(results_eval["prediction"])
            data_xl.to_csv(directory+"evaluation_for_best_depth_"+str(best_depth)+"_and_interve"+str(best_interve)+argg.metrics+str(best_f1)+str(datetime.now())+".csv")
        peformance["interve="+str(j)].append(np.around([mean,std],decimals=3))   
print("best_depth is",best_depth," best_interve" ,best_interve, "which we can get " ,argg.metrics+" as "+str(best_f1)+"for validating")
argg.tree_depth=best_depth
argg.n_interve=best_interve
results_test=batch_predic(data.X_test,data.label_test,best_tree,argg)
print("the final  performance on test data for ",argg.metrics+" is "+str(results_test["metrics"]))
results_test=batch_predic(data.X_train,data.label_train,best_tree,argg)
print("the final  performance on training data for ",argg.metrics+" is "+str(results_test["metrics"]))
data_xl=pd.DataFrame.from_dict(peformance)
data_xl.to_csv(directory+"mean_std_for_corss_validate"+str(datetime.now())+".csv")
param=argg
results_eval=batch_predic(data.X_val,data.label_val,best_tree,argg)
data_xl=pd.DataFrame.from_dict(results_eval["prediction"])
data_xl.to_csv(directory+"evaluation_for_best_depth_"+str(best_depth)+"_and_interve"+str(best_interve)+argg.metrics+str(best_f1)+str(datetime.now())+".csv")
os.rename(directory,param.log_file+" cross_validate_tree_depth "+str(setting['depth_limited_min_max'])+"n_interve"+str(setting['interve_min_max'])+"precision"+str(best_f1)+" metircs"+param.metrics+"    "+str(datetime.now()))