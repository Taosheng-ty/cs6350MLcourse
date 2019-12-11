import os
import sys
sys.path.append("..")
from utils import Logger
from algorithms.decision_tree import conver2numpy,batch_predic,Decision_tree,itera_tree
from utils import calculate_F1
from scipy.sparse import csr_matrix
import numpy as np
import json
from argparse import Namespace
from utils import ml_arg
from utils import get_data
def Random_forest_prediction(X,label,tree_total,parse=None):
    prediction=[]
    tree_total=tree_total["param"]
    for i in range(len(tree_total)):
#         print(tree_total[i][0],"fea")
        results=batch_predic(X,label,tree_total[i],parse)
        prediction.append(results["prediction"])
    prediction=np.array(prediction)
#     print(prediction.shape,"prediction.shape")
    forest_pred=np.ones(X.shape[0])
    for i in range(prediction.shape[1]):
#         print(prediction[i,:10])
        a=prediction[:,i]
        counts = np.bincount(a+1)
#         print(counts,"counts")
#         print(label[i],"label[i]")
        forest_pred[i]=np.argmax(counts)-1
#         print(forest_pred[i],"forest_pred[i]")
#         print(forest_pred)
#         if np.argmax(counts)==0:
#             forest_pred[i]=np.argmax(counts)
#     print(forest_pred.shape,label.shape,"forest_pred.shape,label.shape")
    metrics=calculate_metrics(forest_pred,label,parse)
    feature=np.transpose(np.array(prediction))
    final_results={"prediction":forest_pred,"metrics":metrics,"feature":feature}
    return final_results

def calculate_metrics(prediction,label,parse):
        if parse.metrics=="F1_score":
            ind_label=np.where(label==1)[0]
            ind_prediction=np.where(prediction==1)[0]
            ratio=calculate_F1(ind_label,ind_prediction)
            return ratio
        else:
#             print(prediction==label,"prediction==label",prediction.shape,label.shape)
            ratio=np.where(prediction==label)[0].shape[0]/prediction.shape[0]
            return ratio

def Random_forest(X,label,X_val,label_val,parse=None,k_trees=100):
    k_trees=parse.k_trees
    setting=json.load(open(parse.json_file))
    random_sample=setting["random_sample"]
    random_feature=setting["random_feature"]
    tree_total=[]
    for i in range(k_trees):
        n_feature=np.arange(X.shape[1])
        n_sample=np.arange(X.shape[0])[:random_sample]
#         n_sample=np.random.randint(0,X.shape[0],random_sample)
        np.random.shuffle(n_feature)
        sample_feature=n_feature[0:random_feature] 
#         print(sample_feature,"sample_feature")
        X_sample=X[n_sample,:]
        X_sample=X_sample[:,sample_feature]
#         print(X_sample.shape,"X_sample.shape")
        label_sample=label[n_sample]
        parse.tree_depth=1
        tree=conver2numpy(itera_tree(X_sample,label_sample,parse=parse))
#         print(tree,"tree",)
        results1=batch_predic(X_sample,label_sample,tree,parse)
#         print(results1["metrics"],"in smaple")
        to={"param":[tree]}
        results1=Random_forest_prediction(X_sample,label_sample,to,parse=parse)  
#         print(results1["metrics"],"in random tree prediction")
        for i in range(parse.tree_depth):
            for j in range(tree.shape[0]):
                tree[j,2*i]=n_feature[tree[j,2*i]]
#         print(tree,"tree return")
        results1=batch_predic(X,label,tree,parse)

#         print(results1["metrics"])
        tree_total.append(tree)
    resutls={"param":tree_total}
    uu=Random_forest_prediction(X_val,label_val,resutls,parse=parse)    
    print("performance on validation is "+parse.metrics+ str(uu["metrics"]))
    resutls["metrics"]=uu["metrics"]
    return resutls




class Random_forest:
    def train(self,X,label,X_val,label_val,param=None):
        parse=param
        k_trees=parse.k_trees
#         setting=json.load(open(parse.json_file))
        random_sample=parse.random_sample
        random_feature=parse.random_feature
        tree_total=[]
        self.decision_tree=Decision_tree()
        for i in range(k_trees):
            n_feature=np.arange(X.shape[1])
            n_sample=np.arange(X.shape[0])[:random_sample]
    #         n_sample=np.random.randint(0,X.shape[0],random_sample)
            np.random.shuffle(n_feature)
            sample_feature=n_feature[0:random_feature] 
    #         print(sample_feature,"sample_feature")
            X_sample=X[n_sample,:]
            X_sample=X_sample[:,sample_feature]
    #         print(X_sample.shape,"X_sample.shape")
            label_sample=label[n_sample]
#             parse.tree_depth=1
#             print(param,"in line 113")
            tree=self.decision_tree.train(X_sample,label_sample,X_sample,label_sample,param=parse)["param"]
    #         print(tree,"tree",)
#             results1=decision_tree.train(X_sample,label_sample,tree,parse)
#             print(results1["metrics"],"in sample")
#             to={"param":[tree]}
#             results1=Random_forest_prediction(X_sample,label_sample,to,parse=parse)  
    #         print(results1["metrics"],"in random tree prediction")
#             print(tree[:,3])
            for i in range(parse.tree_depth):
                for j in range(tree.shape[0]):
                    if type(tree[j,2*i])==int:
                        tree[j,2*i]=n_feature[tree[j,2*i]]
    #         print(tree,"tree return")
#             results1=batch_predic(X,label,tree,parse)

#             print(results1["metrics"],"return in whole training")
#             results1=batch_predic(X_val,label_val,tree,parse)

#             print(results1["metrics"],"return in whole validating")
            tree_total.append(tree)
        resutls={"param":tree_total}
#         uu=Random_forest_prediction(X_val,label_val,resutls,parse=parse)    
#         print("Random_forest_prediction performance on validation is "+parse.metrics+ str(uu["metrics"]))
        uu=self.predict(X_val,label_val,resutls,param=parse)    
        print(" self.predict performance on validation  is "+parse.metrics+ str(uu["metrics"]))
        resutls["metrics"]=uu["metrics"]
        return resutls
    def predict(self,X,label,tree_total,param=None):
        prediction=[]
        tree_total=tree_total["param"]
        for i in range(len(tree_total)):
    #         print(tree_total[i][0],"fea")
            tree_total_i={"param":tree_total[i]}
            results=self.decision_tree.predict(X,label,tree_total_i,param)
            prediction.append(results["prediction"])
        prediction=np.array(prediction)
    #     print(prediction.shape,"prediction.shape")
        forest_pred=np.ones(X.shape[0])
        for i in range(prediction.shape[1]):
    #         print(prediction[i,:10])
            a=prediction[:,i]
            counts = np.bincount(a+1)
    #         print(counts,"counts")
    #         print(label[i],"label[i]")
            forest_pred[i]=np.argmax(counts)-1
    #         print(forest_pred[i],"forest_pred[i]")
    #         print(forest_pred)
    #         if np.argmax(counts)==0:
    #             forest_pred[i]=np.argmax(counts)
    #     print(forest_pred.shape,label.shape,"forest_pred.shape,label.shape")
        metrics=calculate_metrics(forest_pred,label,param)
        feature=np.transpose(np.array(prediction))
        final_results={"prediction":forest_pred,"metrics":metrics,"feature":feature}
        return final_results





















if __name__=="__main__":
    arg=ml_arg()
#     argg=arg.parse_args(["--maxium_epoch",' 10',"-e1","1","-C"," 1","--metrics","acc","--norm"])
    argg=arg.parse_args(["--train_data", "/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_train", "--val_data","/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_train","--test_data","/home/taoyang/research/research_everyday/homework/ml6350/hw5/data/data_semeion/hand_data_train","--json_file","../../setting.json","-e1",".1","-C"," .1","--maxium_epoch",' 20',"--metrics","acc"])
    data=get_data(argg)
    argg.depth=1
    argg.n_interve=3
    argg.metrics="acc"
#     tree_decision=Random_forest(data1.X_train,data1.label_train,parse=argg)
    tree_decision=Random_forest(data.X_train,data.label_train,data.X_val,data.label_val,parse=argg)
#     print(data.X_train.shape,"shappe")
#     argg.metrics="F1_score"
    uu=Random_forest_prediction(data.X_val,data.label_val,tree_decision,parse=argg)