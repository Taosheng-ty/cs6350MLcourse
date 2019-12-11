import numpy as np
def calculate_metrics(x,label,W,b,param):
    if param.metrics=="acc":
        wrong_ind=np.where(label*(x@W+b)<=0)
        val_acc=1-len(wrong_ind[0])/x.shape[0]
        return val_acc
    if param.metrics=="F1_score":
        ind_actual=np.where(label>0)[0]
        ind_pred=np.where(x@W+b>0)[0]
        f1_score=calculate_F1(ind_actual,ind_pred)
        return f1_score
def calculate_metrics_prediction(x,label,results,param):
    W=results["W"]
    b=results["b"]
    prediction={}
    predict_label=np.ones_like(label)
    id_neg=np.where((x@W+b)<=0)
    predict_label[id_neg]=-1
    prediction["prediction"]=predict_label   
    prediction["metrics"]=calculate_metrics(x,label,W,b,param)
    return prediction
def calculate_F1(ind_actual,ind_pred):
#         ind_actual=np.where(ind_actual==1)[0]
#         ind_pred=np.where(ind_pred==1)[0]
        ind_actual_set=set(ind_actual)
        ind_pred_set=set(ind_pred)
#         print(ind_actual_set)
#         print(ind_pred_set)
        cross=list(ind_actual_set.intersection(ind_pred_set))
#         print(cross,"cross")
        tp=len(cross)
        fn=ind_actual.shape[0]
        fp=ind_pred.shape[0]
        if fp*fn*tp==0:
#             print(fp,fn,"this is fp and fn")
            return 0
        p=tp/(fp)
        r=tp/(fn)
        f1_score=2*p*r/(p+r)
        return f1_score         
