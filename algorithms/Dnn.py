import torch 
import torch.nn as nn
import numpy as np

def init_weights(m):
    if type(m) == nn.Linear:
        
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in')
        m.bias.data.fill_(0.01)

class Dnn():
    def train(self,X,label,X_val=None,label_val=None,param=None):
        input_shape=X.shape[1]
        X = torch.from_numpy(X).double()
        X_val = torch.from_numpy(X_val).double()
        
        ind=np.where(label==-1.0)
        label_train=label
        label_train[ind]=0    
        label = torch.from_numpy(label_train).double()
        label=label[:,None]
        ind=np.where(label_val==-1.0)
        label_val[ind]=0
        label_val = torch.from_numpy(label_val).double()
        label_val=label_val[:,None]
        batch_size=param.batch_size
        train_data = torch.utils.data.TensorDataset(X, label)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        pos_weight = torch.ones([1])*param.pos_weight

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.logits_fn=nn.Sigmoid()
        dnn_hidden_layer=param.DNN_hidden_layer
        layer=[input_shape]+dnn_hidden_layer+[1]
        model=[]
        for i in range(len(layer)-1):  
            model+=[\
                    nn.BatchNorm1d(layer[i]),
                    nn.LeakyReLU(),
#                     nn.Dropout(),
                    nn.Linear(layer[i],layer[i+1])]
        model_sequantial=nn.Sequential(*model)
        model_sequantial.apply(init_weights)
        dtype=torch.DoubleTensor
        model_sequantial.type(dtype)
        lr=param.eta_1
        decay=param.decay
        optim=torch.optim.Adagrad(model_sequantial.parameters(),lr=lr,lr_decay=decay) 
        regularization_loss = 0
        
        best_model=[]
        results={}
        best_metrics=0
        check_point=param.valid_each
        i=0
        i=i+1
        epoch=param.maxium_epoch
        for i in range(epoch):
            for X_batch,y_batch in train_loader:
            
                layer_out=model_sequantial(X_batch)
#                 y_batch=y_batch
                
                regularization_loss = 0
                if i%check_point==0:
                    logits=self.logits_fn(layer_out)
    #                 metrics=self.get_metrics(logits,y,param)
                    results={"param":model_sequantial}
                    
                    if param.verbose==1:
#                         param.metrics="acc"
#                         pre=self.predict(X,label,results,param)
#                         print("current acc metrics",pre["metrics"])
#                         param.metrics="F1_score"
                        pre=self.predict(X_val,label_val,results,param)
                        print("current  validation metrics"+param.metrics+\
                              " is ",pre["metrics"])
                        pre=self.predict(X,label,results,param)
                        print("current   metrics"+param.metrics+" is ",pre["metrics"])
#                         param.metrics="acc"
                        try:
                            pass
#                             print(loss.detach().numpy(),"loss")
                        except:
                            pass
                    pre=self.predict(X_val,label_val,results,param)
                    metrics=pre["metrics"]
#                     print(metrics,"this is the metrics")
                    if metrics>best_metrics:
                        best_metrics=metrics
                        best_model=model_sequantial
                for parame in model_sequantial.parameters():
                    regularization_loss += torch.sum(torch.abs(parame))
                loss=loss_fn(layer_out,y_batch)
#                 print(y_batch.detach().numpy())
#                 print(layer_out.detach().numpy())
#                 print(loss.detach().numpy(),"loss")
                optim.zero_grad()
                loss.backward()
                optim.step()
        results={"param":best_model,"metrics":best_metrics}
        return  results
    
    def get_metrics(self,logits,y,param):
            if param.metrics=="acc":
                y_np=y.numpy() 
                output = (logits>0.5).float().numpy()
#                 print(output,output.shape,"output.shape")
                correct = (output == y_np).sum()
#                 print(output.shape,"output",y,"output.shape",correct)
                acc=correct/y.shape[0]
                metrics=acc
#             if acc
            if param.metrics=="F1_score":
                    y_np=y.numpy()                  
                    output = (logits>0.5).float()
                    output_np=output.numpy()     
                    tag_1=np.where(y_np==1)[0]
                    tag_pre=np.where(output_np==1)[0]
#                     tag_1=torch.where(y==1)[0]
#                     tag_pre=torch.where(output==1)[0]
#                     tag_1=tag_1.numpy()
#                     tag_pre=tag_pre.numpy()
                    f1_score=self.calculate_F1(tag_1,tag_pre)
                    metrics=f1_score
            return metrics
    def calculate_F1(self,ind_actual,ind_pred):
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
    def predict(self,X,y,results,param):
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).double()
            ind=np.where(y==-1.0)
            y[ind]=0
            y = torch.from_numpy(y).double()
#             y = torch.from_numpy(y).double()
            y=y[:,None]
        model=results["param"]
        layer_out=model(X)
        logits=self.logits_fn(layer_out)
        metrics=self.get_metrics(logits,y,param)
        prediction = (logits>0.5).numpy()
        results={"metrics":metrics,"prediction":prediction}
        return results