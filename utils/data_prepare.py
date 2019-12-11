from scipy.sparse import csr_matrix
import numpy as np
from argparse import Namespace
import sys
sys.path.append("..")
# from algorithms import return_label
def read_libsvm(fname, num_features=0):
	'''
		Reads a libsvm formatted data and outputs the training set (sparse matrix)[1], 
		the label set and the number of features. The number of features
		can either be provided as a parameter or inferred from the data.

		Example usage:
		
		X_train, y_train, num_features = read_libsvm('data_train')
		X_test, y_test, _ = read_libsvm('data_test', num_features)

		[1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
	'''
	data = []
	y = []
	row_ind = []
	col_ind = []
	with open(fname) as f:
		lines = f.readlines()
		for i, line in enumerate(lines):
			elements = line.split()
			y.append(int(elements[0]))
			for el in elements[1:]:
				row_ind.append(i)
				c, v = el.split(":")
				col_ind.append(int(c))
				data.append(float(v))
	if num_features == 0:
		num_features = max(col_ind) + 1
	X = csr_matrix((data, (row_ind, col_ind)), shape=(len(y), num_features))

	return X, np.array(y), num_features
def binary_data(data):
    pos_ind=np.where(data>0)
    data=data*0
    data[pos_ind]=1
    return data
def split_data(X,param):
    n_interve=param.n_interve+1
    ra=np.linspace(-1,1,n_interve)
    ra[0]=np.NINF
    ra[-1]=-np.NINF
    output=np.copy(X)
    for i in range(n_interve-1):
#         print(ra)
        ind=(X>ra[i])*(X<ra[i+1])
#         print(ind)
        output[ind]=i 
    
    return output
def get_data(arg):
    X_val, label_val, num_features=read_libsvm(arg.val_data)
    X_train, label_train, num_features=read_libsvm(arg.train_data)
    X_anon, label_anon, num_features=read_libsvm(arg.anon_data)
    X_train=X_train.toarray()
    X_anon=X_anon.toarray()
    X_val=X_val.toarray()
    ind=np.where(label_val==0)
    label_val[ind]=-1
    ind=np.where(label_train==0)
    label_train[ind]=-1
    ind=np.where(label_anon==0)
    label_anon[ind]=-1    
    if arg.norm==True:
        
        X_train=(X_train-np.mean(X_train,0))/(np.std(X_train,0)+1e-5)
        X_val=(X_val-np.mean(X_val,0))/(np.std(X_val,0)+1e-5)
        X_anon=(X_anon-np.mean(X_anon,0))/(np.std(X_anon,0)+1e-5)
        if hasattr(arg,"n_interve"):
             X_train=split_data(X_train,arg)
             X_val=split_data( X_val,arg)
             X_anon=split_data( X_anon,arg)
    if arg.binary == True:
        X_train=binary_data(X_train)
        X_val=binary_data(X_val)
        X_anon=binary_data(X_anon)
    X_test=None
    label_test=None
    if arg.test_data!=None:
            X_test, label_test, num_features=read_libsvm(arg.test_data)
#             print(X_test.shape)
            X_test=X_test.toarray()
            ind=np.where(label_test==0)
            label_test[ind]=-1
            if arg.norm==True:
                X_test=(X_test-np.mean(X_test,0))/(np.std(X_test,0)+1e-5)
                if hasattr(arg,"n_interve"):
                     X_test=split_data(X_test,arg)
            if arg.binary == True:
                X_test=binary_data(X_test)
    data={"X_train":X_train,"label_train":label_train,"X_val":X_val,\
          "label_val":label_val,"X_test": X_test,"label_test":label_test,\
         "X_anon":X_anon,"label_anon":label_anon}
    data=Namespace(**data)
    return data
def data_loader(data,label,partition=5):
    shuffle_n=np.arange(data.shape[0])
#     print(data.shape[0])
    np.random.shuffle(shuffle_n)
    partition_size=int(data.shape[0]/partition)
#     print(shuffle_n,partition_size)
    five_folds=[shuffle_n[i*partition_size:(i+1)*partition_size] for i in range(partition)]
#     print(five_folds)
    h=np.arange(partition)
    num=0
    while True:
        i=num%partition
        num=num+1
        data_val=data[five_folds[i],:]
        label_val=label[five_folds[i]]
        train_id=five_folds[:i]+five_folds[i+1:]
        train_id=np.reshape(train_id,(-1))
    #             print(train_id.shape)
        data_train=data[train_id,:]
        label_train=label[train_id]
        yield (data_train,label_train,data_val,label_val)