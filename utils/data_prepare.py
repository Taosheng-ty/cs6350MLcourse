from scipy.sparse import csr_matrix
import numpy as np
from argparse import Namespace
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
def get_data(arg):
    X_val, label_val, num_features=read_libsvm(arg.val_data)
    X_train, label_train, num_features=read_libsvm(arg.train_data)
    X_train=X_train.toarray()
    X_val=X_val.toarray()
    X_train=(X_train-np.mean(X_train,0))/(np.std(X_train,0)+1e-5)
    X_val=(X_val-np.mean(X_val,0))/(np.std(X_val,0)+1e-5)
    ind=np.where(label_val==0)
    label_val[ind]=-1
    ind=np.where(label_train==0)
    label_train[ind]=-1
    X_test=None
    label_test=None
    if arg.test_data!=None:
            X_test, label_test, num_features=read_libsvm(arg.test_data)
            ind=np.where(label_test==0)
            label_test[ind]=-1
            X_test=(X_test-np.mean(X_test,0))/(np.std(X_test,0)+1e-5)
    data={"X_train":X_train,"label_train":label_train,"X_val":X_val,"label_val":label_val," X_test": X_test,"label_test":label_test}
    data=Namespace(**data)
    return data