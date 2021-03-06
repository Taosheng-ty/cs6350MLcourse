import argparse
class ml_arg(argparse.ArgumentParser):
    def __init__(self,description=None):
        super(ml_arg,self).__init__(description=description)
        self._add('-e1','--eta_1',default=0.1,type=float,dest="eta_1",help="this is the first learning rate")
        self._add('-e2','--eta_2',default=0.01,type=float,dest="eta_2",help="this is the second learning rate")
        self._add('-C','--C',default=0.01,type=float,dest="C",help="this is the loss tradeoff parameter")
        self._add('--smooth_term',default=0.01,type=float,dest="smooth_term",help="this is the smooth_term")
        self._add('--k_trees',default=10,type=int,dest="k_trees",help="this is the k_trees")
        self._add('-epo','--maxium_epoch',default=10,type=int,dest="maxium_epoch",help="this is the maxium epoch")
        self._add('-d','--decay',action="store_false",default=True,dest="decay",help="decay or not (default True)")
        self._add('--train_data',dest="train_data",default="/home/taoyang/research/research_everyday/mlprojects/datasets/data/data-splits/data.train",help="the file to give training sets")
        self._add('--val_data',dest="val_data",default="/home/taoyang/research/research_everyday/mlprojects/datasets/data/data-splits/data.eval.anon",help="the file to give validation sets")
        self._add('--test_data',dest="test_data",default="/home/taoyang/research/research_everyday/mlprojects/datasets/data/data-splits/data.test",help="the file to give test sets")
        self._add('--log_file',dest="log_file",default="/home/taoyang/research/research_everyday/mlprojects/log/",help="the folder to save the log")
        self._add('--valid_each',dest="valid_each",type=int,default=5,help="validate every # of epoches")
        self._add('--metrics',dest="metrics",default="F1_score",help="the metrics used for measuring")
        self._add('--n_interve',dest="n_interve",default=6,type=int,help="nuber of interve for each feature")
        self._add('--norm',dest="norm",default=False,action="store_true",help="Whether to to norm it or not?")
        self._add('--tree_depth',dest="tree_depth",action="store_false",default=True,help="depth of the tree")
        self._add('--binary',dest="binary",action="store_true",default=False,help="binary the data or not?")
        self._add('--json_file',dest="json_file",default="/home/taoyang/research/research_everyday/mlprojects/setting.json",help="json_file to do some hyperparameter truning.")
    def _add(self,*arg,**kwargs):
        super(ml_arg,self).add_argument(*arg,**kwargs)
#     def parse_args(self,*arg,**kwargs):
#         return super(ml_arg,self).parse_args(*arg,**kwargs)
    def value(self):
        return vars(self.parse_args())
if __name__=="__main__":
        arg=ml_arg()
        print(arg.value())