import argparse
class ml_arg(argparse.ArgumentParser):
    def __init__(self,description=None):
        super(ml_arg,self).__init__(description=description)
        self._add('-e1','--eta_1',default=0.1,type=float,dest="eta_1",help="this is the first learning rate")
        self._add('-e2','--eta_2',default=0.01,type=float,dest="eta_2",help="this is the second learning rate")
        self._add('-epo','--maxium_epoch',default=10,type=int,dest="maxium_epoch",help="this is the maxium epoch")
        self._add('-d','--decay',action="store_false",default=True,dest="decay",help="decay or not (default True)")
        self._add('--train_data',dest="train_data",required=True,help="the file to give training sets")
        self._add('--val_data',dest="val_data",required=True,help="the file to give validation sets")
        self._add('--test_data',dest="test_data",help="the file to give test sets")
        self._add('--log_file',dest="log_file",default="./",help="the folder to save the log")
        self._add('--valid_each',dest="valid_each",type=int,default=5,help="validate every # of epoches")
    def _add(self,*arg,**kwargs):
        super(ml_arg,self).add_argument(*arg,**kwargs)
#     def parse_args(self,*arg,**kwargs):
#         return super(ml_arg,self).parse_args(*arg,**kwargs)
    def value(self):
        return vars(self.parse_args())
if __name__=="__main__":
        arg=ml_arg()
        print(arg.value())