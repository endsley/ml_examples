#!/usr/bin/env python

from identity_net import *
import warnings



np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")
torch.set_printoptions(precision=3, linewidth=300)



X = torch.randn(5,3)



db = {}
db['net_input_size'] = X.shape[1]
db['net_depth'] = 2
db['width_scale'] = 2
db['cuda'] = False

if(db['cuda']): db['dataType'] = torch.cuda.FloatTensor
else: db['dataType'] = torch.FloatTensor				
db['X_var'] = Variable(X.type(db['dataType']), requires_grad=False)

IN = identity_net(db, True)
[y_pred, auto_out] = IN(db['X_var'])
print(X)
print(y_pred)
print(auto_out)
import pdb; pdb.set_trace()
