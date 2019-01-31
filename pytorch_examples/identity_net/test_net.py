#!/usr/bin/env python

from identity_net import *

X = torch.randn(5,3)



db = {}
db['net_input_size'] = X.shape[1]
db['net_depth'] = 4
db['width_scale'] = 2
db['cuda'] = False

if(db['cuda']): db['dataType'] = torch.cuda.FloatTensor
else: db['dataType'] = torch.FloatTensor				
db['X_var'] = Variable(X.type(db['dataType']), requires_grad=False)

IN = identity_net(db, False)
xout = IN(db['X_var'])
import pdb; pdb.set_trace()
