#!/usr/bin/env python

import os
import sys
if os.path.exists('/home/chieh/code/wuML'):
	sys.path.insert(0,'/home/chieh/code/wuML')
import wuml


data = wuml.wData('./letters.original.data', label_type="discrete", label_column_id=0)

wuml.csv_out(data.X, './letters.csv', float_format='%d')
wuml.csv_out(data.Y, './letters_label.csv', float_format='%d')

import pdb; pdb.set_trace()
