#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import pandas as pd
import xgboost as Xgb
from Xgb import XGBRegressor

trainData_file = 'trainData.csv'
train_data = pd.read_csv(trainData_file, index_col=None).values

xgb = XGBRegressor()
xgb.fit(train_data[:, 1:], train_data[:, 0])
xgb.save_model('model.xgb')
