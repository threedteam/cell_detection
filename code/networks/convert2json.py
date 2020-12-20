import pprint, pickle
import json
import numpy

import  pickle
# 重点是rb和r的区别，rb是打开二进制文件，r是打开文本文件
f=open('result_ee.pkl','rb')
data = pickle.load(f)
print(data)