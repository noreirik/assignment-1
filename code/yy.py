from __future__ import division
import os
import csv
f=open('adult.data.txt','r')
f1=open('test.gt','w')
f2=open('test.pred','w')
context=f.readlines()
for index in range(len(context)):
    if index%2==0:
        f1.write(context[index])
    else:
        f2.write(context[index])
f.close()
f1.close()
f2.close()




