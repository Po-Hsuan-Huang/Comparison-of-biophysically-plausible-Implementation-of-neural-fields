# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:11:36 2015

@author: pohsuanhuang
"""

def test(boo,**kargs):
    if boo=='True':
        a =  kargs.get('a')
        b =  kargs.get('b')
        c =  kargs.get('c')
        print boo,a,b,c
    else:
        print 'nothing'

boo = 'False'
alias = {'a':1,'b':2,'c':3}
test(boo,a=1,b=2,c=3)
    
