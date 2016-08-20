# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:22:53 2015

@author: pohsuanhuang
"""
'''
Ban.py imports Bing.py, and execute the module funciton if input = I ; 

It execute Bing.py as a script if input = E
'''
 

print 'This is Ban, calling Bing ...'

loop = raw_input('Import Bing, or execute Bing ? [I/E] ')
duh = 'N'
q = 1
while q==1:
    if loop == 'I':
        if duh == 'Y':
            reload(Bing)        
        elif duh == 'N':
            import Bing
        var = Bing.Boo()
        print var
        
        
        
        duh = raw_input('Show the welcome line agian ? Y/N')
        q  = input('Press 0 to quit.Press 1 to continue.')    
        
    elif loop =='E':
        execfile('Bing.py')
        q  = input('Press 0 to quit. Press 1 to continue.')
