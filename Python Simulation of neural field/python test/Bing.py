# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 16:58:47 2015

@author: pohsuanhuang
"""

'''The program will be run only the first time being imported

To run the file multiple time using import, you must relaod().


'''

if __name__ == '__main__':
    print 'Bing is run by itself as a script'
    
else:
    
    print ' Welcome ! Bing is run as an imported module for the first time'
    
    def Boo():
        print 'Mudule attribute is executed !'
        print ''
        a=1
        b=2
        c=3
        print a,b,c
        return a,b,c