#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:19:22 2019

@author: alejandrogonzales
"""
from RectCoords import RectCoords

def overlap(rect1, rect2):
    
    # Heights and widths
    x1 = rect1[0]
    y1 = rect1[1]
    w1 = rect1[2]
    h1 = rect1[3]
    l1 = RectCoords(x1,y1)
    r1 = RectCoords(x1+w1, y1+h1)
    
    print("L1 = " + str(l1.x) + " : " + str(l1.y))
    print("R1 = " + str(r1.x) + " : " + str(r1.y))
    
    x2 = rect2[0]
    y2 = rect2[1]
    w2 = rect2[2]
    h2 = rect2[3]
    l2 = RectCoords(x2,y2)
    r2 = RectCoords(x2+h2, y2+h2)
    
    print("L2 = " + str(l2.x) + " : " + str(l2.y))
    print("R2 = " + str(r2.x) + " : " + str(r2.y))
    
    # Rectangle on the left side of the other
    if (r2.x < l1.x or r1.x < l2.x):
        return False
    
    # Rectangle above the other
    if (l1.y < r2.y or r1.y < l2.y):
        return False
    
    # One rectangle on the left of the other
    return True
    
rect1 = [160, 319, 111, 202]
rect2 = [165, 259, 127, 144]
#rect2 = [638 259 127 144]
doOverlap = overlap(rect1, rect2)
print("Do overlap = " + str(doOverlap))
print("\n")
