#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 00:35:10 2019

@author: alejandrogonzales
"""
from Rectangle import Rectangle

# 1.
a = Rectangle(0, 0, 1, 1)
b = Rectangle(0.5, 0.5, 1.5, 1.5)
print("A & B Intersection:")
print(a&b)
print("\n")
# Rectangle(0.5, 0.5, 1, 1)
#print("A - B Difference:")
#print(list(a-b))
#print("\n")

# [Rectangle(0, 0, 0.5, 0.5), Rectangle(0, 0.5, 0.5, 1), Rectangle(0.5, 0, 1, 0.5)]

# 2.
b = Rectangle(0.25, 0.25, 1.25, 0.75)
print("A & B Intersection:")
print(a&b)
print("\n")
# Rectangle(0.25, 0.25, 1, 0.75)
#print("A - B Difference:")
#print(list(a-b))
#print("\n")
# [Rectangle(0, 0, 0.25, 0.25), Rectangle(0, 0.25, 0.25, 0.75), Rectangle(0, 0.75, 0.25, 1), Rectangle(0.25, 0, 1, 0.25), Rectangle(0.25, 0.75, 1, 1)]

# 3.
b = Rectangle(0.25, 0.25, 0.75, 0.75)
print("A & B Intersection:")
print(a&b)
print("\n")
# Rectangle(0.25, 0.25, 0.75, 0.75)
#print("A - B Difference:")
#print(list(a-b))
#print("\n")
# [Rectangle(0, 0, 0.25, 0.25), Rectangle(0, 0.25, 0.25, 0.75), Rectangle(0, 0.75, 0.25, 1), Rectangle(0.25, 0, 0.75, 0.25), Rectangle(0.25, 0.75, 0.75, 1), Rectangle(0.75, 0, 1, 0.25), Rectangle(0.75, 0.25, 1, 0.75), Rectangle(0.75, 0.75, 1, 1)]

# 4.
b = Rectangle(5, 5, 10, 10)
print("A & B Intersection:")
print(a&b)
print("\n")
# None
#print("A - B Difference:")
#print(list(a-b))
#print("\n")
# [Rectangle(0, 0, 1, 1)]

# 5.
b = Rectangle(-5, -5, 10, 10)
print("A & B Intersection:")
print(a&b)
print("\n")
# Rectangle(0, 0, 1, 1)
#print("A - B Difference:")
#print(list(a-b))
#print("\n")
# []