#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:18:47 2019

@author: dieter
"""
import numpy 
import scipy.special as special
from matplotlib import pyplot

def pistonmodel(freq,radius=0):
    pi = numpy.pi
    angles = numpy.linspace(-pi/2, pi/2, 100)
    if radius==0:radius = 0.014
    wavelength = 340.29/freq
    K = 2*pi/wavelength
    Z = K*(radius*numpy.sin(angles))
    T = 2* special.jv(1,Z)
    N = K*radius*numpy.sin(angles)
    P = numpy.abs(T/N)
    I = P**2
    degrees = numpy.rad2deg(angles)
    return I, degrees

def plot_pistonmodel(freq, radius=0,db=False):
    values,degrees = pistonmodel(freq, radius)
    values = values / numpy.max(values)
    if db:values = 10*numpy.log10(values)
    pyplot.plot(degrees, values)
    pyplot.grid(True)
    pyplot.show()
    

if __name__ == "__main__":
    # For Murata pieze
    plot_pistonmodel(42000, 0.5/100, db=True)
