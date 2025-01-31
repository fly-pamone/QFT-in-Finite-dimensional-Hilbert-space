#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 00:37:59 2024

@author: haoli
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from math import pi
def main(L,C):
    L=int(L)
    Matrix=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.float64)
    for j in range(-L,L+1):
        for i in range(-L,L+1):
            if i==j:
              Matrix[j+L,i+L]=2*C*j
            else:
              Matrix[j+L,i+L]=-mp.csc(2*pi*L*(j-i)/(2*L+1))
  

    NM=np.array([[0]*L]*L,dtype=np.float64)
    for g in range(0,L):
        for h in range(0,L):
            NM[g,h]=Matrix[g,h]+Matrix[g,(2*L-h)]

    NZ=np.array([0]*L,dtype=np.float64)
    for m in range(0,L):
        NZ[m]=-Matrix[m,L]

    y=np.linalg.solve(NM,NZ)
    f= y[::-1]
    Full=np.array([0]*(2*L+1),dtype=np.float64)
    for g in range(0,L):
        Full[g]=y[g]
    for g in range(L+1,2*L+1):
        Full[g]=f[g-L-1]
    Full[L]=1  
    return Full

# Get the vaccum state

def omega(k,t):
    y=k**2-1/(((-t+1.5)**2)*((np.log(-t+1.5))**2))
    return y
  
# The frequncy of the Harmonic Oscilator  
    

def OP1(d,k,i):
    phi=(2*pi/(d*(omega(k,i)))**(1/2))**(1/2)
    beta=(2*pi*(omega(k,i))**(1/2)/d)**(1/2)
    L=int((d-1)/2)
    Matrix=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.complex64)
    for o in range(-L,L+1):
        for i in range(-L,L+1):
            if i==o:
              Matrix[o+L,i+L]=(2*pi/(d*beta))*o
            else:
              Matrix[o+L,i+L]=0
    return Matrix

#Field operator

def OP2(d,k,i):
    phi=(2*pi/(d*(omega(k,i))**(1/2)))**(1/2)
    beta=(2*pi*(omega(k,i))**(1/2)/d)**(1/2)
    L=int((d-1)/2)
    Matrix=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.complex64)
    for o in range(-L,L+1):
        for i in range(-L,L+1):
            if i==o:
              Matrix[o+L,i+L]=0
            else:
              Matrix[o+L,i+L]=(1j*pi/(d*phi))*mp.csc(2*pi*L*(o-i)/(2*L+1))
    return Matrix

#Momentun operator

def OP3(d):
    L=int((d-1)/2)
    Matrix=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.complex64)
    for o in range(-L,L+1):
        for i in range(-L,L+1):
            if i==o:
              Matrix[o+L,i+L]=0
            else:
              Matrix[o+L,i+L]=(pi*(o-i)/d)*mp.csc(2*pi*L*(o-i)/(2*L+1))
    return Matrix

def CM(A,B):
    C=np.dot(A,B)-np.dot(B,A)
    return C

#Comutator function

def exp(d,w,s,k,i):
    L=int((d-1)/2)
    S=np.array(range(0,s),dtype=np.complex64)
    A=OP1(d,k,i)
    B=OP2(d,k,i)
    dt=-(i)/s
    state=main(L,w)
    for n in range(0,s):
        H=(np.dot(B,B)+omega(k,i+n*dt)*np.dot(A,A))/2
        A=A+1j*CM(H,A)*dt
        B=B+1j*CM(H,B)*dt
        anti=np.dot(A,A)
        state1=np.dot(anti,state)
        state2=np.dot(state,state1)/np.dot(state,state)
        S[n]=state2
    return S

# Expectation value of feild square in finite dimensional Hilbert space



def epp2(s,k,i):
    S=np.array(range(0,s),dtype=np.complex64)
    A=1/(omega(k,i))**(1/4)
    B=1j*(omega(k,i))**(1/4)
    phi=A
    pi=B
    dt=-(i)/s
    for n in range(0,s):
        B=-omega(k,i+n*dt)*phi*dt+pi
        A=pi*dt+phi
        phi=A
        pi=B
        A_conj=A.conjugate()
        state2=A*A_conj/2
        S[n]=state2
    return S
# Expectation value of feild square in infinite dimensional Hilbert space
    
    
def pf(d1,d2,d3,d4,w,s,k,i):
    x=np.array(range(0,s))*(-i/(s-1))+i
    y1=exp(d1,w,s,k,i)
    y2=exp(d2,w,s,k,i)
    y3=exp(d3,w,s,k,i)
    y4=epp2(s,k,i)
   
    plt.plot(x,y1,'r',label='d=3')
    plt.plot(x,y2,'b',label='d=9') 
    plt.plot(x,y3,'y',label='d=27') 
    plt.plot(x,y4,label='Infinite Dimension') 
    plt.xlabel(r"$\tilde{\eta}$")
    plt.ylabel(r"$\langle 0| \hat{\Phi}^2|0\rangle$")
    plt.legend()
    
# compare the expectation value
    
# Here w is the X we defined in the thesis. s is the step number of the differential culculation. i is the time we start the culculation.
# k is the momentum we take.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    