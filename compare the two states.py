#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 21:48:03 2024

@author: haoli
"""

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
from math import pi
import math as ma
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from numpy import linalg as LA

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

def matrix(L,X):
    d=2*L+1
    L=int(L)
    Otirx=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.float64)
    def jojo(x,y):
        SUM=0
        for n in range(-L,L+1):
           if n!=x and n!=y:
               SUM=SUM+mp.csc(2*pi*L*(j-n)/(2*L+1))*mp.csc(2*pi*L*(i-n)/(2*L+1))
        return SUM
    for j in range(-L,L+1):
        for i in range(-L,L+1):
            if i==j:
              Otirx[j+L,i+L]=X*j*j+(1/4)*jojo(j,j)
            else:
              Otirx[j+L,i+L]=(1/4)*jojo(i,j)
   
    return Otirx

# The energy operator

def find(L,X):
    d=2*L+1
    x0=np.array(range(-L,L+1))
    M=matrix(L,X)
    eigenvalues, eigenvectors = LA.eig(M)
    INDEX=np.argmin(eigenvalues)
    MINVECTOR=np.array([0]*(2*L+1),dtype=np.float64)
    for g in range(0,2*L+1):
        MINVECTOR[g]=eigenvectors[g,INDEX]
    NORMALIZATION=MINVECTOR*(1/MINVECTOR[L])
    return NORMALIZATION

#The ground state
 
    
def OP1(d):
    phi=(2*pi/(d))**(1/2)
    beta=(2*pi/d)**(1/2)
    L=int((d-1)/2)
    Matrix=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.complex64)
    for o in range(-L,L+1):
        for i in range(-L,L+1):
            if i==o:
              Matrix[o+L,i+L]=(2*pi/(d*beta))*o
            else:
              Matrix[o+L,i+L]=0
    return Matrix
# field operator
def OP2(d):
    phi=(2*pi/(d))**(1/2)
    beta=(2*pi/d)**(1/2)
    L=int((d-1)/2)
    Matrix=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.complex64)
    for o in range(-L,L+1):
        for i in range(-L,L+1):
            if i==o:
              Matrix[o+L,i+L]=0
            else:
              Matrix[o+L,i+L]=(1j*pi/(d*phi))*mp.csc(2*pi*L*(o-i)/(2*L+1))
    return Matrix

#momentum operator
def OP3(d):
    L=int((d-1)/2)
    Matrix=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.complex64)
    for o in range(-L,L+1):
        for i in range(-L,L+1):
            if i==o:
              Matrix[o+L,i+L]=-1
            else:
              Matrix[o+L,i+L]=(pi*(o-i)/d)*mp.csc(2*pi*L*(o-i)/(2*L+1))
    return Matrix

def aplus(d):
    a=(OP1(d)-1j*OP2(d))/(2**(1/2))
    return a

def aanni(d):
    a=(OP1(d)+1j*OP2(d))/(2**(1/2))
    return a    
    

def antiCT(A,B):
    AT=np.dot(A,B)+np.dot(B,A)
    return AT

    

def CM(A,B):
    C=np.dot(A,B)-np.dot(B,A)
    return C

def showp(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=antiCT(antiCT(aplus(d),OP3(d)),aplus(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2
 
def showm(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=antiCT(antiCT(aanni(d),OP3(d)),aanni(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

def showh(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=antiCT(OP3(d),aplus(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2


def show0(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=antiCT(antiCT(aplus(d),OP3(d)),aanni(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2 

def show000(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=antiCT(antiCT(aanni(d),OP3(d)),aplus(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2 
    
def show1(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=antiCT(antiCT(OP1(d),OP3(d)),OP1(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

def show2(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=antiCT(antiCT(OP2(d),OP3(d)),OP2(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

def show12(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=antiCT(antiCT(OP1(d),OP3(d)),OP2(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

def show121(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=antiCT(antiCT(OP2(d),OP3(d)),OP1(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

def show20(d,w):
    L=int((d-1)/2)
    state=find(L,w)
    anti=antiCT(antiCT(OP2(d),OP3(d)),OP2(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

def show21(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=np.dot(aanni(d),aplus(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

def show22(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=np.dot(aplus(d),aplus(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

def show220(d,w):
    L=int((d-1)/2)
    state=find(L,w)
    anti=np.dot(aplus(d),aplus(d))
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

def show221(d,w):
    L=int((d-1)/2)
    state=main(L,w)
    anti=OP3(d)
    state1=np.dot(anti,state)
    state2=np.dot(state,state1)/np.dot(state,state)
    return state2

    
def cmp(d,x):
    L=int((d-1)/2)
    y1=main(1,x)
    y2=find(1,x)
    y11=main(10,x)
    y22=find(10,x)
    y111=main(100,x)
    y222=find(100,x)
    y1111=main(200,x)
    y2222=find(200,x)
    x0=np.array(range(-1,2),dtype=np.complex64)
    x01=np.array(range(-10,11),dtype=np.complex64)
    x011=np.array(range(-100,101),dtype=np.complex64)
    x0111=np.array(range(-200,201),dtype=np.complex64)
    plt.subplot(2,2,1)
    plt.plot(x0,y1,'r',label='d=3')
    plt.plot(x0,y2,'b_',label='d=3')
    plt.legend(prop = {'size':8})
    plt.subplot(2,2,2)
    plt.plot(x01,y11,'r',label='d=21')
    plt.plot(x01,y22,'b_',label='d=21')
    plt.legend(prop = {'size':8})
    plt.subplot(2,2,3)
    plt.plot(x011,y111,'r',label='d=201')
    plt.plot(x011,y222,'b_',label='d=201')
    plt.xlabel('L which is the label of eigenstate')
    plt.ylabel('Wave function')
    plt.legend(prop = {'size':8})
    plt.subplot(2,2,4)
    plt.plot(x0111,y1111,'r',label='d=401')
    plt.plot(x0111,y2222,'b_',label='d=401')
    plt.legend(prop = {'size':8})
    
# compare the ground state and the vaccum.
    
def DIM(S):
    X=np.array(range(1,10**28+2,10**23),dtype=np.float64)
    X1=(10**(-28))*X
    y=((2*pi*(6*np.log(X1))**2)/9)**(1/2)
    plt.plot(X1,y,'r:')
    plt.xlabel(r"$\tilde{k}$")
    plt.ylabel(r"$\sqrt{D_{min}}$")
    plt.annotate(r"$\tilde{k}=\frac{H_0 a_0}{H_f a_f}$", xy=(10**(-28),((2*pi*(6*np.log(10**(-28)))**2)/9)**(1/2)),xytext=(0.2,300),arrowprops=dict(facecolor='red'))
# 