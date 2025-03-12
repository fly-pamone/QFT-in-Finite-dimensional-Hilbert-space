# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

def omega2(k,t,b):
    if t<=0:
       y=k**2-1/(((-t*k**2+1)**2)*((np.log(-t*k**2+1))**2))
    else:
       y=k**2-2/(3*(t/3+b)**2)
    return y

def omega(k,t,b):
    if t>=0:
       y=(1/t)**4*(k**2-1/((((1/t)*k**2+1)**2)*((np.log((1/t)*k**2+1))**2)))
    else:
       y=k**2-2/(3*(t/3+1)**2)
    return y
 
def OP1(d,k,i,b):
    phi=(2*pi/(d*(omega(k,i,b)))**(1/2))**(1/2)
    beta=(2*pi*(omega(k,i,b))**(1/2)/d)**(1/2)
    L=int((d-1)/2)
    Matrix=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.complex64)
    for o in range(-L,L+1):
        for i in range(-L,L+1):
            if i==o:
              Matrix[o+L,i+L]=(2*pi/(d*beta))*o
            else:
              Matrix[o+L,i+L]=0
    return Matrix

def OP2(d,k,i,b):
    phi=(2*pi/(d*(omega(k,i,b))**(1/2)))**(1/2)
    beta=(2*pi*(omega(k,i,b))**(1/2)/d)**(1/2)
    L=int((d-1)/2)
    Matrix=np.array([[0]*(2*L+1)]*(2*L+1),dtype=np.complex64)
    for o in range(-L,L+1):
        for i in range(-L,L+1):
            if i==o:
              Matrix[o+L,i+L]=0
            else:
              Matrix[o+L,i+L]=(1j*pi/(d*phi))*mp.csc(2*pi*L*(o-i)/(2*L+1))
    return Matrix

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



def exp(d,w,s,k,i,e,b):
    L=int((d-1)/2)
    S=np.array(range(0,s),dtype=np.complex64)
    A=OP1(d,k,i,b)
    B=OP2(d,k,i,b)
    dt=(e-i)/s
    state=main(L,w)
    for n in range(0,s):
        H=(np.dot(B,B)+omega(k,i+n*dt,b)*np.dot(A,A))/2
        A=A+1j*CM(H,A)*dt
        B=B+1j*CM(H,B)*dt
        anti=np.dot(A,A)
        state1=np.dot(anti,state)
        state2=np.dot(state,state1)/np.dot(state,state)
        S[n]=state2
    return S

def epp(d,w,s,k,i,e,b):
    L=int((d-1)/2)
    S=np.array(range(0,s),dtype=np.complex64)
    A=OP1(d,k,i,b)
    B=OP2(d,k,i,b)
    phi=OP1(d,k,i,b)
    pi=OP2(d,k,i,b)
    dt=(e-i)/s
    state=main(L,w)
    for n in range(0,s):
        B=-omega(k,i+n*dt,b)*phi*dt+B
        A=pi*dt+A
        phi=A
        pi=B
        anti=np.dot(A,A)
        state1=np.dot(anti,state)
        state2=np.dot(state,state1)/np.dot(state,state)
        S[n]=state2
    return S

def epp2(s,k,i,e,b):
    S=np.array(range(0,s),dtype=np.complex64)
    A=1/(omega(k,i,b)**2)**(1/8)
    B=1j*(omega(k,i,b)**2)**(1/8)
    phi=A
    pi=B
    dt=(e-i)/s
    for n in range(0,s):
        B=-omega(k,i+n*dt,b)*phi*dt+pi
        A=pi*dt+phi
        phi=A
        pi=B
        A_conj=A.conjugate()
        state2=A*A_conj/2
        S[n]=state2
    return S

def epp21(s,k,i,e,b):
    S=np.array(range(0,s),dtype=np.complex64)
    A=1/(omega(k,i,b))**(1/4)
    B=1j*(omega(k,i,b))**(1/4)
    phi=A
    pi=B
    dt=(e-i)/s
    for n in range(0,s):
        B=-omega(k,i+n*dt,b)*phi*dt+pi
        A=pi*dt+phi
        phi=A
        pi=B
        A_conj=A.conjugate()
        state2=A*A_conj/2
        S[n]=state2
    return S

def dim(k):
    y=(2*pi*(np.log(k))**2)/9
    return y

def pf(d,w,s,k,i,e,b):
    x=np.array(range(0,s))*(-(i-e)/(s-1))+i
    y1=np.log(exp(d,w,s,k,i,e,b))
    y2=np.log(epp2(s,k,i,e,b))
    plt.plot(x,y2,'+r-',label='d=5') 
    plt.plot(x,y1,'b',label='Infinite Dimension') 
    plt.xlabel(r"$\tilde{\eta}$")
    plt.ylabel(r"$\ln \left(\langle0| \hat{q}_k^2|0\rangle\right)$")
    plt.legend()
    
def pf1(s,k,i,e,b):
    x=np.array(range(0,s))
    
    y2=epp2(s,k,i,e,b)
    
    plt.plot(x,y2,'b')  
    
def pf2(d1,d2,d3,d4,w,s,k,i,e,b):
    x=np.array(range(0,s))*(-(i-e)/(s-1))+i
    y1=np.log(exp(d1,w,s,k,i,e,b))
    y2=np.log(exp(d2,w,s,k,i,e,b))
    y4=np.log(epp2(s,k,i,e,b))
    
    plt.plot(x,y1,'b',label='d1=3') 
    plt.plot(x,y2,'r',label='d2=5') 
  
    plt.plot(x,y4,label='Infinite Dimension') 
    plt.plot([-0.3,-0.3],[-1,4],'y',linestyle='--')
    plt.annotate(r'$\tilde{\eta}=-0.3$',
         xy=(-0.3,3), xycoords='data',
         xytext=(-100, 1), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.plot([-10,-0.01],[np.log(pi*(d2-1)*(d2+1)/(6*d2)),np.log(pi*(d2-1)*(d2+1)/(6*d2))],'r',linestyle='--')
    plt.annotate(r'$\frac{\left(d_2+1\right)\left(d_2-1\right) \pi}{6d_2}$',
         xy=(-10,np.log(pi*(d2-1)*(d2+1)/(6*d2))), xycoords='data',
         xytext=(-10, 40), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel(r"$\tilde{\eta}$")
    plt.ylabel(r"$\ln \left(\langle0| \hat{q}_k^2|0\rangle\right)$")
    plt.legend()
    
def pf3(d1,d2,d3,d4,w,s,k,i,e,b):
    x=np.array(range(0,s))*(-(i-e)/(s-1))+i
    y1=np.log(epp(d1,w,s,k,i,e,b))
    y2=np.log(epp(d2,w,s,k,i,e,b))
 
    
    plt.plot(x,y1,'y',label='d=3')
    plt.plot(x,y2,'b',label='d=5') 

    plt.xlabel(r"$\tilde{\eta}$")
    plt.ylabel(r"$\ln \left(\langle0| \hat{q}_k^2|0\rangle\right)$")
    plt.legend()    

    
def pf4(d1,d2,d3,d4,w,s,k,i,e,b):
    x=np.array(range(0,s))*(-(i-e)/(s-1))+i
    y1=np.log(exp(d1,w,s,k,i,e,b))
    y2=np.log(exp(d2,w,s,k,i,e,b))
    y4=np.log(epp2(s,k,i,e,b))
    
    plt.plot(x,y1,'b',label='d1=3') 
    plt.plot(x,y2,'r',label='d2=5') 
  
    plt.plot(x,y4,label='Infinite Dimension') 
    plt.plot([10/3,10/3],[-6,13],'y',linestyle='--')
    plt.annotate(r'$\tau=3.33$',
         xy=(10/3,12), xycoords='data',
         xytext=(100, 1), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.plot([0.1,100],[np.log(pi*(d2-1)*(d2+1)/(6*d2)),np.log(pi*(d2-1)*(d2+1)/(6*d2))],'r',linestyle='--')
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\ln \left(\langle0| \hat{Q}_k^2|0\rangle\right)$")
    plt.legend()    

        
