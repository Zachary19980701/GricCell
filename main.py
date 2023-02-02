import numpy as np
import sympy
import scipy
import matplotlib.pyplot as plt
import math

N = 32 #size of neural
s_0 = np.random.normal(size=(N, N)) #init neural martix
#s_0 = np.ones((N, N))
s_old = s_0

    

def five_point_lappace_diff(s, dx):
    top = np.roll(s, -dx, axis=1)
    bottom = np.roll(s, dx, axis=1)
    left = np.roll(s, dx, axis=0)
    right = np.roll(s, -dx, axis=0)
    grad = (top+bottom+left+right-4*s)/(dx)
    
    return grad
    

def filt(x):  #filt_martix, E(3)
    #init param
    global N
    a = 1
    beta = 3/(169)
    gamma = 1.05*beta
    #compute w0
    w0 = a*math.exp(-gamma*abs(x)*abs(x)) - math.exp(-beta*abs(x)*abs(x))
    return w0

def compute_weight_temp(i, j): #compute weight i and j 
    global N
    pos_i_x = int(i%N)
    pos_i_y = int(i/N)
    
    pos_j_x = int(j%N)
    pos_j_y = int(j/N)
    dist = math.sqrt(math.pow(pos_i_x-pos_j_x, 2) + math.pow(pos_i_y-pos_j_y, 2))
    #print(dist)
    w = filt(dist)
    
    return w

def compute_weight(x, y):
    dist = math.sqrt(math.pow(x, 2)+math.pow(y, 2))
    w = filt(dist)
    return w

weight = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        weight[i, j] = compute_weight(abs(i-N/2), abs(j-N/2))
        
f_w = np.fft.fft2(weight)

        
plt.matshow(weight)
plt.show()

s_new = np.zeros((N, N))
dt = 0.01
tau = 10
for i in range(20000):
    #grad = five_point_lappace_diff(s_old, 1)
    #martix = np.zeros((N, N))
    martix = (np.real(np.fft.ifft2((f_w*np.fft.fft2(s_old)))))*dt/tau
    for j in range(N):
        for k in range(N):
            martix[j, k] = max(0, martix[j, k])
    #s_new = (martix - grad - s_old)*dt/tau + s_old
    s_new = s_old + martix
    s_old = s_new
    '''if i%50==0:
        plt.clf()
        plt.matshow(s_new)
        plt.show()
        plt.pause(0.5)'''
plt.matshow(s_new)
plt.show()
#plt.pause(0.1)