'''
Created on 6. jan. 2016

@author: Hlynur
'''

import modelFunctions as mf
import densityPlot as dp
import numpy as np

def bootstrapFilter(distro, pdf, N, T, y):
    #initialization
    x = []
    xLine = [0] * N
    w = [[0] * N]*T
    for i in range(0, N):
        xLine[i] = distro(0)
    x.append(xLine)
    #importance sampling step
    for t in range(1, T):
        print t
        xTilde = [0] * N
        xLine = [0] * N
        w = [0] * N
        for i in range(0, N):
            xTilde[i] = distro(t, x[t-1][i])
            w[i] = pdf(y[t], xTilde[i])
        w = [float(i)/sum(w) for i in w]
        #selection step
        for i in range(0, N):
            resampleIndex = np.random.choice(N, p=w)    
            xLine[i] = xTilde[resampleIndex]
        x.append(xLine)
    return x
    
particles = 800
T = 60
x = [0] * T
y = [0] * T
#x[0] = toyTransitions(0) 
#for t in range(0, T):
    #x[t] = toyTransitions(t, x[t-1])
    #y[t] = toyEmissions(x[t])
x[0] = mf.svTransitions(0) 
for t in range(0, T):
    x[t] = mf.svTransitions(t, x[t-1])
    y[t] = mf.svEmissions(x[t])
#xEst = bootstrapFilter(toyTransitions, toyPdf, particles, T, y)
xEst = bootstrapFilter(mf.svTransitions, mf.svPdf, particles, T, y)
for t in range(0, T):
    if t is 5 or t is 25 or t is 50 or t is 100:
        #plotPdf(xEst[t], t)
        continue
dp.lericsonPlot(xEst, x, T)