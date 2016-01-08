'''
Created on 8. jan. 2016

@author: Hlynur
'''
import numpy as np

def toyTransitions(t, xPrev = None):
    sigma2v = 10
    sigma2one = 10
    v = np. random.normal(0, sigma2v)
    if t is 0:
        return np.random.normal(0, sigma2one)
    else:
        return 0.5 * xPrev +25 * xPrev / (1 + (xPrev)*(xPrev)) + 8 * np.cos(1.2*t) + v
def svTransitions(t, xPrev = None):
    alpha = 0.91
    alpha2 = alpha * alpha
    sigma = 1
    sigma2 = sigma*sigma
    if t is 0:
        return np.random.normal(0, sigma2/(1-alpha2))
    else:
        return np.random.normal(alpha*xPrev, sigma2)
def svEmissions(x):
    beta = 0.5
    return np.random.normal(0, beta*beta*np.exp(x))
#here we are denoting the pdf as p(y; ...) see comment for toy pdf
def svPdf(x,y):
    mu = 0
    beta = 0.5
    sigma2 = beta * beta * np.exp(x)
    return (1/np.sqrt(sigma2*2*np.pi))*np.exp(-(y-mu)*(y-mu)/(2*sigma2))
def toyEmissions(x):
    sigma2w = 1
    w = np. random.normal(0, sigma2w)
    return (x * x) / 20 + w 
#the pdf is ~ N(y;x^2/20, 1)
def toyPdf(y, x):
    mu = (x * x) / 20
    sigma2 = 1
    return (1 / np.sqrt(sigma2 * 2 * np.pi)) * np.exp(-(y - mu)*(y - mu)/(2*sigma2))

