import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from smc.models import stochastic_volatility

with open(sys.argv[1], 'rb') as f:
    f.seek(0, os.SEEK_END)
    f.seek(f.tell() - (1<<23))
    f.readline()
    m = np.genfromtxt(f, delimiter=',', dtype=float)
    m = m[::8, :]

m[:, 0] -= m[0, 0]
m[:, 1] -= m[:, 1].mean()
#m[:, 
print(m)

model = stochastic_volatility(alpha=0.99, sigma=1, beta=0.5)

scale = 3600
T = int(m[-1, 0]/scale)

print('T =', T)

N = 1000
X = np.zeros((T, N))
W = np.zeros((T, N))

for t in range(T):
    if t == 0:
        y = 0
        X[t, :] = model.p_initial.rvs(N)
        W[t, :] = model.p_emission(t, X[t, :]).pdf(y)
    else:
        y0 = m[(m[:, 0] >= (t-1)*scale) * (m[:, 0] <= t*scale), 1]
        y1 = m[(m[:, 0] >= t*scale) * (m[:, 0] <= (t+1)*scale), 1]
        y = y1.mean() - y0.mean()
        resampled = np.random.choice(N, N, p=W[t-1, :])
        eliminated = 1 - len(set(resampled))/N
        print('t {}, eliminated {:.2f}% of particles'.format(t, 100*eliminated), file=sys.stderr)
        X[t-1, :] = X[t-1, resampled]
        W[t-1, :] = 1/N

        X[t, :] = model.sample_transitions(t, X[t-1, :])
        if y is None:
            W[t, :] = W[t-1, :]
        else:
            W[t, :] = W[t-1, :]*model.p_emission(t, X[t, :]).pdf(y)

    W[t, :] = W[t, :] / W[t, :].sum()

#plt.scatter(m[1:, 0], np.diff(m[:, 1]))
plt.plot(m[:, 0], m[:, 1])
print(X.mean(axis=1).shape)
plt.plot(scale*np.arange(0, T), X.mean(axis=1))
plt.show()
