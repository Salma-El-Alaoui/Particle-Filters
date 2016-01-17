import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('mathtext', default='regular')

from smc.models import stochastic_volatility

scale = 2

with open(sys.argv[1], 'rb') as f:
    f.readline()
    m = np.genfromtxt(f, delimiter=',', dtype=float)
    m = m[::scale]

model = stochastic_volatility(alpha=0.86, sigma=1, beta=0.005)

T = m.shape[0]

print('T =', T)

N = 10000
X = np.zeros((T, N))
W = np.zeros((T, N))

for t in range(T):
    if t == 0:
        y = 0
        X[t, :] = model.p_initial.rvs(N)
        W[t, :] = model.p_emission(t, X[t, :]).pdf(y)
    else:
        y = m[t] - m[t-1]
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

ts = scale*np.arange(0, T)

fig, ax1 = plt.subplots()
ax1.grid()
lns1 = ax1.plot(ts, m, label='GOOG')
ax2 = ax1.twinx()
lns2 = ax2.plot(ts, X.mean(axis=1), 'r', label='Estimated volatility')
lns = lns1 + lns2
ax1.legend(lns, [ln.get_label() for ln in lns], loc=1)
#ax1.set_xlim(0, scale*T)
ax1.set_xlabel("Day")
ax1.set_ylabel(r"Closing price (USD)")
ax2.set_ylabel(r"Volatility")
plt.savefig('googplot.pdf')
plt.show()
