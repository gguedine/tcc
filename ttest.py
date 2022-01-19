from numpy.random import seed
from scipy.stats import ttest_rel
# seed the random number generator
seed(1)
# generate two independent samples

data1 = [0.5,0.8,0.6,1,0.3]
data2 = [0.9,0.9,0.8,0.3,0.8]
# compare samples
stat, p = ttest_rel(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')