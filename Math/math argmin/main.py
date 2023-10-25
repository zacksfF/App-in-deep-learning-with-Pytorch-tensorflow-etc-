import numpy as np
import torch

av = np.array([1, 3, 5, 6, -9])
#find the max and min val
minval = np.min(av)
maxval = np.max(av)
print("min, max: %g, %g" %(minval, maxval))

#Find the argmnix
minarg = np.argmin(av)
maxarg = np.argmax(av)

print("Min, Max indices for argmix : %g, %g" %(minarg, maxarg)), print(" ")
print(f'Min val is { av[minarg] }, max val is { av[maxarg] }') #max


