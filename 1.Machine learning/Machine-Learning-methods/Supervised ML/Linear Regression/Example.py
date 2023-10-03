import matplotlib.pyplot as plt 
from scipy import stats

x  = [5,7,8,9,8,7,5,6,6]
y = [66,99,66,11,77,88,22, 55, 58]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def myfunc(x):
    return slope * x + intercept
mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()