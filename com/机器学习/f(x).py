import numpy as np
import matplotlib.pyplot as plt
def f(x):
    y=np.ones_like(x)
    i=x>0
    y[i]=np.power(x[i],x[i])
    i= x<0
    y[i]=np.power(-x[i],-x[i])
    return y
x= np.linspace(0.1,1.,101)
y=f(x)
plt.plot(x,y,"r-",label="x^x",linewidth=2)
plt.grid()
plt.legend(loc="upper left")
plt.show()
