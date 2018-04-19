import matplotlib.pyplot as plot
import numpy as np
p=0.45 #初始值c的实部
q=-0.1428 #初始值c的虚部
N=800 #最大迭代次数
M=100 #迭代区域的界值
a=3.0 #绘制图的横轴大小
b=3.0 #绘制图的纵轴大小
step=0.005 #绘制点的步长

def iterate(z,N,M):
    z=z*z+c
    for i in range(N):
        if abs(z)>M:
            return i
        z=z*z+c
    return N

c=p+q*1j
i=np.arange(-a/2.0,a/2.0,step)
j=np.arange(b/2.0,-b/2.0,-step)
I,J=np.meshgrid(i, j)
ufunc=np.frompyfunc(iterate,3,1)
Z=ufunc(I+1j*J,N,M).astype(np.float)
plot.imshow(Z,extent=(-a/2.0,a/2.0,-b/2,b/2.0))
cb = plot.colorbar(orientation='vertical',shrink=1)
cb.set_label('iteration counts')
plot.show()