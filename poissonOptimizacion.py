import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import time

N=40
h=1.0/N
lamda=0.01

def uprueba(x,y):
	k=np.exp(-(x-0.5)**2-(y-0.5)**2)
	#k[int(N/2)][int(N/2)]=1
	return k

def f(u,y,ydes):
	return 0.5*np.sum(h*h*(y-ydes)**2)+0.5*lamda*np.sum(h*h*u**2)

def resolverPoisson(u):
	a=np.ones((N-1)**2)*4
	b=np.ones((N-1)**2-1)*(-1)
	for i in range((N-1)**2-1):
		if (i+1)%(N-1)==0:
			b[i]=0
	c=np.ones((N-1)**2-(N-1))*(-1)
	A=diags([a,b,b,c,c],[0,1,-1,N-1,-N+1],format='dia')
	ys=spsolve(A,h*h*u.flatten())
	return np.reshape(ys,(N-1,N-1))
	
x=np.linspace(h,1-h,N-1)
y=np.linspace(h,1-h,N-1)



X,Y=np.meshgrid(x,y)
u=uprueba(X,Y)	
sol=uprueba(X,Y)
t0=time.time()
resp=resolverPoisson(u)

ydes=resp

i=0
u0=np.zeros(shape=(N-1,N-1))
ya=resolverPoisson(u0)
p=resolverPoisson(ya-ydes)
alpha=0.5
i=0
while(i<1000):
	u0=u0-alpha*(lamda*u0+p)
	ya=resolverPoisson(u0)
	p=resolverPoisson(ya-ydes)
	print("Actual ",f(u0,ya,ydes))
	i+=1


print("Calculado en ",time.time()-t0)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, ya, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
