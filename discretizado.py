import numpy as np
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from cvxopt import matrix,sparse
from cvxopt import solvers

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

N=40
h=1.0/N
lamda=0.01


def uprueba(x,y):
	k=np.exp(-(x-0.5)**2-(y-0.5)**2)
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
resp=resolverPoisson(u)
ydes=resp	

vec=np.ones(2*(N-1)**2)*h*h
vec[(N-1)**2:]=lamda*h*h
P=np.diag(vec)

q=np.zeros(2*(N-1)**2)
q[:(N-1)**2]=-ydes.flatten()*h*h


a=np.ones((N-1)**2)*4
b=np.ones((N-1)**2-1)*(-1)
for i in range((N-1)**2-1):
	if (i+1)%(N-1)==0:
		b[i]=0
c=np.ones((N-1)**2-(N-1))*(-1)
Aaux=np.diag(a,0)+np.diag(b,1)+np.diag(b,-1)+np.diag(c,N-1)+np.diag(c,-N+1)
A=np.block([Aaux,-h*h*np.identity((N-1)**2)])

P=matrix(P)
q=matrix(q)
A=matrix(A)
b=matrix(np.zeros((N-1)**2))

sol = solvers.qp(P, q, None, None, A, b)

print(sol['primal objective'])
respuesta=np.reshape(np.array(sol['x'])[:(N-1)**2],(N-1,N-1))


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, respuesta, rstride=1, cstride=1, 
                      cmap=cm.RdBu,linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()



