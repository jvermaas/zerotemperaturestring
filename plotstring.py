import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, interp1d
from scipy.optimize import minimize
#Load and reshape the data
data = np.loadtxt("data.txt")

x = data[:,0].reshape(44,34)
y = data[:,1].reshape(44,34)
z = data[:,2].reshape(44,34)
gridpoints = np.vstack([x.flatten(),y.flatten()]).T
dx = x[1][0] - x[0][0]
dy = y[0][1] - y[0][0]
#Guesses for where the basins go. Since the input data is malformed, you need to also provide an intermediate point t.
a=(1.4,3.0)
b=(2.7,1.3)
t=(1.75,1.75)
npts = 20 #Number of intermediate points on the string.

#Make a figure and a axes
fig, ax = plt.subplots(1,1)



pts = np.vstack([np.hstack([np.linspace(a[0],t[0],npts),np.linspace(t[0],b[0],npts)]),np.hstack([np.linspace(a[1],t[1],npts),np.linspace(t[1],b[1],npts)])]).T
gradx, grady = np.gradient(z,dx,dy)
#Evolve points so that they respond to the gradients. This is the "zero temperature string method"
stepmax=100
for i in range(stepmax):
	#Find gradient interpolation
	Dx = griddata(gridpoints,gradx.flatten(),(pts[:,1],pts[:,0]), method='linear')
	Dy = griddata(gridpoints,grady.flatten(),(pts[:,1],pts[:,0]), method='linear')
	h = np.amax(np.sqrt(np.square(Dx)+np.square(Dy)))
	#Evolve
	pts -= 0.01 * np.vstack([Dy,Dx]).T / h
	#Reparameterize
	arclength = np.hstack([0,np.cumsum(np.linalg.norm(pts[1:] - pts[:-1],axis=1))])
	arclength /= arclength[-1]
	pts = np.vstack([interp1d(arclength,pts[:,0])(np.linspace(0,1,2*npts)), interp1d(arclength,pts[:,1])(np.linspace(0,1,2*npts))]).T
	if i % 10 == 0:
		print(i, np.sum(griddata(gridpoints,z.flatten(),(pts[:,1],pts[:,0]), method='linear')))
		#This draws the intermediate states to visualize how the string evolves.
		ax.plot(pts[:,0],pts[:,1], color=plt.cm.spring(i/float(stepmax)))

ax.plot(pts[:,0],pts[:,1], color='k', linestyle='--')

heatmap = ax.imshow(z, cmap=plt.cm.rainbow, vmin = np.nanmin(z), vmax=np.nanmin(z)+25, origin='lower', aspect='auto', extent = (y[0][0], y[-1][-1],x[0][0], x[-1][-1]), interpolation="bicubic")
heatmap.cmap.set_over('white')
ax.autoscale(False)

bar = fig.colorbar(heatmap)
bar.set_label("Free Energy (kcal/mol)", rotation=90, fontname="Avenir", fontsize=14)
bar.set_ticks([-5,0,5,10,15,20])

ax.xaxis.set_ticks([1.25,1.50,1.75,2.0,2.25,2.50,2.75])
ax.yaxis.set_ticks([1.25,1.50,1.75,2.0,2.25,2.50,2.75,3.0,3.25])
fig.savefig("demo.png", dpi=450)
#FYI this can also save .pdf or .svg or .eps, if you want to go in a vectorly direction.

fig, ax = plt.subplots(1,1)
ax.plot(np.linspace(0,1,2*npts),griddata(gridpoints,z.flatten(),(pts[:,1],pts[:,0]), method='linear'))
ax.set_ylabel("Free Energy (kcal/mol)")
ax.set_xlabel("Reaction Progress")
fig.savefig("1Dpmf.png")
exit()
