import numpy as np
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
matplotlib.use("TkAgg")



def lorenz(r0, t0 = 0):
    
    # initial conditions
    sigma=10.0
    beta=8.0/3.0
    rho=28.0
    
    # unpack position vector
    x=r0[0]
    y=r0[1]
    z=r0[2]
    
    # compute the time derivatives of the x, y, and z coordinates        
    return sigma*(y-x) , (x*(rho-z))-y , (x*y)-(beta*z)



t = np.linspace(0, 20, num=2000)

# initial position vector
r0 = [1, 1, 1]

# integrate the ODEs using Scipy's odeint
r_t = integrate.odeint(lorenz, r0, t)
"""
#plot the trajectory
fig = plt.figure()
ax = fig.gca(projection='3d')  # gca (get current axes) 
ax.plot(r_t[:,0], r_t[:,1], r_t[:,2],'--b')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')



plt.show()
"""


"""_______________________NOW ANIMATE IT_____________________________"""

#Number of particles to simulate
N_p = 3


# Generate random initial positions for the N_p particles, uniformly distributed from -10 to 10
#r0 = -10 + 20 * np.random.random_sample((N_p, 3))
#print(r0)


# slightly different inital positions
r0=[[1 , 1 , 20.01],
    [1 , 1 , 20.00],
    [1 , 1 , 19.99]]
t = np.linspace(0, 700, 14000)

# Solve the system of equations to obtain the trajectories
# and convert the trajectories into an numpy array with np.asarray and list comprehension
r_t = np.asarray([integrate.odeint(lorenz, r0_i, t) for r0_i in r0])

# Create figure and 3D axes
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))

# Choose point of view of the animation
ax.view_init(40, 0)

# Choose the color map of the particles, the style of the trace and point particle
colors = plt.cm.viridis(np.linspace(0, 1, N_p))
traces = sum([ax.plot([], [], [], '-', c=c) for c in colors], [])
points = sum([ax.plot([], [], [], 'o', c=c) for c in colors], [])

# This takes every frame number and creates the trace and new point
# it also rotates the point of view
def animate(i):

    i = (2 * i) % r_t.shape[1]

    for trace, pt, r in zip(traces, points, r_t):
        x, y, z = r[:i].T
        trace.set_data(x, y)
        trace.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(15, 0.3 * i)
    fig.canvas.draw()
    return traces + points
    
# Plot the background of each frame
def init():
    for trace, pt in zip(traces, points):
        trace.set_data([], [])
        trace.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return traces + points

animation = animation.FuncAnimation(fig, animate, init_func=init, frames=500, interval=10, blit=True)

# Save as .mp4 using FFmpeg
#animation.save('lorenz_attractor.mp4', fps=20, extra_args=['-vcodec', 'libx264'])

#plt.show()
