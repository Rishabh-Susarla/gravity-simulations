import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

S = [
    ( 1.0e11,  2.5e11,  0.8e11,  1.6e30,  25e3, 0.7*np.pi, 0.4*np.pi)
    , (-2.0e11, -1.0e11, -0.5e11,  2.0e30,  18e3, 1.2*np.pi, -0.3*np.pi) 
    , ( 0.0,     3.0e11, -1.2e11,  1.2e30,  30e3, 0.3*np.pi, 0.9*np.pi)  
    , (-2.5e11,  1.5e11,  0.0,     1.8e30,  22e3, -0.5*np.pi, 0.1*np.pi) 
    , ( 1.8e11, -2.0e11,  1.0e11,  1.4e30,  28e3, 1.5*np.pi, -0.8*np.pi)  
]

## tuple format: (x_0, y_0, z_0, mass, velocity magnitude, velocity angle from x axis, velocity angle from z axis)

G = 6.67 * 10 ** (-11) ## Gravitational constant [N * m^2/kg^2]
n = 7 ## Number of years
frames_tot = n * 200 ## Total number of frames [1]

N = len(S) ## Number of bodies [1]

## Coupled differential equation describing the components of acceleration of each body

def cde(var, t):
    
    pos = []
    vel = []

    for i in range(3 * N):
        pos.append(var[i])
        vel.append(var[i + 3 * N]) ## Leaves pos = [x1, x2, x3...] and vel = [v_x_1, v_x_2, v_x_3...]

    dydx = [0] * (6 * N)

    for i in range(3 * N):
        dydx[i] = vel[i] ## Leaves dydx = [dx1dt = v_x_1, dx2dt = v_x_2 dx3dt = v_x_3...]

    for i in range(N):
        a_net_x = 0
        a_net_y = 0
        a_net_z = 0
        for j in range(N):
            if i != j:
                dx = pos[i] - pos[j]
                dy = pos[i + N] - pos[j + N]
                dz = pos[i + 2 * N] - pos[j + 2 * N]
                r = np.sqrt((dx)**2 + (dy)**2 + (dz)**2)
                r = max(r, 1e-6)
                m = S[j][3]
                
                a_net_x += -G * m/r**3 * (dx)
                a_net_y += - G * m/r**3 * (dy)
                a_net_z += -G * m/r**3 * (dz)

        dydx[3 * N + i] = a_net_x
        dydx[4 * N + i] =  a_net_y 
        dydx[5 * N + i] = a_net_z ## Leaves dydx = [dx1dt = v_x_1, dx2dt = v_x_2...dv1xdt = -Gm2/r1^2 - Gm3/r2^2...]

    return dydx

## Initial conditions of simulation

incon = []

for i in range(3):
    for j in range(N):
        incon.append(S[j][i])

for i in range(N):
    incon.append(S[i][4] * np.cos(S[i][5]) * np.sin(S[i][6]))

for i in range(N):
    incon.append(S[i][4] * np.sin(S[i][5]) * np.sin(S[i][6]))

for i in range(N):
    incon.append(S[i][4] * np.cos(S[i][6]))
    
t = np.linspace(0, n * 31536000, frames_tot)

## Solves the differential equations to output the positions and velocities of each body

sol = odeint(cde, incon, t)

## Creates plot

fig, axis = plt.subplots(subplot_kw = {'projection': '3d'})

## Flattens solution list

sol_flat = []

for temp1 in sol:
    for temp2 in temp1:
        sol_flat.append(temp2)

## Scales axes based off of solution x, y, and z values 

sol_x = []
sol_y = []
sol_z = []

for i in range(frames_tot):
    for j in range(N):
        sol_x.append(sol_flat[j + i * 6 * N])
        sol_y.append(sol_flat[j + (6 * i + 1) * N])
        sol_z.append(sol_flat[j + (6 * i + 2) * N])

axis.set_xlim(min(sol_x), max(sol_x))
axis.set_ylim(min(sol_y), max(sol_y))
axis.set_zlim(min(sol_z), max(sol_z))

axis.set_xlabel('X [m]')
axis.set_ylabel('Y [m]')
axis.set_zlabel('Z [m]')

## Creates and differentiates planets and trails to be animated

trails = []
bodies = []
colors = ['red', 'blue', 'purple', 'green', 'yellow', 'orange']
c = 0
for planet in S:
    trail, = axis.plot([], [], color = colors[c % len(colors)], linewidth = 0.5)
    body, = axis.plot([], [], 'o', markersize = 1/3 * np.log10(planet[3]), color = colors[c % len(colors)])
    trails.append(trail)
    bodies.append(body)
    c += 1

## Updates the data of each body every frame

def update_data(frame):
    for i in range(N):
        trails[i].set_data(sol[: frame, i], sol[: frame, N + i])
        trails[i].set_3d_properties(sol[: frame, 2 * N + i])
    
    for i in range(N):
        bodies[i].set_data([sol[frame, i]], [sol[frame, N + i]])
        bodies[i].set_3d_properties([sol[frame, 2 * N + i]])
    
    return trails + bodies

animation_system = FuncAnimation(fig = fig, func = update_data, frames = len(t), interval = 0.05, repeat = False, blit = True)

animation_system.save("fiveBodySimulation.gif"
                      , writer = 'pillow'
                      , fps = 30)


plt.show()

