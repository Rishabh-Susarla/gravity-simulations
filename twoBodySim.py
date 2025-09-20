import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

S = [
    (1e10, 1e10, 1e10, 9e26, 500, np.pi/3, -5 * np.pi/4)
    , (-1e9, -1e9, -1e9, 3e26, 1000, np.pi/4, np.pi/4)
]

## tuple format: (x_0, y_0, z_0, mass, velocity magnitude, velocity angle from x axis, velocity angle from z axis)
G = 6.67 * 10 ** (-11) ## Gravitational constant [N * m^2/kg^2]
n = 7 ## Number of years [yrs]

## Coupled differential equation describing the components of acceleration of each body

def cde(var, t):
    x_1 = var[0]
    x_2 = var[1]
    y_1 = var[2]
    y_2 = var[3]
    z_1 = var[4]
    z_2 = var[5]
    v_x_1 = var[6]
    v_x_2 = var[7]
    v_y_1 = var[8]
    v_y_2 = var[9]
    v_z_1 = var[10]
    v_z_2 = var[11]

    r = np.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2 + (z_1 - z_2)**2)
    g = -G/r**3 

    dx = x_1 - x_2
    dy = y_1 - y_2
    dz = z_1 - z_2 
    
    dx1dt = v_x_1
    dx2dt = v_x_2
    dy1dt = v_y_1
    dy2dt = v_y_2
    dz1dt = v_z_1
    dz2dt = v_z_2

    dvx1dt = g * S[1][3] * (dx)
    dvx2dt = g * S[0][3] * (-dx)
    dvy1dt = g * S[1][3] * (dy)
    dvy2dt = g * S[0][3] * (-dy)
    dvz1dt = g * S[1][3] * (dz)
    dvz2dt = g * S[0][3] * (-dz)

    return [dx1dt
            , dx2dt
            , dy1dt
            , dy2dt
            , dz1dt
            , dz2dt
            , dvx1dt
            , dvx2dt
            , dvy1dt
            , dvy2dt
            , dvz1dt
            , dvz2dt]

## Initial conditions of simulation

incon = [S[0][0]
         , S[1][0]
         , S[0][1]
         , S[1][1]
         , S[0][2]
         , S[1][2]
         , S[0][4] * np.cos(S[0][5]) * np.sin(S[0][6])
         , S[1][4] * np.cos(S[1][5]) * np.sin(S[1][6])
         , S[0][4] * np.sin(S[0][5]) * np.sin(S[0][6])
         , S[1][4] * np.sin(S[1][5]) * np.sin(S[1][6])
         , S[0][4] * np.cos(S[0][6])
         , S[1][4] * np.cos(S[1][6])]

t = np.linspace(0, n * 31536000, n * 500)

## Solves the differential equations to output the positions and velocities of each body

sol = odeint(cde, incon, t)

## Plots the motions of each body, and sets graph scale to the minimum and maximum of 

fig, axis = plt.subplots(subplot_kw = {'projection': '3d'})

axis.set_xlim(min(min([temp[0] for temp in sol]), min([temp[1] for temp in sol]))
              , max(max([temp[0] for temp in sol]), max([temp[1] for temp in sol])))
axis.set_ylim(min(min([temp[2] for temp in sol]), min([temp[3] for temp in sol]))
              , max(max([temp[2] for temp in sol]), max([temp[3] for temp in sol])))
axis.set_zlim(min(min([temp[4] for temp in sol]), min([temp[5] for temp in sol]))
              , max(max([temp[4] for temp in sol]), max([temp[5] for temp in sol])))

axis.set_xlabel('X [m]')
axis.set_ylabel('Y [m]')
axis.set_zlabel('Z [m]')

animated_trail_1, = axis.plot([], [], color = 'red')
animated_trail_2, = axis.plot([], [], color = 'blue')
animated_trail_1.set_linewidth(0.5)
animated_trail_2.set_linewidth(0.5)

animated_planet_1, = axis.plot([], [], 'o', markersize = 20 * np.log10(S[0][3])/30, color = 'red')
animated_planet_2, = axis.plot([], [], 'o', markersize = 20 * np.log10(S[1][3])/30, color = 'blue')

## Updates the data of each body every frame

def update_data(frame):
    animated_trail_1.set_data(sol[: frame, 0], sol[: frame, 2])
    animated_trail_1.set_3d_properties(sol[: frame, 4])
    animated_trail_2.set_data(sol[: frame, 1], sol[: frame, 3])
    animated_trail_2.set_3d_properties(sol[: frame, 5])

    animated_planet_1.set_data([sol[frame, 0]], [sol[frame, 2]])
    animated_planet_1.set_3d_properties([sol[frame, 4]])
    animated_planet_2.set_data([sol[frame, 1]], [sol[frame, 3]])
    animated_planet_2.set_3d_properties([sol[frame, 5]])

    return animated_trail_1, animated_trail_2, animated_planet_1, animated_planet_2

animation_system = FuncAnimation(fig = fig, func = update_data, frames = len(t), interval = 0.00005, repeat = False, blit = True)


plt.show()

