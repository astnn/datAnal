# Homework 1 of the course "30230 Data Analysis and Modeling in Geoscience 
# and Astrophysics" by Nils Olsen (based on the lectures given in 2018)
# Python code by A.S. Nielsen, 2021

#%% Load packages and define data:
import numpy as np
import matplotlib.pyplot as plt

t = np.array([1,2,3,4,5,6,7,8,9,10]) #Time in seconds
d = np.array([109.4,187.5,267.5,331.9,386.1,428.4,452.2,
              498.1,512.3,513.0]) #Height in meters

#%% Plot data
plt.scatter(t,d)
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.show()
#%% Create linear solutions from last time

G_lin = np.column_stack( (np.ones(len(t)),t) ) #Design matrix of linear system

m_lin = np.linalg.solve(G_lin.T @ G_lin,G_lin.T @ d)

d_lin_synth = G_lin @ m_lin
r_lin = d-d_lin_synth
rms_lin = np.sqrt(r_lin.T @ r_lin /len(r_lin) )

#%% Plot linear solution

plt.scatter(t,d,label='Data')
plt.scatter(t,d_lin_synth,label='Linear model')
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.legend()
plt.show()

#%% Create quadratic solution, calc synth data, calc residuals

G = np.column_stack( (np.ones(len(t)),t,-0.5*t**2) ) #Design matrix quad. syst.

m = np.linalg.solve(G.T @ G, G.T @ d) # Using lin.solver and the normal eqs.

d_synth = G @ m

r = d - d_synth

rms = np.sqrt(r.T @ r/len(r))
#%% Plot solutions

plt.scatter(t,d,label='Data')
plt.scatter(t,d_lin_synth,label='Linear model')
plt.scatter(t,d_synth,label='Quadratic model')
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.legend()
plt.show()

#%% Plot residuals

#fig = plt.figure()
plt.scatter(t, r_lin, label = 'Linear model')
plt.scatter(t, r, label = 'Quadratic model')
plt.xlabel("Time [s]")
plt.ylabel("Residuals [m]")
plt.legend()
plt.savefig("Homework_1_residualPlot.pdf", bbox_inches='tight') #Store figure
plt.show()

# It might be worthwile to spend 20 mins searching out different options for
# plt.savefig(), such as size, aspect ratio, ect.