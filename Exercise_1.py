# Exercise 1 of the course "30230 Data Analysis and Modeling in Geoscience 
# and Astrophysics" by Nils Olsen (based on the lectures given in 2018)
# Python code by A.S. Nielsen, 2021

#%% Load packages and define data:
import numpy as np
import matplotlib.pyplot as plt

t = np.array([1,2,3,4,5]) # time in seconds
d = np.array([109.4,187.5,267.5,331.9,386.1]) # height in meters

#%% Plot data
plt.scatter(t,d)
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.show()
#%% Create design matrix

G = np.column_stack( (np.ones(len(t)),t) )

#%% Invert using Least Squares
#Given an overdetermined system, the LS solution is found as:
# m_LS = (G.T G)⁻¹ G.T d    Nils will prove this later. It is worthwile
# to learn the proof by heart.


#A number of methods exists for finding the LS-solution:
m_LS0 = np.linalg.inv(G.T @ G) @ G.T @ d 
m_LS1 = np.linalg.solve(G.T @ G,G.T @ d)
m_LS2 = np.linalg.lstsq(G,d,rcond=None) #Returns a tupple holding the 
    #solution and some diagnostics (read the docs for details. 
    #You will learn what the diagnostics means later in the course.)
m_LS2[0] #LS solution is stored at index 0.

# Note: "rcond=None" cancels a warning. I invite you to run it without and 
# read the error message if you are curious.

# Which type of solution to use? 
# In this simple example the result is the same in all cases. In general it is 
# not recommended to use inv() if it can be avoided, as it can be numerically 
# unstable.
# Method1 solves the Normal Equation " (G.T G)⁻¹ m = G.T d ". This is my 
# preffered solution, as it more numerically stable and reminds the user which
# system is actually solved.
# Method2 is (more or less) equivalent to Matlabs "\" operator, i.e. m=G\d
# and m = np.linalg.lstsq(G,d). While this method is powerfull, the user should
# carefully note that it might result in unexpected behaviour. If the system 
# of equations is underdetermined (i.e. more unknowns than equations), this
# method will not return a LS-solution, but rather a Minimum Norm solution (in
# Python). Use with care and check the diagnostics once you understand them.
#%% Plot results of linear model
d_synth = G @ m_LS1

plt.scatter(t,d,label='Data')
plt.scatter(t,d_synth,label='Linear model')
plt.xlabel("Time [s]")
plt.ylabel("Height [m]")
plt.legend()
plt.show()

#%% Find and plot residuals

r = d - d_synth
r_sq_sum = r.T @ r
rms = np.sqrt(r_sq_sum/len(r))

plt.scatter(t,r)
plt.xlabel("Time [s]")
plt.ylabel("Residuals [m]")
plt.show()
#%% End remarks
# It is instructive to compare r_sq_sum to the second value of m_LS2 to get a
# further understanding of what np.linalg.lstsq() returns.