#!/usr/bin/env python
# coding: utf-8

# # A 1D diffusion model

# Here we develop a one-dimensional model of diffusion.
# -It assumes a constant diffusivity.
# -It uses a regular grid.
# -It has fixed boundary conditions.

# The diffusion equation:
# $$ \frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial x^2} $$
# The discretized version of the diffusion equation that we'll solve with our model:
# $$ C^{t+1}_x = C^t_x + {D \Delta t \over \Delta x^2} (C^t_{x+1} - 2C^t_x + C^t_{x-1}) $$
# This is the explicit FTCS scheme as described in Slingerland and Kump (2011). (Or see Wikipedia.)

# We will use two libraries, Numpy (for arrays) and Matplotlib (for plotting) that are not a part of the base Python distribution.

# In[ ]:


import numpy as np #np is the generic nickname for numpy
import matplotlib.pyplot as plt #plt is the generic nickname for matplotlib, "plot"


# Set two fixed model parameters, the diffusivity and the size of the model domain.

# In[ ]:


D = 100
Lx = 300 #Lx = Domain size, D = Diffusivity concept


# Next, set up the model grid using a NumPy array

# In[ ]:


dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)
# dx = distance between x cells or "grid spacing", arange = array range, nx = len, or length, of amount of elements.
# In this case, 600.


# In[ ]:


whos


# In[ ]:


x[0]


# In[ ]:


x[5]


# In[ ]:


x[-1]


# In[ ]:


x[0:5]


# Set the initial concentration profile for the model.
# The concentration `C` is a step function
# wiht a high value on the left,
#  a low value on the right,
#  and the step at the center of the domain.

# In[ ]:


C = np.zeros_like(x) #define new variable called 'C'. NumPy library, function zeros_like. Makes C 600 element array of floats with 0's
C_left = 500 #left side C start at 500
C_right = 0 #Right side C start at 0
C[x<= Lx//2] = C_left #values of x less than or equal to the domain size divided by 2 is equal to C_left (500)
C[x > Lx//2] = C_right #values of x greater than the domain size divided by 2 is equal to C_right (0)


# Plot the initial profile.

# In[ ]:


plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial concentration profile")


# Set up the start time of the model and the number of time steps.
# Calculate a stable time step for the model using a stability criterion.

# In[ ]:


time = 0
nt = 5000 #number of time steps
dt = 0.5 * (dx**2 / D) #time step equation itself, "**" is squared


# In[ ]:


dt


# Loop over the time steps of the model,
# solving the diffusion equation using the FTCS explicit scheme
# described above.
# The boundary conditions are fixed, so reset them at each time step.

# In[ ]:


# for loop; t is loop counter, t is also purely for model ie. no relation to nt, "t" for time; 
# C is from discretized version of the diffusion equation *way* above, Concentration
# D = dissfusivity; np.roll shifts model. 500, 500... 0, 0,... then 500 again
for t in range(0, nt):
    C += D * dt / dx**2 * (np.roll(C, -1) - 2*C + np.roll(C, 1))
    C [0] = C_left
    C[-1] = C_right


# Plot the result

# In[ ]:


plt.figure()
plt.plot(x, C, "b") #"b" is blue color
plt.xlabel("X")
plt.ylabel("C")
plt.title("Final concentration profile")


# In[ ]:


#new array example to help visualize line 18
z = np.arange(5)
z


# In[ ]:


np.roll(z, -1)


# In[ ]:


np.roll(z, +1)


# In[ ]:


z + 1


# In[ ]:


#another example of array operation
for i in range(len(z)):
    print(z[i] + 1)


# In[ ]:




