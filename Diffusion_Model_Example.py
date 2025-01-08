import numpy as np #np is the generic nickname for numpy
import matplotlib.pyplot as plt #plt is the generic nickname for matplotlib, "plot"
D = 100
Lx = 300 #Lx = Domain size, D = Diffusivity concept
dx = 0.5
x = np.arange(start=0, stop=Lx, step=dx)
nx = len(x)
whos
x[0]
x[5]
x[-1]
x[0:5]
C = np.zeros_like(x) #define new variable called 'C'. NumPy library, function zeros_like. Makes C 600 element array of floats with 0's
C_left = 500 #left side C start at 500
C_right = 0 #Right side C start at 0
C[x<= Lx//2] = C_left #values of x less than or equal to the domain size divided by 2 is equal to C_left (500)
C[x > Lx//2] = C_right #values of x greater than the domain size divided by 2 is equal to C_right (0)
plt.figure()
plt.plot(x, C, "r")
plt.xlabel("x")
plt.ylabel("C")
plt.title("Initial concentration profile")
time = 0
nt = 5000 #number of time steps
dt = 0.5 * (dx**2 / D) #time step equation itself, "**" is squared
dt
for t in range(0, nt):
    C += D * dt / dx**2 * (np.roll(C, -1) - 2*C + np.roll(C, 1))
    C [0] = C_left
    C[-1] = C_right
plt.figure()
plt.plot(x, C, "b") #"b" is blue color
plt.xlabel("X")
plt.ylabel("C")
plt.title("Final concentration profile")
z = np.arange(5)
z
np.roll(z, -1)
np.roll(z, +1)
z + 1
for i in range(len(z)):
    print(z[i] + 1)
