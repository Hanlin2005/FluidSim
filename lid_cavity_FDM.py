#This code simulates the lid cavity problem using a finite difference method
import numpy as np

#Initialize variables
grid_size = (2,2)
nx = 41         #number of x cells
ny = 41         #number of y cells
nt = 500        #number of time steps
dx = grid_size[0] / (nx-1)  #find dx
dy = grid_size[1] / (ny-1)  #find dy

def simulate():
    un = u.copy()
    vn = v.copy()

    un = un - un * dt/dx * (un[1:] - un[:-1])