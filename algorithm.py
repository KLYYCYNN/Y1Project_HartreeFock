'''
This file containts the code for running the SCF algorithm. The inputs are given matrices & 
functions performing the operations, and its output is a .cube file.
The self-consistent field cycle, definition of the bases, and the idea for the iteration are derived from 
https://github.com/nickelandcopper/HartreeFockPythonProgram/blob/main/Hartree_Fock_Program.ipynb
'''

import numpy as np
from scipy import linalg
import matrices as mat
import intergrals as integrals
import atoms
import matplotlib.pyplot as plt

#---Define the molecule and basis functions---
class contracted_gaussians():
    def __init__(self, alpha, d, coords, L): # angular momentum L = [lx, ly, lz]
        self.alpha = alpha 
        self.d = d #contraction coefficient
        self.coords = coords
        self.Normalisation = (2*alpha/np.pi)**0.75
        self.L = L
        

def electron_density(points, density_matrix, Slater_bases):
    '''
    Calculate the charge density at an array of x values given a density matrix
    '''
    dens = 0
    for i in range(nbasis):
        for j in range(nbasis):
            n_contracted_i = len(Slater_bases[i])
            n_contracted_j = len(Slater_bases[j])

            for k in range(n_contracted_i):
                for l in range(n_contracted_j):
                    p, R_p, R_sep, K, N = integrals.gauss_product(Slater_bases[i][k], Slater_bases[j][l])
                    r2 = np.sum((points - R_p)**2, axis=1)
                    dens += density_matrix[i][j] * N * K * np.exp(-p*r2)
    return dens

def E_tot(H, P, F):
    '''
    Calculate the total energy of the system given the core hamiltonian, density matrix, and fock matrix
    Note that this is total and not electronic energy as H_core includes the nuclear-nuclear repulsion.
    '''
    E = integrals.nuclear_repulsion(atomic_masses, atomic_coordinates) #Add nuclear-nuclear repulsion energy to E
    for i in range(nbasis):
        for j in range(nbasis):
            E += P[j][i] * (H[i][j] + F[i][j]) / 2
    return E 


def SCF_cycle(H, V, S_inverse_sqrt):
    '''
    Calculates a self-consistent set of coefficients
    '''
    P=np.zeros((nbasis,nbasis)) # initial guess for density matrix
    max_cycles = 20 #Maximum cycles to prevent n infinite loop
    C_conv=10**(-6) #convergence criterion
    for i in range(max_cycles):
        Old_P = P 
        G = mat.G_matrix(P, V, nbasis)
        F = H + G #Calculate Fock matrix 
        F_diagonalised = np.dot(S_inverse_sqrt, np.dot(F, S_inverse_sqrt)) #Diagonalise Fock matrix
        C_eigenvalues, C_eigenvectors = np.linalg.eigh(F_diagonalised) #Calculate C' eigeneverything
        C = np.dot(S_inverse_sqrt, C_eigenvectors)
        P = mat.density_matrix(C, nbasis, nelectrons)
        if np.max(np.abs(P - Old_P)) < C_conv:
            print('Convergence criterion satisfied')
            return P, F
        else:
            i +=1
    return P, F


atomic_masses = [2, 1]
xval = np.arange(0.01, 3, 0.01)
x3dval = np.array([[i, 0.0, 0.0] for i in xval]) #smart way to calculate at different seperations
Energy = np.zeros([len(xval)])
for i in range(len(xval)):
    points = x3dval[i]
    atomic_coordinates = [np.array([0,0,0]), points]
    assert len(atomic_coordinates) == len(atomic_masses)
    
    Slater_bases = atoms.He(atomic_coordinates[0]) + atoms.H(atomic_coordinates[1])
    nelectrons = 2
    nbasis = len(Slater_bases)

    H = mat.H_core(Slater_bases, atomic_coordinates, atomic_masses, nbasis) #Core hamiltonian
    V = mat.V_ee(Slater_bases, nbasis) #Electron-electron repulsion
    S = mat.Overlap_matrix(Slater_bases, nbasis) #Overlap matrix
    S_inverse_sqrt = linalg.sqrtm(linalg.inv(S))
    Density_matrix, Fock_matrix = SCF_cycle(H, V, S_inverse_sqrt)
    print(Density_matrix)
    Energy[i] = E_tot(H, Density_matrix, Fock_matrix)

plt.plot(xval, Energy)
plt.xlabel('Distance, Angstrom')
plt.ylabel('Energy, Hartrees')
plt.show()


#---Graph charge density----

xval = np.arange(-3, 3, 0.01)
points = np.array([[i, 0.0, 0.0] for i in xval])
plt.plot(xval, electron_density(points, Density_matrix, Slater_bases))
plt.xlabel('Distance, Angstrom')
plt.ylabel('Electron Density (e/Bohr^3)')
plt.title('Electron Density')
plt.show()

