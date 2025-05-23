'''
This file containts the matrix operations used in during the Hartree-Fock method. Its inputs are the two- and one-electron integrals, 
and outputs functions containing the matrix operations.
References: 
'''

#necessary libraries
import sys
import numpy as np
from scipy import linalg

#import the integrals
import intergrals as integrals

def Kinetic(Slater_bases, nbasis): #Kinetic Energy matrix
    T = np.zeros([nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            n_contracted_i = len(Slater_bases[i])
            n_contracted_j = len(Slater_bases[j])

            for k in range(n_contracted_i):
                for l in range(n_contracted_j):
                    T[i, j] += integrals.one_electron_kinetic(Slater_bases[i][k], Slater_bases[j][l])
    return T

def Nuclear_electron(Slater_bases, atomic_coordinates, atomic_masses, nbasis):
    V = np.zeros([nbasis, nbasis])
    natoms = len(atomic_masses)
    for i in range(nbasis):
        for j in range(nbasis):
            n_contracted_i = len(Slater_bases[i])
            n_contracted_j = len(Slater_bases[j])

            for k in range(n_contracted_i):
                for l in range(n_contracted_j):
                    for m in range(natoms):
                        V[i, j] += integrals.one_electron_potential(Slater_bases[i][k], Slater_bases[j][l], atomic_coordinates[m], atomic_masses[m])
    return V

#V = V_ne(Slater_bases, atomic_coordinates, atomic_masses)
#print(V)

def H_core(Slater_bases, atomic_coordinates, atomic_masses, nbasis):
    '''
    Kinetic, nuclear-nuclear, and nuclear-electron repulsion
    '''
    T = Kinetic(Slater_bases, nbasis)
    V_nucec = Nuclear_electron(Slater_bases, atomic_coordinates, atomic_masses, nbasis) 
    H_core = T + V_nucec
    return H_core

def Overlap_matrix(Slater_bases, nbasis):
    S = np.zeros([nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            n_contracted_i = len(Slater_bases[i])
            n_contracted_j = len(Slater_bases[j])

            for k in range(n_contracted_i):
                for l in range(n_contracted_j):
                    S[i, j] += integrals.overlap(Slater_bases[i][k], Slater_bases[j][l])
    return S

def overlap_matrix(Slater_bases, nbasis):
    S = np.zeros([nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            n_contracted_i = len(Slater_bases[i])
            n_contracted_j = len(Slater_bases[j])

            for k in range(n_contracted_i):
                for l in range(n_contracted_j):
                    L1, L2 = Slater_bases[i][k].L, Slater_bases[j][l].L
# if else statement to deterimine which function is used
                    if L1 == [0, 0, 0] and L2 == [0, 0, 0]:
                        S[i, j] += integrals.ss_overlap(Slater_bases[i][k], Slater_bases[j][l])
                    elif 1 in L1 and 1 in L2:
                        S[i, j] += integrals.pp_overlap(Slater_bases[i][k], Slater_bases[j][l])
                    else:
                        S[i, j] += integrals.sp_overlap(Slater_bases[i][k], Slater_bases[j][l])

    return S

#S = Overlap_matrix(Slater_bases)
#print(S)

def Transformation_matrix():
    return

def V_ee(Slater_bases, nbasis):
    V_elecelec = np.zeros([nbasis, nbasis, nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            for k in range(nbasis):
                for l in range(nbasis):
                    n_contracted_i = len(Slater_bases[i])
                    n_contracted_j = len(Slater_bases[j])
                    n_contracted_k = len(Slater_bases[k])
                    n_contracted_l = len(Slater_bases[l])

                    for ii in range(n_contracted_i):
                        for jj in range(n_contracted_j):
                            for kk in range(n_contracted_k):
                                for ll in range(n_contracted_l):
                                    V_elecelec[i, j, k, l] += integrals.two_electron(Slater_bases[i][ii], Slater_bases[j][jj], Slater_bases[k][kk], Slater_bases[l][ll])
    return V_elecelec

#V_elecelec = V_ee(Slater_bases)
#print(V_elecelec)

def G_matrix(density_matrix, V_ee, nbasis):
    G = np.zeros([nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            for k in range(nbasis):
                for l in range(nbasis):
                    J = V_ee[i, j, k, l]
                    K = V_ee[i, l, k, j]
                    G[i, j] += density_matrix[k, l] * (J - 0.5*K)
    return G

def density_matrix(Basis_coefficients, nbasis, nelectrons):
    P = np.zeros([nbasis, nbasis])
    half = nelectrons // 2
    for i in range(nbasis):
        for j in range(nbasis):
            for k in range(half):
                P[i, j] += 2*Basis_coefficients[i][k]*Basis_coefficients[j][k]
                #Assume that C is real and hence Cdagger = C
    return P 
