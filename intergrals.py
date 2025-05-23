'''
This file contains the integrals and forms the matrices from the input values
'''

#necessary packages
import sys 
import numpy as np
from scipy import special
from scipy import linalg

#input files - basis, num of electrons, separation, etc

def gauss_product(Gaussian_1, Gaussian_2):
    # Most of the integrals involve taking the product between two Gaussians.
    # Two unnormalised gaussians g(r-R_a) and g(r-R_b) combine to give Kg(r-R_p)
    # Where K, the proportionality constant, is defined as below. See pp410 Modern Quantum Chemistry
    alpha, d_a, R_a, N_a = Gaussian_1.alpha, Gaussian_1.d, Gaussian_1.coords, Gaussian_1.Normalisation
    beta, d_b, R_b, N_b = Gaussian_2.alpha, Gaussian_2.d, Gaussian_2.coords, Gaussian_2.Normalisation
    p = alpha + beta
    R_p = (alpha * R_a + beta * R_b) / p
    R_sep = np.dot(R_a-R_b, R_a-R_b)
    K = np.exp( -alpha*beta/p * R_sep)
    N = N_a * N_b * d_a * d_b
    return p, R_p, R_sep, K, N

def one_electron_kinetic(Gaussian_1, Gaussian_2):
    '''
    This function calculates the one-electron integral for the kinetic energy of the electron (-1/2*laplace) and returns its value.
    Inputs: r,s - indeces of the basis functions being integrated.
    '''
    alpha, d_a, R_a, N_a = Gaussian_1.alpha, Gaussian_1.d, Gaussian_1.coords, Gaussian_1.Normalisation
    beta, d_b, R_b, N_b = Gaussian_2.alpha, Gaussian_2.d, Gaussian_2.coords, Gaussian_2.Normalisation
    p, R_p, R_sep, K, N = gauss_product(Gaussian_1, Gaussian_2)
    Term1 = alpha * beta / p
    Term2 = 3 - 2*alpha*beta/p * R_sep
    Term3 = (np.pi/p)**1.5
    return N*Term1*Term2*Term3*K

def boys(t):
    # This variant of the boys function is needed to evaluate nuclear-electron potential
    if t == 0:
        return 1
    else:
        return 0.5 * (np.pi/t)**0.5 * special.erf(t**0.5)

def one_electron_potential(Gaussian_1, Gaussian_2, R_c, Z):
    '''
    This is function calculates the one-electron integral for the potential term (Z/r) and returns its numerical value.
    Inputs are the two contracted gaussians, and the atomic mass and coordinate it is currently being interacted with
    '''
    p, R_p, R_sep, K, N = gauss_product(Gaussian_1, Gaussian_2)
    Term1 = -2*np.pi*Z/p
    boys_input = p * np.dot(R_p - R_c, R_p-R_c)
    Term2 = boys(boys_input)
    return N*Term1*K*Term2

def two_electron(Gaussian_1, Gaussian_2, Gaussian_3, Gaussian_4):
    '''
    Calculates the two-electron integral and returns its value (psi psi 1/r psi psi)
    '''
    p, R_p, R_ab, K_ab, N_ab = gauss_product(Gaussian_1, Gaussian_2)
    q, R_q, R_cd, K_cd, N_cd = gauss_product(Gaussian_3, Gaussian_4)
    Term1 = 2*np.pi**2.5 / (p * q * np.sqrt(p+q))
    Term2 = K_ab*K_cd
    boys_input = p*q / (p+q) * np.linalg.norm(R_p-R_q)**2
    Term3 = boys(boys_input)
    N = N_ab*N_cd
    return N * Term1 * Term2 * Term3

def overlap(Gaussian_1,Gaussian_2):
    '''
    Returns the overlap matrix of the basis set
    input: m, n - indeces of functions
    '''
    p, R_p, R_sep, K, N = gauss_product(Gaussian_1, Gaussian_2)
    return N*(np.pi/p)**1.5*K

def nuclear_repulsion(atomic_masses, atomic_coordinates):
    natoms = len(atomic_masses)
    E_nn = 0 
    if natoms == 1:
        return 0
    else:
        for i in range(natoms):
            for j in range(i+1, natoms):
                Rij = np.linalg.norm(atomic_coordinates[i] - atomic_coordinates[j])
                E_nn += atomic_masses[i]*atomic_masses[j]/Rij
        return E_nn

#wrirte all teh values out for each integral into a file

def ss_overlap(g1, g2):
    alpha1, alpha2 = g1.alpha, g2.alpha
    p, q = alpha1 + alpha2, alpha1 * alpha2           #define parameters
    r1, r2 = np.array(g1.coords), np.array(g2.coords)   
    N1, N2  = g1.d, g2.d  #contraction coeff
    c1, c2 = ((2*alpha1)/np.pi) ** 0.75, ((2*alpha2)/np.pi) ** 0.75   #normalization factors
    
    dstc_sqr = np.dot( r1-r2, r1-r2 )                  #distence squared
    K_12 = np.exp(-(q/p) * dstc_sqr ) #exponential prefactor

    inner_prod = K_12 * (N1*N2*c1*c2) * (np.pi/p)**1.5

    return inner_prod

def sp_overlap(g1, g2):
    alpha1, alpha2 = g1.alpha, g2.alpha
    p, q = alpha1 + alpha2, alpha1 * alpha2           #define parameters
    r1, r2 = np.array(g1.coords), np.array(g2.coords)   
    N1, N2  = g1.d, g2.d  #contraction coeff
    dstc_sqr = np.dot( r1-r2, r1-r2 )                  #distence squared
    K_12 = np.exp(-(q/p) * dstc_sqr ) #exponential prefactor

    if g1.L == [0, 0, 0] :# if g2 is the p-type gaussian
        c1, c2 = ((2*alpha1)/np.pi) ** 0.75, 2**1.75 * alpha2**1.25 * np.pi**(-0.75) #normalization factors
        if g2.L[0] == 1:#if g2 is px
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha1 * p**(-2.5) * (r1[0] - r2[0]) * np.pi**1.5
        elif g2.L[1] == 1: # if g2 is py
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha1 * p**(-2.5) * (r1[1] - r2[1]) * np.pi**1.5
        else: # if g2 is pz
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha1 * p**(-2.5) * (r1[2] - r2[2]) * np.pi**1.5

    else: #g1 is p-type
        c1, c2 = 2**1.75 * alpha1**1.25 * np.pi**(-0.75), ((2*alpha2)/np.pi) ** 0.75
        if g1.L[0] == 1:#if g1 is px
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha2 * p**(-2.5) * (r2[0] - r1[0]) * np.pi**1.5
        elif g1.L[1] == 1: # if g1 is py
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha2 * p**(-2.5) * (r2[1] - r1[1]) * np.pi**1.5
        else: #if G1 is pz
            inner_prod = K_12 * (N1*N2*c1*c2) * alpha2 * p**(-2.5) * (r2[2] - r1[2]) * np.pi**1.5

    return inner_prod

def pp_overlap(g1, g2):
    alpha1, alpha2 = g1.alpha, g2.alpha
    p, q = alpha1 + alpha2, alpha1 * alpha2           #define parameters
    r1, r2 = np.array(g1.coords), np.array(g2.coords)   
    N1, N2  = g1.d, g2.d  #contraction coeff
    dstc_sqr = np.dot( r1-r2, r1-r2 )                  #distence squared
    K_12 = np.exp(-(q/p) * dstc_sqr ) #exponential prefactor
    c1, c2 = 2**1.75 * alpha1**1.25 * np.pi**(-0.75), 2**1.75 * alpha2**1.25 * np.pi**(-0.75) #normalization factors
    L1, L2 = g1.L, g2.L

    if L1 == [1, 0, 0] and L2 == [1, 0, 0]:#both px
        inner_prod = K_12 * (N1*N2*c1*c2) * ( 1/(2*p) - q * ((r1[0]-r2[0])/p)**2 ) * (np.pi/p)**1.5
    elif L1 == [0, 1, 0] and L2 == [0, 1, 0]:#both py
        inner_prod = K_12 * (N1*N2*c1*c2) * ( 1/(2*p) - q * ((r1[1]-r2[1])/p)**2 ) * (np.pi/p)**1.5
    elif L1 == [0, 0, 1] and L2 == [0, 0, 1]:#both pz
        inner_prod = K_12 * (N1*N2*c1*c2) * ( 1/(2*p) - q * ((r1[2]-r2[2])/p)**2 ) * (np.pi/p)**1.5
    elif L1 == [1, 0, 0] and L2 == [0, 1, 0]:#g1 px, g2 py
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r2[0] - r1[0]) * (r1[1] - r2[1])
    elif L1 == [1, 0, 0] and L2 == [0, 0, 1]:#g1 px, g2 pz
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r2[0] - r1[0]) * (r1[2] - r2[2])
    elif L1 == [0, 1, 0] and L2 == [1, 0, 0]: #g1 py, g2 px
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r1[0] - r2[0]) * (r2[1] - r1[1])
    elif L1 == [0, 1, 0] and L2 == [0, 0, 1]: #g1 py, g2 pz
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r1[2] - r2[2]) * (r2[1] - r1[1])
    elif L1 == [0, 0, 1] and L2 == [1, 0, 0]: #g1 pz, g2 px
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r1[0] - r2[0]) * (r2[2] - r1[2])
    else: #g1 pz, g2 py
        inner_prod = K_12 * (N1*N2*c1*c2) * q * p**(-3.5) * np.pi**1.5 * (r1[1] - r2[1]) * (r2[2] - r1[2])
        
    return inner_prod
