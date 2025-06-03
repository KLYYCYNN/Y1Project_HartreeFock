# This could possibly make alogrithm.py nicer
# The functions take in the cooridinate of an atom and return the list of basis functions
# for diatomic compounds add the lists of the two atoms togethers to get the list of all basis functions

import numpy as np
from scipy import linalg
from scipy.special import factorial

def factorial2(x):
     if x<0:
          return 1
     else:
          return factorial(factorial(x))

class contracted_gaussians():
    def __init__(self, alpha, d, coords, L): # angular momentum L = [lx, ly, lz]
        self.alpha = alpha 
        self.d = d #contraction coefficient
        self.coords = coords
        self.L = L
        l, m, n = L
        self.l = l
        self.m = m
        self.n = n
        self.Normalisation = (2/np.pi)**0.75 * 2**(l+m+n)*alpha**((2*l + 2*m + 2*n + 3)/4) / np.sqrt(factorial2(2*l-1)*factorial2(2*m-1)*factorial2(2*n-1))


def H(coordinates): #hydrogen atom
    H_cg1a = contracted_gaussians(0.3425250914E+01, 0.1543289673E+00, coordinates, [0, 0, 0])
    H_cg1b = contracted_gaussians(0.6239137298E+00, 0.5353281423E+00, coordinates, [0, 0, 0])
    H_cg1c = contracted_gaussians(0.1688554040E+00, 0.4446345422E+00, coordinates, [0, 0, 0])
    
    return [[H_cg1a, H_cg1b, H_cg1c]]




def He(coordinates): #helium atom
     He_cg1a = contracted_gaussians(0.6362421394E+01, 0.1543289673E+00, coordinates, [0, 0, 0])
     He_cg1b = contracted_gaussians(0.1158922999E+01, 0.5353281423E+00, coordinates, [0, 0, 0])
     He_cg1c = contracted_gaussians(0.3136497915E+00, 0.4446345422E+00, coordinates, [0, 0, 0])
     
     return  [[He_cg1a, He_cg1b, He_cg1c]]
 
    
 
    
    
def C(coordinates): #carbon atom
    ls, lx, ly, lz = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]
    
    C_1s1 = contracted_gaussians(0.7161683735E+02, 0.1543289673E+00, coordinates, ls)
    C_1s2 = contracted_gaussians(0.1304509632E+02, 0.5353281423E+00, coordinates, ls)
    C_1s3 = contracted_gaussians(0.3530512160E+01, 0.4446345422E+00, coordinates, ls)
    
    C_2s1 = contracted_gaussians(0.2941249355E+01, -0.9996722919E-01, coordinates, ls)
    C_2s2 = contracted_gaussians(0.6834830964E+00, 0.3995128261E+00,  coordinates, ls)
    C_2s3 = contracted_gaussians(0.2222899159E+00, 0.7001154689E+00,  coordinates, ls)
    
    C_2p1x = contracted_gaussians(0.2941249355E+01, 0.1559162750E+00, coordinates, lx)
    C_2p1y = contracted_gaussians(0.2941249355E+01, 0.1559162750E+00, coordinates, ly)
    C_2p1z = contracted_gaussians(0.2941249355E+01, 0.1559162750E+00, coordinates, lz)
    
    C_2p2x = contracted_gaussians(0.6834830964E+00, 0.6076837186E+00, coordinates, lx)
    C_2p2y = contracted_gaussians(0.6834830964E+00, 0.6076837186E+00, coordinates, ly)
    C_2p2z = contracted_gaussians(0.6834830964E+00, 0.6076837186E+00, coordinates, lz)
    
    C_2p3x = contracted_gaussians(0.2222899159E+00, 0.3919573931E+00, coordinates, lx)
    C_2p3y = contracted_gaussians(0.2222899159E+00, 0.3919573931E+00, coordinates, ly)
    C_2p3z = contracted_gaussians(0.2222899159E+00, 0.3919573931E+00, coordinates, lz)
    
    C_1s = [C_1s1, C_1s2, C_1s3]
    C_2s = [C_2s1, C_2s2, C_2s3]
    C_2px = [C_2p1x, C_2p2x, C_2p3x]
    C_2py = [C_2p1y, C_2p2y, C_2p3y]
    C_2pz = [C_2p1z, C_2p2z, C_2p3z]
    
    return [C_1s, C_2s, C_2px, C_2py, C_2pz]
    




 
def O(coordinates): #oxygen atom
    ls, lx, ly, lz = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

    O_1s1 = contracted_gaussians(0.1307093214E+03, 0.1543289673E+00, coordinates, ls)
    O_1s2 = contracted_gaussians(0.2380886605E+02, 0.5353281423E+00, coordinates, ls)
    O_1s3 = contracted_gaussians(0.6443608313E+01, 0.4446345422E+00, coordinates, ls)
    
    O_2s1 = contracted_gaussians(0.5033151319E+01, -0.9996722919E-01, coordinates, ls)
    O_2s2 = contracted_gaussians(0.1169596125E+01, 0.3995128261E+00,  coordinates, ls)
    O_2s3 = contracted_gaussians(0.3803889600E+00, 0.7001154689E+00,  coordinates, ls)
    
    O_2p1x = contracted_gaussians(0.5033151319E+01, 0.1559162750E+00, coordinates, lx)
    O_2p1y = contracted_gaussians(0.5033151319E+01, 0.1559162750E+00, coordinates, ly)
    O_2p1z = contracted_gaussians(0.5033151319E+01, 0.1559162750E+00, coordinates, lz)
    
    O_2p2x = contracted_gaussians(0.1169596125E+01, 0.6076837186E+00, coordinates, lx)
    O_2p2y = contracted_gaussians(0.1169596125E+01, 0.6076837186E+00, coordinates, ly)
    O_2p2z = contracted_gaussians(0.1169596125E+01, 0.6076837186E+00, coordinates, lz)
    
    O_2p3x = contracted_gaussians(0.3803889600E+00, 0.3919573931E+00, coordinates, lx)
    O_2p3y = contracted_gaussians(0.3803889600E+00, 0.3919573931E+00, coordinates, ly)
    O_2p3z = contracted_gaussians(0.3803889600E+00, 0.3919573931E+00, coordinates, lz)
    
    O_1s = [O_1s1, O_1s2, O_1s3]
    O_2s = [O_2s1, O_2s2, O_2s3]
    O_2px = [O_2p1x, O_2p2x, O_2p3x]
    O_2py = [O_2p1y, O_2p2y, O_2p3y]
    O_2pz = [O_2p1z, O_2p2z, O_2p3z]
    
    return [O_1s, O_2s, O_2px, O_2py, O_2pz]





    
def Na(coordinates): # sodium atom
    Na_1s1 = contracted_gaussians(0.2507724300E+03, 0.1543289673E+00, coordinates, [0, 0, 0])
    Na_1s2 = contracted_gaussians(0.4567851117E+02, 0.5353281423E+00, coordinates, [0, 0, 0])
    Na_1s3 = contracted_gaussians(0.1236238776E+02, 0.4446345422E+00, coordinates, [0, 0, 0])
    
    Na_2s1 = contracted_gaussians(0.1204019274E+02, -0.9996722919E-01, coordinates, [0, 0, 0])
    Na_2s2 = contracted_gaussians(0.2797881859E+01, 0.3995128261E+00, coordinates, [0, 0, 0])
    Na_2s3 = contracted_gaussians(0.9099580170E+00, 0.7001154689E+00, coordinates, [0, 0, 0])
    
    Na_2p1x = contracted_gaussians(0.1204019274E+02, 0.1559162750E+00, coordinates, [1, 0, 0])
    Na_2p1y = contracted_gaussians(0.1204019274E+02, 0.1559162750E+00, coordinates, [0, 1, 0])
    Na_2p1z = contracted_gaussians(0.1204019274E+02, 0.1559162750E+00, coordinates, [0, 0, 1])
    
    Na_2p2x = contracted_gaussians(0.2797881859E+01, 0.6076837186E+00, coordinates, [1, 0, 0])
    Na_2p2y = contracted_gaussians(0.2797881859E+01, 0.6076837186E+00, coordinates, [0, 1, 0])
    Na_2p2z = contracted_gaussians(0.2797881859E+01, 0.6076837186E+00, coordinates, [0, 0, 1])
    
    Na_2p3x = contracted_gaussians(0.9099580170E+00, 0.3919573931E+00, coordinates, [1, 0, 0])
    Na_2p3y = contracted_gaussians(0.9099580170E+00, 0.3919573931E+00, coordinates, [0, 1, 0])
    Na_2p3z = contracted_gaussians(0.9099580170E+00, 0.3919573931E+00, coordinates, [0, 0, 1])
    
    Na_3s1 = contracted_gaussians(0.1478740622E+01, -0.2196203690E+00, coordinates, [0, 0, 0])
    Na_3s2 = contracted_gaussians(0.4125648801E+00, 0.2255954336E+00,  coordinates, [0, 0, 0])
    Na_3s3 = contracted_gaussians(0.1614750979E+00, 0.9003984260E+00,  coordinates, [0, 0, 0])
    
    Na_3p1x = contracted_gaussians(0.1478740622E+01, 0.1058760429E-01, coordinates, [1, 0, 0])
    Na_3p1y = contracted_gaussians(0.1478740622E+01, 0.1058760429E-01, coordinates, [0, 1, 0])
    Na_3p1z = contracted_gaussians(0.1478740622E+01, 0.1058760429E-01, coordinates, [0, 0, 1])
    
    Na_3p2x = contracted_gaussians(0.4125648801E+00, 0.5951670053E+00, coordinates, [1, 0, 0])
    Na_3p2y = contracted_gaussians(0.4125648801E+00, 0.5951670053E+00, coordinates, [0, 1, 0])
    Na_3p2z = contracted_gaussians(0.4125648801E+00, 0.5951670053E+00, coordinates, [0, 0, 1])
    
    Na_3p3x = contracted_gaussians(0.1614750979E+00, 0.4620010120E+00, coordinates, [1, 0, 0])
    Na_3p3y = contracted_gaussians(0.1614750979E+00, 0.4620010120E+00, coordinates, [0, 1, 0])
    Na_3p3z = contracted_gaussians(0.1614750979E+00, 0.4620010120E+00, coordinates, [0, 0, 1])
    
    
    chi_1s = [Na_1s1, Na_1s2, Na_1s3]
    chi_2s = [Na_2s1, Na_2s2, Na_2s3]
    chi_3s = [Na_3s1, Na_3s2, Na_3s3]
    chi_2px = [Na_2p1x, Na_2p2x, Na_2p3x]
    chi_2py = [Na_2p1y, Na_2p2y, Na_2p3y]
    chi_2pz = [Na_2p1z, Na_2p2z, Na_2p3z]
    
    
    return [chi_1s, chi_2s, chi_2px, chi_2py, chi_2pz, chi_3s]



def Cl(coordinates):  #chlorine atom
        ls, lx, ly, lz = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]

        Cl_1s1 = contracted_gaussians(0.6013456136E+03, 0.1543289673E+00, coordinates, ls)
        Cl_1s2 = contracted_gaussians(0.1095358542E+03, 0.5353281423E+00, coordinates, ls)
        Cl_1s3 = contracted_gaussians(0.2964467686E+02, 0.4446345422E+00, coordinates, ls)
        
        Cl_2s1 = contracted_gaussians(0.3896041889E+02, -0.9996722919E-01, coordinates, ls)
        Cl_2s2 = contracted_gaussians(0.9053563477E+01, 0.3995128261E+00,  coordinates, ls)
        Cl_2s3 = contracted_gaussians(0.2944499834E+01, 0.7001154689E+00,  coordinates, ls)
        
        Cl_2p1x = contracted_gaussians(0.3896041889E+02, 0.1559162750E+00, coordinates, lx)
        Cl_2p1y = contracted_gaussians(0.3896041889E+02, 0.1559162750E+00, coordinates, ly)
        Cl_2p1z = contracted_gaussians(0.3896041889E+02, 0.1559162750E+00, coordinates, lz)
        
        Cl_2p2x = contracted_gaussians(0.9053563477E+01, 0.6076837186E+00, coordinates, lx)
        Cl_2p2y = contracted_gaussians(0.9053563477E+01, 0.6076837186E+00, coordinates, ly)
        Cl_2p2z = contracted_gaussians(0.9053563477E+01, 0.6076837186E+00, coordinates, lz)
        
        Cl_2p3x = contracted_gaussians(0.2944499834E+01, 0.3919573931E+00, coordinates, lx)
        Cl_2p3y = contracted_gaussians(0.2944499834E+01, 0.3919573931E+00, coordinates, ly)
        Cl_2p3z = contracted_gaussians(0.2944499834E+01, 0.3919573931E+00, coordinates, lz)
        
        Cl_3s1 = contracted_gaussians(0.2129386495E+01, -0.2196203690E+00, coordinates, ls)
        Cl_3s2 = contracted_gaussians(0.5940934274E+00, 0.2255954336E+00,  coordinates, ls)
        Cl_3s3 = contracted_gaussians(0.2325241410E+00, 0.9003984260E+00,  coordinates, ls)
        
        Cl_3p1x = contracted_gaussians(0.2129386495E+01, 0.1058760429E-01, coordinates, lx)
        Cl_3p1y = contracted_gaussians(0.2129386495E+01, 0.1058760429E-01, coordinates, ly)
        Cl_3p1z = contracted_gaussians(0.2129386495E+01, 0.1058760429E-01, coordinates, lz)
        
        Cl_3p2x = contracted_gaussians(0.5940934274E+00, 0.5951670053E+00, coordinates, lx)
        Cl_3p2y = contracted_gaussians(0.5940934274E+00, 0.5951670053E+00, coordinates, ly)
        Cl_3p2z = contracted_gaussians(0.5940934274E+00, 0.5951670053E+00, coordinates, lz)
        
        Cl_3p3x = contracted_gaussians(0.2325241410E+00, 0.4620010120E+00, coordinates, lx)
        Cl_3p3y = contracted_gaussians(0.2325241410E+00, 0.4620010120E+00, coordinates, ly)
        Cl_3p3z = contracted_gaussians(0.2325241410E+00, 0.4620010120E+00, coordinates, lz)
        
        
        Cl_1s = [Cl_1s1, Cl_1s2, Cl_1s3]
        Cl_2s = [Cl_2s1, Cl_2s2, Cl_2s3]
        Cl_2px = [Cl_2p1x, Cl_2p2x, Cl_2p3x]
        Cl_2py = [Cl_2p1y, Cl_2p2y, Cl_2p3y]
        Cl_2pz = [Cl_2p1z, Cl_2p2z, Cl_2p3z]
        Cl_3s = [Cl_3s1, Cl_3s2, Cl_3s3]
        Cl_3px = [Cl_3p1x, Cl_3p2x, Cl_3p3x]
        Cl_3py = [Cl_3p1y, Cl_3p2y, Cl_3p3y]
        Cl_3pz = [Cl_3p1z, Cl_3p2z, Cl_3p3z]
        
        return  [Cl_1s, Cl_2s, Cl_2px, Cl_2py, Cl_2pz, Cl_3s, Cl_3px, Cl_3py, Cl_3pz]
        
