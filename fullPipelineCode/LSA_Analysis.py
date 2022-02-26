import numpy as np
from numpy import linalg as LA
import itertools

# Head check
def stability_no_diffusion(eigenvalues):

    eigenvalues_ss = eigenvalues[0,-1]

    #oscillations in ss
    if np.iscomplex(eigenvalues_ss):
        complex_ss=True
    else:
        complex_ss=False

    #stability in  ss
    if eigenvalues_ss > 0:
        stability_ss = 'unstable'
    elif eigenvalues_ss == 0:
        stability_ss = 'lyapunov stable'
    elif eigenvalues_ss < 0:
        stability_ss = 'stable'

    #classification of ss
    if complex_ss==False:
        if stability_ss == 'stable':
            ss_class =  'stable point'
        elif stability_ss == 'lyapunov stable':
            ss_class =  'neutral point'
        elif stability_ss == 'unstable':
            ss_class =  'unstable point'
    if complex_ss==True:
        if stability_ss == 'stable':
            ss_class =  'stable spiral'
        elif stability_ss == 'lyapunov stable':
            ss_class =  'neutral center'
        elif stability_ss == 'unstable':
            ss_class =  'unstable spiral'

    return ss_class, complex_ss, stability_ss

# Curve check
def stability_diffusion(eigenvalues, complex_ss, stability_ss):
    maxeig = np.amax(eigenvalues) #highest eigenvalue of all wvn's and all the 6 eigenvalues.
    maxeig_real = maxeig.real #real part of maxeig

    if stability_ss == 'stable' or stability_ss == 'neutral':#turing I, turing I oscillatory, turing II, Unstable
        if maxeig_real <= 0:
            system_class = 'simple stable'
        elif maxeig_real > 0: #turing I, turing I oscillatory, turing II
            if np.any(eigenvalues[-1,:] == maxeig):
                system_class = 'turing II'
            elif np.all(eigenvalues[-1,:] != maxeig): #highest instability does not appear with highest wavenumber)
                if np.any(eigenvalues.imag[:,-1]>0): #if the highest eigenvalue contains imaginary numbers oscillations might be found
                    system_class = 'turing I oscillatory'
                elif np.all(eigenvalues.imag[:,-1]<=0): #if the highest eigenvalue does not contain imaginary numbers
                    system_class = 'turing I'
                else:
                    system_class = 'unclassified'
            else:
                system_class = 'unclassified'
        else:
            system_class = 'unclassified'

    elif stability_ss == 'unstable': #fully unstable or hopf
        if complex_ss==False:
                system_class = 'simple unstable'

        elif complex_ss==True and eigenvalues.imag[-1,-1]==0: #hopf
            #sign changes
            real_dominant_eig = eigenvalues.real[:,-1]
            sign_changes = len(list(itertools.groupby(real_dominant_eig, lambda real_dominant_eig: real_dominant_eig > 0))) -1
            if sign_changes == 1:
                system_class = 'hopf'

            elif sign_changes > 1:
                if np.any(eigenvalues[-1,-1] == maxeig_real):
                    system_class = 'turing II hopf'
                elif np.all(eigenvalues[-1,-1] != maxeig_real): #highest instability does not appear with highest wavenumber)
                    system_class = 'turing I hopf'
                else:
                    system_class = 'unclassified'
            else:
                system_class = 'unclassified'
        else:
            system_class = 'unclassified'
    else:
        system_class = 'unclassified'


    return system_class, maxeig
