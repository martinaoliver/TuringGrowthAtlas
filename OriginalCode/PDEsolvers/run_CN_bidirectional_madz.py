#%%
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.Spectral_r
import numpy as np
from scipy.sparse import spdiags, diags
from tqdm import tqdm
import copy
from sklearn import preprocessing
from scipy.linalg import solve_banded


def run_CN(L,x_gridpoints,T,t_gridpoints,params,growth, rate, initialL=1):
    J=L*x_gridpoints#Number of gridpoints
    D=[1,10] #Diffusion parameters for madz system 
    dx = float(L)/float(J-1)
    x_grid = numpy.array([j*dx for j in range(J)])
    

    N=T*t_gridpoints #Number of timepoints
    dt = float(T)/float(N-1)
    t_grid = numpy.array([n*dt for n in range(N)])

    n_species=2 #number of chemical species/variables/equations

    #Define initial concentrations of chemical system. 
    #In this case, a uniform concentration of 0.1 (with some noise) is defined through space for both chemical species. 
    U0 = []
    np.random.seed(1)

    a,b=params

    Ustar = [a+b, b/(a+b)**2] #klika65 or madz

    for index in range(n_species):
        U0.append(Ustar[index]*(1+np.random.normal(loc=0,scale=0.001,size=J)))

    #Look back at mathematical derivation above for understanding of the A and B matrices.

    def alpha(D,dt,dx,n_species):
        return [D[n]*dt/(2.*dx*dx) for n in range(n_species)]

    def A(alphan,J):
        bottomdiag = [-alphan for j in range(J-1)]
        centraldiag = [1.+alphan]+[1.+2.*alphan for j in range(J-2)]+[1.+alphan]
        topdiag = [-alphan for j in range(J-1)]
        diagonals = [bottomdiag,centraldiag,topdiag]
        A = diags(diagonals, [ -1, 0,1]).toarray()
        return A

    def B(alphan,J):
        bottomdiag = [alphan for j in range(J-1)]
        centraldiag = [1.-alphan]+[1.-2.*alphan for j in range(J-2)]+[1.-alphan]
        topdiag = [alphan for j in range(J-1)]
        diagonals = [bottomdiag,centraldiag,topdiag]
        B = diags(diagonals, [ -1, 0,1]).toarray()
        return B

    #Equations defining the klika2021 system. (eq65)
    #params is a vector including the kinetic parameters.

    def madz(u,params): 
        a,b = params
        f_u0 = a - u[0] + (u[0]**2)*u[1]
        f_u1 = b - (u[0]**2)*u[1]
        return f_u0,f_u1
        
    def plot1D(U,morphogen='both', savefig=False,filename='1'):
        if morphogen == 0:
            plt.plot(U[0], label='U')
        if morphogen ==1: 
            plt.plot(U[1], label='V')
        if morphogen == 'both': 
            fig,ax = plt.subplots()
            ax.plot(U[0], label='U', color='blue')
            ax.set_ylim(np.amin(U[0]), np.amax(U[0]))


            ax.ticklabel_format(useOffset=False)

            ax2=ax.twinx()
            ax2.plot(U[1], label='V', color='red')
            ax2.set_ylim(np.amin(U[1]), np.amax(U[1]))


            ax.ticklabel_format(useOffset=False)




        plt.ticklabel_format(useOffset=False)
        plt.xlabel('Space')
        plt.ylabel('Time')
        plt.legend()
        if savefig==True:
            plt.savefig('%s_final.png'%filename)

        plt.show()
        return fig
    plot1D(U0)

    reduced_t_grid = np.linspace(0,T,T)
    def surfpattern(results,growth, grids=[x_grid,reduced_t_grid], rate=0, morphogen = 0,savefig=False,filename='1',logResults=False, normalize=False):
        if normalize == True:
            results = [preprocessing.normalize(array, norm="l1") for array in results]

        results = np.transpose(results[morphogen])
        x_grid = grids[0]
        t_grid = grids[1]
        values = results.reshape(len(t_grid),len(x_grid))
        x, t = np.meshgrid(x_grid, t_grid)
        plt.contourf(x,t,results, cmap=cmap)
        if logResults==True:
            plt.colorbar(label='Concentration (logscale)')
        else:
            plt.colorbar()


        plt.ylabel('Time')
        plt.xlabel('Space')
        if savefig==True:
            plt.savefig('%s_overtime.png'%filename)
        plt.show()

    def exponential_growth(t, s=0.0001, initialL=initialL):
        return (initialL*np.exp(s*t))


    def linear_growth(t,s=0.00005, initialL=initialL):
        return initialL + t*s



    def CN_growth(growth,rate):
        U = copy.deepcopy(U0) 
        #copydeepcopy is useful to make sure the original U0 concentration is not modified and we can retrieve it later on if needed. 
        #we will work with U and U_new from here onwards (U_new is the updated U after calculation).

        U_record=[]
        for species_index in range(n_species):
            U_record.append(np.zeros([J, T])) #DO NOT SIMPLIFY TO U_record = [np.zeros([J, I, T])]*n_species

        #These two lists contain the A and B matrices for every chemical specie. They are adapted to the size of the field, 
        #meaning that if the field is J=3, the matrix will be 3x3.
        A_list = [A(alphan,J) for alphan in alpha(D,dt,dx,n_species)]  
        B_list = [B(alphan,J) for alphan in alpha(D,dt,dx,n_species)]  


        A_inv = [np.linalg.inv(a) for a in A_list] # Find inverse matrix of A


        #for loop iterates over time recalculating the chemical concentrations at each timepoint (ti). 
        for ti in tqdm(range(N), disable = False): 

            U_new = copy.deepcopy(U)
            f0 = madz(U, params)




            #iterate over every chemical specie when calculating concentrations. 
            for n in range(n_species):
                U_new[n] = A_inv[n].dot(B_list[n].dot(U[n]) +  f0[n]*(dt/2)) # Dot product with inverse rather than solve system of equations

            hour = ti / (N / T)
            if growth == 'exponential':
                newL = int(exponential_growth(hour,s=rate))

            if growth == 'linear':
                newL = int(linear_growth(hour,s=rate))

                
            newJ = newL*x_gridpoints
            len_pad = int((J - newJ) / 2)

            shape = np.concatenate([np.zeros(len_pad),np.ones(newJ),np.zeros(len_pad)])
            if len(shape)!=len(U_new[0]):
                            shape = np.concatenate([np.zeros(len_pad), np.ones(newJ), np.zeros(len_pad+1)])
            for n in range(n_species):
                U_new[n] = np.multiply(U_new[n], shape)

            if hour % 1 == 0 :  #only grow and record at unit time (hour)
                for n in range(n_species):
                    U_record[n][:,int(hour)] = U_new[n] #Solution added into array which records the solution over time (JxT dimensional array)

        
            U = copy.deepcopy(U_new)
        return U,U_record


    U,U_record = CN_growth(growth, rate) 

    plot1D(U, savefig=False,filename='')
    surfpattern(U_record, growth, morphogen=1, rate=rate, savefig=False,filename='',logResults=False,normalize=False)

    surfpattern(U_record, growth, morphogen=0, rate=rate, savefig=False,filename='',logResults=False,normalize=False)
  


# %%



L= 200; #Lenght of system
x_gridpoints=2


T = 200#Total time
t_gridpoints = 20
params = [0.1,0.6]


growth = 'linear'
rate=1
print('L:',L, ' x_gridpoints:', x_gridpoints,  ' T:', T, ' t_gridpoints:', t_gridpoints, ' params:',params, ' growth:', growth, 'rate:', rate)
run_CN(L, x_gridpoints, T, t_gridpoints, params, growth, rate, initialL=1)


print('L:',L, ' x_gridpoints:', x_gridpoints,  ' T:', T, ' t_gridpoints:', t_gridpoints, ' params:',params, ' growth:', growth, 'rate:', rate)
rate=0
run_CN(L, x_gridpoints, T, t_gridpoints, params, growth, rate, initialL=L)

