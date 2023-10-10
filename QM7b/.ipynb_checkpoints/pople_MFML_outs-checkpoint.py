#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:25:30 2023

@author: vvinod
"""
def shuffle_indexes(n_trains, index_array, seed):
    '''
    Function to shuffle the indexes for repeated training to generate learning curves for MFML.

    Parameters
    ----------
    n_trains : np.ndarray(int)
        Array of training sizes across fidelities.
    index_array : np.ndarray(object)
        Complete array of indexes.
    seed : int
        Seed to be used during shuffling using numpy.

    Returns
    -------
    shuffled_index_array : np.ndarray(object)
        Array of shuffled indexes with sizes corresponding to n_trains.

    '''
    import numpy as np
    seed = np.copy(seed)
    shuffled_index_array = np.zeros((len(n_trains)), dtype=object)
    
    index_array = np.copy(index_array)
    
    #shuffle index_0
    ind_i = index_array[int(len(n_trains)-1)]    
    #####shuffle index_i#########
    np.random.seed(seed)
    np.random.shuffle(ind_i)
    ind_tilda_i = ind_i[0:n_trains[-1],:]
    shuffled_index_array[int(len(n_trains)-1)] = ind_tilda_i
    
    for i in range(len(n_trains)-2,-1,-1):
        ind_im1 = index_array[i]
        
        #set difference between im1 and i
        index_common = []
        for j in range(ind_im1.shape[0]):
            if (ind_im1[j,0] in ind_tilda_i[:,0]):
                index_common.append(j)
        ind_im1_temp = np.delete(ind_im1,index_common,axis=0)
        #shuffle ind_im1_temp
        np.random.seed(seed)
        np.random.shuffle(ind_im1_temp)
        
        #append shuffled ind_i to this
        ind_tilda_i = np.concatenate((ind_im1[index_common,:],
                                      ind_im1_temp[0:n_trains[i]-len(ind_tilda_i),:]))
        shuffled_index_array[i] = ind_tilda_i
        
        #prepare for next run of the loop
        ind_i = ind_im1
    
    return shuffled_index_array


def training_sizes(highest_train, n_fidelities, factor=2.0):
    '''
    Function to generate training dataset sizes across 
    fidelities given the number of training samples in 
    the highest fidelity.

    Parameters
    ----------
    highest_train : int
        Number of molecules/samples to be used at the highest level of 
        fidelity. PLEASE NOTE: The highest level of fidelity is the 
        lowest training number.
    n_fidelities : int
        Total number of fidelities. It is advisable to use len(fidelities) 
        as input here
    factor : float, optional
        The factor by which the l-1 fidelity training size changes with 
        respect to the training size of fidelity l. The default is 2.0.

    Returns
    -------
    size_list : np.ndarray
        Training sizes of different fidelities.

    '''
    
    import numpy as np
    assert(n_fidelities>1), "You cannot perform MFML with only one fidelity"
    #special case for 2 fidelities
    if n_fidelities==2:
        size_list = np.asarray([factor*highest_train,highest_train],dtype=int)
    #in other cases
    else:
        size_list = []
        for i in range(n_fidelities-1,-1,-1):
            size_list.append((factor**i)*highest_train)
        
        #return an np array with int entries
        size_list = np.asarray(size_list, dtype=int) #so that there is no trouble calling indices
    return size_list


def energies_differences(fidelities):
    '''
    Function to generate energy differences between fidelitie

    Parameters
    ----------
    fidelities : np.ndarray(str)
        Array of fidelities of energy to be used in the MFML.

    Returns
    -------
    energy_array : np.ndarray(object)
        Array of energies of fidelities.
    diff_array : np.ndarray(object)
        Array of energy differences.
    index_array : np.ndarray(object)
        Array of indexes corresponding to the time-stamp.
    '''
    import numpy as np
    from tqdm import tqdm
    
    energy_array = np.zeros((len(fidelities)), dtype=object)
    diff_array = np.zeros((len(fidelities)), dtype=object)
    index_array = np.zeros((len(fidelities)), dtype=object)
    
    #load lowest fidelity energy file
    E0 = np.loadtxt('Data/energies/'+fidelities[0])
    diff_array[0] = E0[:,1]
    energy_array[0] = E0
    
    #save index_0
    index_array[0] = np.asarray([np.where(E0[:,1])[0],np.where(E0[:,1])[0]]).T
    
    #run energy difference loop
    for i in tqdm(range(0,len(fidelities)-1),desc='Generating energy differences and indexing...',leave=True):
        E_diff = []
        index = []
        
        Ei = np.loadtxt('Data/energies/'+fidelities[i])
        Eip1 = np.loadtxt('Data/energies/'+fidelities[i+1])
        energy_array[i+1] = Eip1
        
        #quadratic issue here - FIX LATER?
        for j in tqdm(range(len(Ei)),desc='For Fidelity '+str(i)+' and '+str(i+1), leave=False):
            for k in range(len(Eip1)):
                if Ei[j,0]==Eip1[k,0]:
                    E_diff.append(Eip1[k,1]-Ei[j,1])
                    #where time stamp of Ei is the same as time stamp of E0
                    index.append([np.where(E0[:,0]==Eip1[k,0])[0][0], k])
        
        index = np.asarray(index,dtype=int)
        E_diff = np.asarray(E_diff,dtype=float)
        index_array[i+1] = index
        diff_array[i+1] = E_diff
        
    return energy_array, diff_array, index_array


def FCHL_kernels(sigma, X1, X2=None):
    '''
    Function to return the FCHL-specific kernels

    Parameters
    ----------
    sigma : float
        Kernel Width.
    X1 : np.ndarray
        (of) FCHL representation.
    X2 : np.ndarray, optional
        (of) FCHL representation. The default is None.

    Returns
    -------
    K : np.ndarray
        Kernel matrix for given combinations of representations.

    '''
    
    from qml.fchl import get_local_kernels, get_local_symmetric_kernels
    
    #if no X2 is specified, interpret as symmetric kernel
    if X2==None:
        K = get_local_symmetric_kernels(X1, [sigma])[0]
    
    #if X2 is specified use local_kernels routine
    else:
        K = get_local_kernels(X1,X2,[sigma])[0]
    
    return K


def kernel_generators(X1, X2=None, k_type='laplacian_kernel', sigma = 600.0, gammas=None):
    '''
    Function to return various kernels to be used in the MFML if the representations are not FCHL. 

    Parameters
    ----------
    X1 : np.ndarray
        Array of representations. Usually refers to the training representations. 
    X2 : np.ndarray, optional
        Array of representations. Refers to the test representations. If not specified, then it is considered to be same as X1.The default is None.
    k_type : str, optional
        Type of kernel to be used. The default is 'laplacian_kernel'.
    sigma : float, optional
        Kernel width. Default is 600.0
    gammas : np.array(float), optional
        Gamma parameters to be used if using sargan kernels. The default is None.

    Returns
    -------
    K : np.ndarray
        Kernel matrix of representations.

    '''
    
    Warning('FCHL representations should be used with only its specific kernel. Using FCHL representations with one of the Kernels in this module will result in incorrect learning and prediction.\n')
           
    import qml.kernels as k
    import numpy as np
    
    if type(X2)==type(None):
        X2=np.copy(X1) #make X2 a copy of X1 if X2 is not specified
    
    #generating kernels
    if k_type=='sargan_kernel':
        assert gammas!= None, 'sargan kernels require additional parameter of gammas. See qml.kernel.sargan_kernel for more details. Terminated.'
        K = k.sargan_kernel(X1, X2, sigma, gammas)
        
    elif k_type=='gaussian_kernel':
        K = k.gaussian_kernel(X1, X2, sigma)
    
    elif k_type=='laplacian_kernel':
        K = k.laplacian_kernel(X1, X2, sigma)
    
    elif k_type=='linear_kernel':
        K = k.linear_kernel(X1, X2)
    
    elif k_type=='matern_kernel':
        K = k.matern_kernel(X1, X2, sigma, order=1, metric='l2')
    
    #if none of the kernels match to what this package offers
    #then ask if user wishes to continue with laplacian kernel
    else:
        tempin = input(k_type+' kernel is currently not supported, please refer to documentation. Returning gaussian_kernel by default. Would you like to continue? [y/n]')
        if tempin in ('y','Y','yes','Yes','YES'):
            K = k.gaussian_kernel(X1, X2, sigma)
        else:
            print("Returning None type. The procedure will be terminated.")
            K = None
    
    return K


def KRR(K,y,lamda):
    '''
    Function to perform KRR given a generic kernel and corresponding learning feature.
    
    Parameters
    ----------
    K : np.ndarray (n_mols x n_mols)
        Kernel matrix generated from the representations.
    y : np.ndarray (n_mols x 1)
        DESCRIPTION.
    lamda : float
        Regularisation parameter for the KRR.
    
    Returns
    -------
    alpha : np.ndarray
        Array of KRR coefficients.
    
    '''
    
    import numpy as np
    from qml.math import cho_solve
    
    assert (K.shape[0]==y.shape[0]), "Kernels and Energy need to have same number of entries but have incompatible shape "+str(K.shape[0])+" and "+str(y.shape[0])+" respectively."
    Ktemp = np.copy(K)
    
    Ktemp[np.diag_indices_from(Ktemp)] += lamda #regularisation
    alpha = cho_solve(Ktemp,y) #perform KRR
    
    return alpha


def evaluation_kernels(X_train, X_test, rep_type, sigma, ker_type=None, gammas=None):
    '''
    Function to generate train and evaluation parent kernels to be used in the learning curve generation

    Parameters
    ----------
    X_train : np.ndarray
        Training set representations.
    X_test : np.ndarray
        Evaluation set representations.
    rep_type : str
        Type of representation being used.
    sigma : float
        Kernel width.
    ker_type : TYPE, optional
        DESCRIPTION. The default is None.
    gammas : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    K_train : TYPE
        DESCRIPTION.
    K_test : TYPE
        DESCRIPTION.

    '''
    
    #generate test and train parent kernels
    if rep_type=='FCHL':
        K_test = FCHL_kernels(sigma = sigma, X1 = X_train, X2 = X_test)
    else:
        K_test = kernel_generators(X1=X_train, X2=X_test, k_type=ker_type, sigma=sigma, gammas=gammas)
    
    return K_test


def training_kernels(X_train, rep_type, sigma, ker_type=None, gammas=None):
    '''
    Function to generate train and evaluation parent kernels to be used in the learning curve generation

    Parameters
    ----------
    X_train : np.ndarray
        Training set representations.
    rep_type : str
        Type of representation being used.
    sigma : float
        Kernel width.
    ker_type : TYPE, optional
        DESCRIPTION. The default is None.
    gammas : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    K_train : TYPE
        DESCRIPTION.
    K_test : TYPE
        DESCRIPTION.

    '''
    
    
    #generate test and train parent kernels
    if rep_type=='FCHL':
        K_train = FCHL_kernels(sigma = sigma, X1 = X_train)
    else:
        K_train = kernel_generators(X1 = X_train, X2 = None, k_type = ker_type, sigma = sigma, gammas = gammas)
    
    
    return K_train


def learning_MFML(fidelities, energy_array, index_array, diff_array, X_train_parent, rep, kernel, sigmas, regularisation):
    '''

    Parameters
    ----------
    fidelities : np.ndarray(str)
        Ordered array of strings containing file names of hte fidelities.
    energy_array : np.ndarray(object)
        Array of energies of each fidelity.
    index_array : np.ndarray(object)
        Array of indexes corresponding to the time stamps.
    diff_array : np.ndarray(object)
        Array of energy differences between fidelities.
    X_train_parent : np.ndarray(float)
        Representations for training set.
    rep : str
        Type of representation to be used.
    kernel : str
        Type of kernel to be used. Ignored if FCHL representation is used.
    sigmas : np.ndarray(float)
        Kernel widths for corresponding fidelities.
    sig_diffs : np.ndarray(float)
        Kernel widths for corresponding difference in fidelities.
    regularisation : float
        Regulariser for the KRR.
    
    
    Returns
    -------
    alpha_array : np.ndarray(object)
        Array of coeeficients of the KRR for this specific run.

    '''
    
    
    from tqdm import tqdm
    import numpy as np
    
    alpha_array = np.zeros((len(fidelities)), dtype=object)
    diff_alpha_array = np.zeros((len(fidelities)), dtype=object)
    
    #Performs MFML; and standard KRR for all fidelities
    for i in tqdm(range(0,len(fidelities)),desc='Normal KRR...',leave=False):
        
        #General KRR at all fidelities
        #load Y_i
        Y_total = energy_array[i]
        
        #sample generated training representations
        index_i = index_array[i]
        X_i = X_train_parent[index_i[:,0]]
        Y_i = Y_total[index_i[:,1]]
        
        #Training kernel for first n_trains[i] sample sizes
        K_train_i = training_kernels(X_train = X_i, rep_type = rep, sigma = sigmas[i], ker_type = kernel, gammas=None)
        
        #KRR and saving
        alpha_array[i] = KRR(K = K_train_i, y = Y_i[:,1], lamda = regularisation)
        
        #performing MFML
        #<if> condition to prevent over indexing of the fidelity set in MFML
        if i<(len(fidelities)-1):    
            #load energy differences and indexes
            deltaE_all = diff_array[i+1]
            index_ip1 = index_array[i+1]
            
            deltaE = deltaE_all[index_ip1[:,1]]
            
            #indexed representations
            X_train_indexed = X_train_parent[index_ip1[:,0]]
            
            #Training kernel for first n_trains[i+1] sample sizes
            K_train_indexed = training_kernels(X_train = X_train_indexed, rep_type = rep, sigma=sigmas[i+1], ker_type = kernel, gammas=None)
            
            #MFML KRR and saving
            alpha_pi = KRR(K = K_train_indexed, y = deltaE, lamda = regularisation)
            diff_alpha_array[i+1] = alpha_pi
            
        #at the end of this <for> loop, all f_i and f_i_i+1 have been saved
    return alpha_array, diff_alpha_array


def MAE_for_LC(fidelities, X_train_parent, X_test, y_test, n_trains, rep, kernel, sigmas, alpha_array, diff_alpha_array, index_array):
    '''
    Function to evaluate and thus generate learning curves for MFML.

    Parameters
    ----------
    fidelities : np.ndarray(str)
        Ordered array of strings containing file names of hte fidelities.
    X_train : np.ndarray(float)
        Representations of training.
    X_test : np.ndarray(float)
        Representations of testing/evaluation.
    y_test : np.ndarray(float)
        True energies of the evaluation data.
    n_trains : np.ndarray(int)
        Array of number of training samples to be used at each fidelity.
    rep : str
        Type of representation to be used.
    kernel : str
        Type of kernel to be used. Ignored if FCHL representation is used.
    sigmas : np.ndarray(float)
        Kernel width array for each fidelity.
    sig_diffs : np.ndarray(float)
        Kernel widths for corresponding difference in fidelities.
    alpha_array : np.ndarray(object)
        Array of coefficients of KRR generated by the learning_MFML module.
    diff_alpha_array : np.ndarray(object)
        Array of coefficients of KRR for the differences generated by the learning_MFML module.
    index_array : np.ndarray(object)
        Array of indexes corresponding to time stamps for fidelities.

    Returns
    -------
    mae_list : np.ndarray(float)
        List of MAE corresponding to a given level of training sizes.
    mae_baseline : np.ndarray(float)
        MAE corresponding to the baseline fidelity.
    prediction_array : np.ndarray(object)
        Array of predictions containing predictions at baseline and predictions with the complete MFML model.
    '''
    import numpy as np
    from tqdm import tqdm
    
    mae_list = []
    mae_baseline = []
    prediction_array = np.zeros((len(fidelities),2), dtype = object)
    
    for i in tqdm(range(len(fidelities)-1,-1,-1),desc='LC evaluation at train size'+str(n_trains[-1]),leave=False):
        #evaluate at f_i
        index_i = index_array[i]#np.load('Data/differences/index_'+str(i)+'.npy')
        X_i = X_train_parent[index_i[:,0]]
        K_test = evaluation_kernels(X_train = X_i, X_test = X_test, 
                                    rep_type = rep, sigma = sigmas[i], ker_type=kernel, gammas=None)
        alpha_i = alpha_array[i]#np.load('Data/alphas/a_'+str(i)+'.npy')
        
        #prediction
        predicted_energy = np.dot(alpha_i, K_test)
        prediction_array[i,0] = np.copy(predicted_energy)
        
        #MAE for KRR at fidelity
        mae_baseline.append(np.mean(np.abs(predicted_energy-y_test)))
        
        for j in tqdm(range(i,len(fidelities)-1),desc = 'For fidelity'+str(i),leave = False):
            #load index and alpha(coeffs of KRR)
            index_jp1 = index_array[j+1]
            alpha_jp1 = diff_alpha_array[j+1]
            
            X_indexed_jp1 = X_train_parent[index_jp1[:,0]]
            
            K_test_jp1 = evaluation_kernels(X_train = X_indexed_jp1, X_test = X_test, rep_type = rep, sigma = sigmas[j+1], ker_type=kernel, gammas=None)
            
            predicted_energy = predicted_energy + np.dot(alpha_jp1, K_test_jp1)
        
        #extract MAE at this level
        prediction_array[i,1] = np.copy(predicted_energy)
        MAE = np.mean(np.abs(predicted_energy-y_test))
        mae_list.append(MAE)
    return mae_list, mae_baseline, prediction_array
   

def shuffled_LC_routine(energy_array, diff_array, index_array, X_train_parent, fidelities, X_test, y_test, factor, rep, kernel, sigmas, regularisation, n_averages=10, nmax = None,seed=0):
    '''
    Function to generate learning curve plot for MFML and to save the KRR coefficients thus building the MFML model. 

    Parameters
    ----------
    X_train_parent : numpy.ndarray(float)
        The cumulative representation of the compound(s) that is to be used during training.
    fidelities : numpy.npdarray
        Ordered array of fidelities.
    X_test : np.ndarray(float)
        Representation of the compounds which are held out as the test set.
    y_test : np.ndarray(float)
        Energy/Property that is being predicted corresponding to X_test.
    factor : int
        Scaling factor of samples across fidelities.
    rep : str
        Type of representation used - must be same as used for X_train_parent and X_test.
    kernel : str
        Type of kernel to be used. Ignored if FCHL representation
    sigmas : np.ndarray(float)
        Array of kernel widths to be used across the fidelities.
    regularisation : float
        Regularisation parameter of the KRR.
    n_averages : int, optional
        Number of times to run the shuffled LC to be averaged. The default is 10.
    nmax : int, optional
        Maximum number of training samples to be used at the highest fidelity. The default value is Nonetype. This will cause the module to calculate nmax using the true shape of the energy file.
    seed : int, optional
        Seed for random shuffling in the averaging of the LC. The default is 0.

    Returns
    -------
    None.

    '''
    import numpy as np
    from tqdm import tqdm
    from datetime import datetime
    
    #higest fidelity size
    if type(nmax) == type(None):
        nmax = len(np.loadtxt('Data/energies/'+fidelities[-1])[:,1])
    #maximum number of SPLITS of training samples at highest fidelity
    max_split = int(np.floor(np.log2(nmax))) 
    
    
    #list to store intermediary data
    MAE_lists = np.zeros((len(fidelities),max_split))
    baseline_MAE_lists = np.zeros((len(fidelities),max_split))
    Model_array = np.zeros((n_averages), dtype = object)
    
    #list to store n_trains
    n_train_list = np.zeros((len(fidelities),max_split),dtype=int)
    
    for i in tqdm(range(0,n_averages),desc='Averaged Learning Curve Generation',leave=True):
        for j in tqdm(range(1,max_split+1),desc='LC routine '+str(i)+' of '+str(n_averages),leave=False):
            
            n_trains = training_sizes(highest_train = 2**j, 
                                      n_fidelities = len(fidelities), 
                                      factor = factor) #training size generation
            
            #shuffle the indexes for each ntrains size
            shuffled_indexes_array = shuffle_indexes(n_trains = n_trains, 
                                              index_array = index_array, 
                                              seed = seed)
            
            alpha_array, diff_alpha_array = learning_MFML(energy_array = energy_array, diff_array = diff_array, index_array = shuffled_indexes_array, fidelities = fidelities, X_train_parent = X_train_parent, rep = rep, kernel = kernel, sigmas = sigmas, regularisation = regularisation)
            
            mae_lists, mae_baseline, predictions = MAE_for_LC(fidelities = fidelities, X_train_parent = X_train_parent, X_test = X_test, y_test = y_test, n_trains = n_trains, rep = rep, kernel = kernel, sigmas = sigmas, alpha_array = alpha_array, index_array = shuffled_indexes_array, diff_alpha_array=diff_alpha_array)
            
            n_train_list[:,j-1] = n_trains
            MAE_lists[:,j-1] += mae_lists
            baseline_MAE_lists[:,j-1] += mae_baseline
        Model_array[i] = alpha_array #is saved for each of the n_averages run.
    
    
    np.savez('Data/MP_and_CCSD_SLATM'+kernel+'_sig'+str(sigma_list[0])+'_lam'+str(reg)+format(datetime.now().strftime("%m%d%Y_%H%M"))+'.npz', n = n_train_list, MAE = MAE_lists/n_averages, baseline_MAE = baseline_MAE_lists/n_averages, Predictions = predictions, Model = Model_array)


        

########################################################################

def main():
    import numpy as np
    seed = 0 #seed for random shuffling
    #energy differences and indexes
    energy_array, diff_array, index_array = energies_differences(fidelities = fidelity_list)
    
    #training and evaluation
    shuffled_LC_routine(energy_array = energy_array, diff_array = diff_array, 
                        index_array = index_array, X_train_parent = X_train_parent, 
                        fidelities = fidelity_list, X_test = X_test, 
                        y_test = y_test, factor = fac,
                        rep = 'SLATM', kernel = kernel_type, 
                        sigmas = sigma_list, regularisation = reg,
                        n_averages=10, nmax=2**7,seed=seed)

if __name__ == '__main__':
    import numpy as np
    
    fidelity_list = np.asarray(['MP2-STO3G.dat','MP2-631G.dat','MP2-ccpvdz.dat',
                                'CCSD-STO3G.dat','CCSD-631G.dat','CCSD-ccpvdz.dat'])
    
    sigma_list = np.full(6, 400.0)
    
    kernel_type = 'laplacian_kernel'
    
    reg = 1e-10 #laventiev regulariser for KRR
    fac = 2.0 #scaling factor across fidelities for training
    
    #load X_test
    X_test = np.load('Evaluation/GlobalSLATMEval.npy')[367:,:]
    
    #load Y_test
    y_test = np.loadtxt('Evaluation/eval_CCSD-ccpvdz.dat')[367:,1]
    
    #load X_train - will be sampled from during MFML
    X_train_parent = np.load('Data/GlobalSLATMTrain.npy')
    
    print('First sanity checks...\nX_test shape: ',X_test.shape,'\ny_test shape: ',y_test.shape,'X_train_parent shape: ',X_train_parent.shape,'\n')
    
    
    
    main()
  

