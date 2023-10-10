#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:51:28 2023

@author: vvinod
"""
import numpy as np
from tqdm import tqdm
import time
import qml.kernels as kernels
from qml.representations import get_slatm_mbtypes
import qml

#main class
class MFML:
    '''
    **Documentation for MFML**
    Multi-fidelity machine learning (MFML) is a method of using less number 
    of high-cost data and more cheaper data to achieve accuracy as offered by 
    the high accuracy data models. 
    This package implements the MFML model using KRR.
    '''    
    def __init__(self, kernel:str, sigma:float, reg:float=1e-9, 
                 gammas=None, order:int=2, metric:str='l2'):
        '''
        Initialising the MFML class

        Parameters
        ----------
        kernel : str
            Type of kernel to be used for MFML. Currently, package supports 
            'gaussian','laplacian','matern','linear',and 'sargan' kernels.
        sigma : float
            Width of the kernels. Ignored by 'linear' kernel..
        reg : float, optional
            Lavrentiev regularization parameter for KRR. The default is 1e-9.
        gammas : array, optional
            The gammas parameter for the 'sargan' kernel. 
            It is ignored by other kernel types. The default is None.
        order : int, optional
            Order of matern kernel used. Matern kernel allows order 
            0,1, and 2. The default is 2.
        metric : str, optional
            Distance metric to be used for Matern kernel. 
            It currently allows for 'l1' and 'l2' metric. 
            The default is 'l2'.

        Returns
        -------
        None.

        '''
        self.kernel = kernel
        self.sigma = sigma
        self.reg = reg
        self.gammas = gammas
        self.order = order
        self.metric = metric
        self.model = None
        self.X_train = None
        self.y_trains = None
        self.indexes = None
        self.fidelities = None
        self.mae = None
        self.rmse = None    
        self.train_time = 0.0
        self.predict_time = 0.0
        self.pbar = True
    
    def property_differences(self, fidelities=None):
        '''
        Function to generate property differences between fidelities.
        The first object in the diff_array is the baseline fidelity. This
        is not a difference between fidelities. The subsequent objects
        are differences between corresponding fidelities.

        Parameters
        ----------
        fidelities : np.ndarray(str)
            Array of fidelities of properties to be used in the MFML.

        Returns
        -------
        diff_array : np.ndarray(object)
            Array of differences of property of interest.
        index_array : np.ndarray(object)
            Array of indexes corresponding to the properties. 
            This indicates where the properties are with 
            respect to the features.
        '''
        #if user specifies fidelities
        if not isinstance(fidelities, type(None)):
            self.fidelities = np.copy(fidelities)
        #instatiate variables to store diff and indexes
        diff_array = np.zeros((len(self.fidelities)), dtype=object)
        index_array = np.zeros((len(self.fidelities)), dtype=object)
        #load lowest fidelity file
        #currently hard-coded to a directory. Should be flexible.
        E0 = np.loadtxt('Data/energies/'+self.fidelities[0])
        diff_array[0] = E0[:,1]
        #save index_0
        index_array[0] = np.asarray([np.where(E0[:,1])[0],
                                     np.where(E0[:,1])[0]]).T
        #run energy difference loop
        for i in tqdm(range(0,len(self.fidelities)-1),
                      desc='Generating energy differences and indexing...',
                      leave=True,disable=pbar):
            E_diff = []
            index = []
            #load fidelities i and i+1
            Ei = np.loadtxt('Data/energies/'+self.fidelities[i])
            Eip1 = np.loadtxt('Data/energies/'+self.fidelities[i+1])
            #loop to check location with respect to baseline fidelity
            #this is currently also assumed to be the feature index.
            #quadratic issue here - FIX LATER
            for j in range(len(Ei)):
                for k in range(len(Eip1)):
                    if Ei[j,0]==Eip1[k,0]:
                        E_diff.append(Eip1[k,1]-Ei[j,1])
                        #where time stamp of Ei is the same as 
                        #time stamp of E0
                        index.append([np.where(E0[:,0]==Eip1[k,0])[0][0], k])
            #save as np arrays of specific types.
            index = np.asarray(index,dtype=int)
            E_diff = np.asarray(E_diff,dtype=float)
            index_array[i+1] = np.copy(index)
            diff_array[i+1] = np.copy(E_diff)
        #return array of differences and corresponding indexes
        return diff_array, index_array
    
    def shuffle_indexes(self, n_trains=None, seed=0):
        '''
        Function to shuffle the indexes for MFML training. 
        This method of shuffling ensures that the nested 
        structure of the training data is retained.

        Parameters
        ----------
        n_trains : np.ndarray(int)
            Array of training sizes across fidelities.
        seed : int
            Seed to be used during shuffling using numpy.

        Returns
        -------
        shuffled_index_array : np.ndarray(object)
            Array of shuffled indexes with sizes corresponding to n_trains.

        '''
        #if no n_trains specified, use the entire training data
        if  isinstance(n_trains, type(None)):
            n_trains = np.asarray([self.y_trains[i].shape[0] 
                                   for i in range(self.y_trains.shape[0])])
            print("Training sizes not provided.", 
                  f"\nManually setting training sizes as: {n_trains}")
        #instantiate variable to store shuffled indexes
        shuffled_index_array = np.zeros((len(n_trains)), dtype=object)
        #shuffle index_0
        ind_i = self.indexes[int(len(n_trains)-1)]    
        #####shuffle index_i#########
        np.random.seed(seed)
        np.random.shuffle(ind_i)
        ind_tilda_i = ind_i[0:n_trains[-1],:]
        shuffled_index_array[int(len(n_trains)-1)] = ind_tilda_i
        #shuffle subsequent fidelities
        for i in range(len(n_trains)-2,-1,-1):
            ind_im1 = self.indexes[i]
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
                                          ind_im1_temp[0:n_trains[i]-
                                                       len(ind_tilda_i),:]))
            shuffled_index_array[i] = ind_tilda_i
            #prepare for next run of the loop
            ind_i = ind_im1
        #return the shuffled indexes
        return shuffled_index_array
        
    def kernel_generators(self, X1, X2=None):
        '''
        Function to return various kernels to be used in the MFML if the 
        representations are not FCHL. 

        Parameters
        ----------
        X1 : np.ndarray
            Array of representations. Usually refers to the training 
            representations. 
        X2 : np.ndarray, optional
            Array of representations. Refers to the test representations. 
            If not specified, then it is considered to be same as X1.
            The default is None.

        Returns
        -------
        K : np.ndarray
            Kernel matrix of representations.

        '''
        #Case for training kernel
        if isinstance(X2,type(None)):
            X2=np.copy(X1) #make X2 a copy of X1 if X2 is not specified
        #generating kernels
        if self.kernel=='sargan':
            assert self.gammas is not None, 'sargan kernels require additional parameter of gammas. See qml.kernel.sargan_kernel for more details. Terminated.'
            K = kernels.sargan_kernel(X1, X2, self.sigma, self.gammas)
        elif self.kernel=='gaussian':
            K = kernels.gaussian_kernel(X1, X2, self.sigma)
        elif self.kernel=='laplacian':
            K = kernels.laplacian_kernel(X1, X2, self.sigma)
        elif self.kernel=='linear':
            K = kernels.linear_kernel(X1, X2)
        elif self.kernel=='matern':
            K = kernels.matern_kernel(X1, X2, sigma=self.sigma, order=self.order, metric=self.metric)
        #if the kernel type does not match to what this package offers
        #then ask if user wishes to continue with gaussian kernel
        #this might be additional, could consider offering the use
        #an option to select the kernel from this list.
        else:
            tempin = input(self.kernel+' kernel is currently not supported.'
                           +' Only the following are supported:'
                           +'\n"gaussian"\n"laplacian"\n"matern"\n"sargan"\n'+
                           '"linear". Returning gaussian kernel by default.'+
                           ' Would you like to continue? [y/n]')
            if tempin in ('y','Y','yes','Yes','YES'):
                K = kernels.gaussian_kernel(X1, X2, self.sigma)
            else:
                print("Returning None type.",
                      " The procedure will be terminated.")
                K = None
        #return kernel for thi sub-model
        return K
    
    def KRR(self, K, y):
        '''
        Function to perform KRR given a generic kernel and corresponding 
        learning feature.
        
        Parameters
        ----------
        K : np.ndarray (n_mols x n_mols)
            Kernel matrix generated from the representations.
        y : np.ndarray (n_mols x 1)
            Reference property array.
        
        Returns
        -------
        alpha : np.ndarray
            Array of KRR coefficients.
        
        '''
        from qml.math import cho_solve
        #copy to prevent overwriting in class atributes
        Ktemp = np.copy(K)
        #regularization
        Ktemp[np.diag_indices_from(Ktemp)] += self.reg #regularisation
        alpha = cho_solve(Ktemp,y) #perform KRR
        #return the coeff for this submodel
        return alpha
    
    def train(self, X_train, fidelities=None, y_trains=None, indexes=None, 
              shuffle=False, seed:int=0, n_trains = None):
        '''
        Function to train an MFML model

        Parameters
        ----------
        X_train : np.ndarray (float)
            Training features.
        fidelities : np.ndarray(str), optional
            The fidelities to be used for MFML. The default is None.
        y_trains : np.ndarray(object), optional
            The different training samples of different fidelities. 
            The default is None.
        indexes : np.ndarray(object), optional
            The indexing of the features with respect to the properties 
            at fidelities. The default is None.
        shuffle : bool, optional
            whether to shuffle the training data. Shuffling is carried out
            while retaining nested structure of the training samples at
            different fidelities. The default is False.
        seed : int, optional
            Seed to be used for random shuffling of data. Ignore if
            shuffle=False. The default is 0.
        n_trains : np.ndarray(int), optional
            The number of training samples to be used at each fidelity. 
            The first entry corresponds to the training sample at the 
            lowest fidelity. The default is None.

        Returns
        -------
        None.

        '''
        #measure time of process
        tstart = time.time()
        #save X_train into the class attribure, will be used in predictions
        self.X_train = np.copy(X_train)
        self.fidelities = fidelities
        #either fidelity list or y_trains and indexes
        if isinstance(self.fidelities, type(None)):
            assert not isinstance(y_trains,type(None)) and not isinstance(indexes,type(None)), "If fidelity list is not specified, y_trains and indexes are expected."
            self.y_trains = np.copy(y_trains)
            self.indexes = np.copy(indexes) 
        else:
            self.y_trains, self.indexes = self.property_differences()
        #shuffle indexes
        if shuffle:
            self.indexes = self.shuffle_indexes(n_trains=n_trains,seed=seed)
        #variable makes referencing easier
        nfids = self.y_trains.shape[0]
        #to save coefficients of KRR
        alpha_array = np.zeros((nfids), dtype=object)
        #MFML training process
        for i in tqdm(range(nfids),
                      desc=f"Training MFML for {nfids} fidelities",
                      leave=True,disable=self.pbar):
            index_i = np.copy(self.indexes[i])
            X_i = np.copy(self.X_train[index_i[:,0]])
            Y_i = np.copy(self.y_trains[i][index_i[:,1]])
            
            K_i = self.kernel_generators(X1 = X_i, X2 = None)
            
            #perform KRR for currrent fidelity
            alpha_array[i] = self.KRR(K = K_i, y = Y_i)
        tend = time.time()
        self.train_time = tend-tstart
        self.model = np.copy(alpha_array)
    
    def scores(self, y_test, predicted):
        '''
        Function to calculate mae rmse for predictions if y_test is given

        Parameters
        ----------
        y_test : np.ndarray
            Reference values of the property from the evaluation set.
        predicted : np.ndarray
            Predicted values of the property from the evaluation set.

        Returns
        -------
        None.

        '''
        #mae 
        self.mae = np.mean(np.abs(y_test - predicted))
        #rmse
        self.rmse = np.sqrt(np.mean((y_test - predicted)**2))
            
    def predict(self, X_test, y_test = None):
        '''
        Function to predict using the MFML routine.

        Parameters
        ----------
        X_test : np.ndarray
            Test features.
        y_test : np.ndarray, optional
            Reference values of the property from the evaluation set. The default is None.

        Returns
        -------
        predicted : np.ndarray
            Predicted values of the property from the evaluation set.

        '''
        #time the process
        tstart = time.time()
        predicted = np.zeros((X_test.shape[0]), dtype=float)
        #Predict for each sub-model
        for i in tqdm(range(self.model.shape[0]),
                      desc='Predicting with MFML routine',
                      leave=True,disable=self.pbar):
            index_i = np.copy(self.indexes[i])
            model_i = np.copy(self.model[i])
            #Sample X_train
            X_train_i = np.copy(self.X_train[index_i[:,0]])
            #Generate K_train
            K_i = self.kernel_generators(X1 = X_train_i, X2 = X_test)
            #dot product of K_train and alpha
            temppred = np.dot(model_i,K_i)
            predicted[:] += np.copy(temppred)
        #measure prediction time
        tend = time.time()
        self.predict_time = tend-tstart
        #if y_test is given, perform scores
        if not isinstance(y_test,type(None)):
            try:
                self.scores(y_test, predicted)
            except Exception:
                self.mae = None
                self.rmse = None
                print('Could not calculate scores; set to default None.',
                      ' See error messages for more details.')
        #consider saving predictions into a class attribute.
        return predicted
