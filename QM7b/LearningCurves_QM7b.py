import numpy as np
from tqdm import tqdm
from Model_MFML import ModelMFML as MFML
 

def LC_routine(y_trains, indexes, X_train, X_test, X_val, y_test, y_val, navg=10):
    nfids = y_trains.shape[0]

    MAEs_OLS = np.zeros((7),dtype=float) #for OLS MFML
    #MAEs_LRR = np.zeros((7),dtype=float) #for LRR MFML
    MAEs_def = np.zeros((7),dtype=float) # for default MFML
    
    for i in tqdm(range(navg),desc='Averaged Learning Curve...', leave=False):
        mae_ntr_OLS = []
        #mae_ntr_LRR = []
        mae_ntr_def = []
        for j in tqdm(range(1,8),leave=False, desc='Loop over training sizes'):
            n_trains = np.asarray([2**(j+5),2**(j+4),2**(j+3),2**(j+2),2**(j+1),2**j])[6-nfids:]
            ###TRAINING######
            model = MFML(reg=reg, kernel=kernel, sigma=sigma, p_bar=False)
            model.train(X_train_parent=X_train, 
                        y_trains=y_trains, indexes=indexes, 
                        shuffle=True, n_trains=n_trains, seed=i)
            ##########OLS##########
            predsOLS = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='OLS', copy_X= True, 
                                     fit_intercept= False)
            mae_ntr_OLS.append(model.mae)
            
            '''
            ##########LRR#########
            predsLRR = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='LRR', alpha=1e-6, 
                                     fit_intercept=True, copy_X=True, 
                                     max_iter=None, tol=1e-4, solver='lsqr', 
                                     random_state=0)
            mae_ntr_LRR.append(model.mae)
            
            
            '''
            ######default#########
            predsdef = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='default')
            mae_ntr_def.append(model.mae)
            '''
            ##########LASSO#########
            predsLasso = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='LASSO', alpha=1e-1,
                                       fit_intercept=False, precompute=False, 
                                       copy_X=True, max_iter=60000, 
                                       tol=1e-5, warm_start=False, 
                                       positive=False, random_state=0, 
                                       selection='random')
            mae_ntr_LRR.append(model.mae)
            '''
    
        #store each avg run MAE  
        mae_ntr_OLS = np.asarray(mae_ntr_OLS)
        #mae_ntr_LRR = np.asarray(mae_ntr_LRR)
        mae_ntr_def = np.asarray(mae_ntr_def)
        
        MAEs_OLS += mae_ntr_OLS
        #MAEs_LRR += mae_ntr_LRR
        MAEs_def += mae_ntr_def
        
    #return averaged MAE
    MAEs_OLS = MAEs_OLS/navg
    #MAEs_LRR = MAEs_LRR/navg
    MAEs_def = MAEs_def/navg
    return MAEs_OLS, MAEs_def#, MAEs_LRR, MAEs_def


def save_coeffs(y_trains, indexes, X_train, X_test, X_val, y_test, y_val):
    nfids = y_trains.shape[0]
    coeff_list = np.zeros((7,2*nfids-1),dtype=float)
    int_list = np.zeros((7),dtype=float)
    Xranklist = np.zeros((7),dtype=int)
    for j in tqdm(range(1,8),leave=False, desc='Saving Coeffs for OLS...'):
        n_trains = np.asarray([2**(j+5),2**(j+4),2**(j+3),2**(j+2),2**(j+1),2**j])[6-nfids:]
        ###TRAINING######
        model = MFML(reg=reg, kernel=kernel, sigma=sigma, p_bar=False)
        model.train(X_train_parent=X_train, 
                    y_trains=y_trains, indexes=indexes, 
                    shuffle=True, n_trains=n_trains, seed=0)
        ##########OLS##########
        predsOLS = model.predict(X_test = X_test, y_test = y_test, 
                                 X_val = X_val, y_val = y_val, 
                                 optimiser='OLS', copy_X= True, 
                                 fit_intercept= True)
        
        coeff_list[j-1,:] = np.copy(model.LCCoptimizer.coef_)
        int_list[j-1] = np.copy(model.LCCoptimizer.intercept_)
        Xranklist[j-1] = np.copy(model.LCCoptimizer.rank_)
    
    return coeff_list, int_list, Xranklist


def save_preds(y_trains, indexes, X_train, X_test, X_val, y_test, y_val):
    
    model = MFML(reg=reg, kernel=kernel, sigma=sigma, p_bar=False)
    j=7
    n_trains = np.asarray([2**(j+5),2**(j+4),2**(j+3),2**(j+2),2**(j+1),2**j])
    model.train(X_train_parent=X_train, 
                y_trains=y_trains, indexes=indexes, 
                shuffle=True, n_trains=n_trains, seed=0)
    preds_default = model.predict(X_test = X_test, y_test = y_test, 
                                  X_val = X_val, y_val = y_val, 
                                  optimiser='default')
    preds_OLS = model.predict(X_test = X_test, y_test = y_test, 
                              X_val = X_val, y_val = y_val, 
                              optimiser='OLS', copy_X= True, 
                              fit_intercept= False)
    
    
    return preds_default, preds_OLS
    

def main():
    for fb in tqdm(range(5),desc='Baseline loop...'):
        try:
            y_trains = np.load(f'Data/energies/y_trains_{str(fb)}.npy',allow_pickle=True)
            indexes = np.load(f'Data/energies/indexes_{str(fb)}.npy',allow_pickle=True)
        except:
            y_trains, indexes = MFML.property_differences(fidelities=fidelity_list[fb:])
            np.save(f'Data/energies/y_trains_{str(fb)}.npy',y_trains,allow_pickle=True)
            np.save(f'Data/energies/indexes_{str(fb)}.npy',indexes,allow_pickle=True)
        
        #center all data
        #for i in range(y_trains.shape[0]):
        #    y_trains[i] = y_trains[i] - np.mean(y_trains[i])
        #y_val = y_val - np.mean(y_trains[-1]) #center validation with train mean 
        #y_test = y_test - np.mean(y_trains[-1]) #center test with train mean
        
        maeols,maedef = LC_routine(y_trains=y_trains, indexes=indexes, 
                                   X_train=X_train, X_test=X_test, 
                                   X_val=X_val, y_test=y_test, y_val=y_val, 
                                   navg=10)
        np.save(f'Data/outputs/OLS_{str(fb)}.npy',maeols)
        np.save(f'Data/outputs/def_{str(fb)}.npy',maedef)
        #np.save(f'Data/ModelOuts/globalSLATM/LASSO_{str(fb)}.npy',maelrr)
        #np.save(f'Data/ModelOuts/globalSLATM/intercept/def_{str(fb)}.npy',maedef)
        
        #coeffs, intercepts, Xrank = save_coeffs(y_trains=y_trains, indexes=indexes, 
        #                     X_train=X_train, X_test=X_test, 
        #                     X_val=X_val, y_test=y_test, y_val=y_val)
        #np.save(f'Data/ModelOuts/globalSLATM/intercept/coeffOLS_{fb}.npy',coeffs)
        #np.save(f'Data/ModelOuts/globalSLATM/intercept/intOLS_{fb}.npy',intercepts)
        #np.save(f'Data/ModelOuts/globalSLATM/intercept/XrankOLS_{fb}.npy',Xrank)
        
        #defpreds, olspreds = save_preds(y_trains=y_trains, indexes=indexes, 
        #                                X_train=X_train, X_test=X_test, 
        #                                X_val=X_val, y_test=y_test, y_val=y_val)
        #np.save('Data/ModelOuts/globalSLATM/preds_default.npy',defpreds)
        #np.save('Data/ModelOuts/globalSLATM/preds_OLS.npy',olspreds)
        

if __name__=='__main__':
    X_train = np.load('Data/GlobalSLATMTrain.npy',allow_pickle=True)
    #1067 total - 367 val, 700 test 
    y_val = np.loadtxt('Evaluation/eval_CCSD-ccpvdz.dat')[0:367,1]
    y_test = np.loadtxt('Evaluation/eval_CCSD-ccpvdz.dat')[367:,1]
    
    X_val = np.load('Evaluation/GlobalSLATMEval.npy',allow_pickle=True)[0:367]
    X_test = np.load('Evaluation/GlobalSLATMEval.npy',allow_pickle=True)[367:]
    
    print(f'First Sanity Checks: \nX_train size: {X_train.shape}\nX_val size: {X_val.shape}\nX_test size: {X_test.shape}')
    
    reg = 1e-10
    sigma=400.0
    kernel = 'laplacian'
    
    #fidelity_list = np.asarray(['MP2-STO3G.dat','MP2-631G.dat','MP2-ccpvdz.dat',
    #                            'CCSD-STO3G.dat','CCSD-631G.dat','CCSD-ccpvdz.dat'])
    main()