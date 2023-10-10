import numpy as np
from tqdm import tqdm
from Model_MFML import ModelMFML as MFML
from sklearn.model_selection import train_test_split

def MAE_LearningCurve(X_train,y_trains,indexes,X_val,y_val,X_test,y_test,sigma,navg=10):
    
    nfids = y_trains.shape[0]
    
    MAEs_OLS = np.zeros((9),dtype=float) #for OLS MFML
    MAEs_LRR = np.zeros((9),dtype=float) #for LRR MFML
    MAEs_def = np.zeros((9),dtype=float) # for default MFML
    
    for i in tqdm(range(navg), leave=False, desc='Averaged loop'):
        mae_ntr_OLS = []
        mae_ntr_LRR = []
        mae_ntr_def = []
        
        for j in tqdm(range(1,10),leave=False, desc='Loop over training sizes'):
            n_trains = np.asarray([2**(j+4),2**(j+3),2**(j+2),2**(j+1),2**j])[5-nfids:]
            ###TRAINING######
            model = MFML(reg=1e-9, kernel='matern', sigma=sigma, order=1, 
                             metric='l2', gammas=None, p_bar=False)
            model.train(X_train_parent=X_train, 
                        y_trains=y_trains, indexes=indexes, 
                        shuffle=True, n_trains=n_trains, seed=0)
            ##########OLS##########
            predsOLS = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='OLS', copy_X= True, 
                                     fit_intercept= False)
            mae_ntr_OLS.append(model.mae)
            
            
            ##########LRR#########
            predsLRR = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='LRR', alpha=1e-6, 
                                     fit_intercept=False, copy_X=True, 
                                     max_iter=None, tol=1e-4, solver='lsqr', 
                                     random_state=0)
            mae_ntr_LRR.append(model.mae)
            
            
            ######default#########
            predsdef = model.predict(X_test = X_test, y_test = y_test, 
                                     X_val = X_val, y_val = y_val, 
                                     optimiser='default')
            mae_ntr_def.append(model.mae)
    
        #store each avg run MAE  
        mae_ntr_OLS = np.asarray(mae_ntr_OLS)
        mae_ntr_LRR = np.asarray(mae_ntr_LRR)
        mae_ntr_def = np.asarray(mae_ntr_def)
        
        MAEs_OLS += mae_ntr_OLS
        MAEs_LRR += mae_ntr_LRR
        MAEs_def += mae_ntr_def
        
       
    
    #return averaged MAE
    MAEs_OLS = MAEs_OLS/navg
    MAEs_LRR = MAEs_LRR/navg
    MAEs_def = MAEs_def/navg
    return MAEs_OLS, MAEs_LRR, MAEs_def
        

def coeff_intercept_saver(X_train,y_trains,indexes,X_val,y_val,X_test,y_test,sigma):
    nfids = y_trains.shape[0]
    
    OLS_intercepts = np.zeros((9),dtype=object)
    OLS_coeffs = np.zeros((9),dtype=object)
    LRR_intercepts = np.zeros((9),dtype=object)
    LRR_coeffs = np.zeros((9),dtype=object)
    
    #Loop over training sizes
    for j in tqdm(range(1,10),leave=False, desc='Loop over training sizes'):
        n_trains = np.asarray([2**(j+4),2**(j+3),2**(j+2),2**(j+1),2**j])[5-nfids:]

        ###TRAINING######
        model = MFML(reg=1e-9, kernel='matern', sigma=sigma, order=1, 
                         metric='l2', gammas=None, p_bar=False)
        model.train(X_train_parent=X_train, 
                    y_trains=y_trains, indexes=indexes, 
                    shuffle=True, n_trains=n_trains, seed=0)
        #OLS
        predsOLS = model.predict(X_test = X_test, y_test = y_test, 
                                 X_val = X_val, y_val = y_val, 
                                 optimiser = 'OLS', copy_X = True, 
                                 fit_intercept = False)
        OLS_coeffs[j-1] = np.copy(model.LCCoptimizer.coef_)
        OLS_intercepts[j-1] = np.copy(model.LCCoptimizer.intercept_)

        #LRR
        predsLRR = model.predict(X_test = X_test, y_test = y_test, 
                                 X_val = X_val, y_val = y_val, 
                                 optimiser = 'LRR', alpha = 1e-6, 
                                 fit_intercept=False, copy_X = True, 
                                 max_iter = None, tol = 1e-4, solver = 'lsqr', 
                                 random_state = 0)
        LRR_coeffs[j-1] = np.copy(model.LCCoptimizer.coef_)
        LRR_intercepts[j-1] = np.copy(model.LCCoptimizer.intercept_)
    
    return OLS_coeffs, OLS_intercepts, LRR_coeffs, LRR_intercepts


def main():
    for m in tqdm(range(3),desc='Loop over molecules', leave=True):
        for t in tqdm(range(2),desc='Loop over trajectories',leave=False):
            X_train = np.load(f'Data/representations/{mol[m]}_{traj[t]}_CM.npy')
            molname = np.full(5,mol[m]+'_'+traj[t]+'_E_')
            #print(molname)
            
            Xtemp = np.load(f'Evaluation/EVAL_{mol[m]}_{traj[t]}_EVAL_CM.npy')
            ytemp = np.loadtxt(f'Evaluation/{mol[m]}_{traj[t]}_EVAL_E_def2-tzvp.dat')
            
            X_val,X_test,y_val,y_test = train_test_split(Xtemp, ytemp, 
                                                         random_state=0, 
                                                         train_size=712.0/2712.0)
            
            for b in tqdm(range(4),desc='Loop over baselines',leave=False):    
                fidelity_list = np.asarray([molname[i]+fidnames[i] for i in range(b,5)])
                
                y_trains = np.load(f'Data/energyarrs/{mol[m]}/{mol[m]}_{traj[t]}_{b}_ytrains.npy',
                                   allow_pickle=True)
                indexes = np.load(f'Data/energyarrs/{mol[m]}/{mol[m]}_{traj[t]}_{b}_indexes.npy',
                                  allow_pickle=True)
                
                
                #generate the ytrains and indexes 
                #y_trains, indexes = MFML.property_differences(fidelities=fidelity_list)
                
                #np.save(f'Data/energyarrs/{mol[m]}/{mol[m]}_{traj[t]}_{b}_ytrains.npy',y_trains)
                #np.save(f'Data/energyarrs/{mol[m]}/{mol[m]}_{traj[t]}_{b}_indexes.npy',indexes)
                
                #save coeffs and intercepts of LCC optimizer
                OLS_co, OLS_int, LRR_co, LRR_int = coeff_intercept_saver(X_train[0::strides[b]],
                                                                         y_trains, indexes, 
                                                                         X_val, y_val, X_test, y_test,
                                                                         sigma=sigmas[m,t])
                np.save(f'Data/outputsnew/OLS/{mol[m]}_{traj[t]}_{b}_co_OLS.npy', OLS_co, allow_pickle=True)
                #np.save(f'Data/outputsnew/OLS/{mol[m]}_{traj[t]}_{b}_int_OLS.npy', OLS_int, allow_pickle=True)
                np.save(f'Data/outputsnew/LRR/{mol[m]}_{traj[t]}_{b}_co_LRR.npy', LRR_co, allow_pickle=True)
                #np.save(f'Data/outputsnew/LRR/{mol[m]}_{traj[t]}_{b}_int_LRR.npy', LRR_int, allow_pickle=True)
                
                
                #generate learning curve MAEs for OLS, LRR, and the default SGCT LCCs
                #OLS_MAE, LRR_MAE, def_MAE = MAE_LearningCurve(X_train[0::strides[b]],
                #                                              y_trains, indexes, 
                #                                              X_val, y_val, X_test, y_test,
                #                                              sigma=sigmas[m,t])
                #save for each baseline, trajectory, and molecule
                #np.save(f'Data/outputsnew/OLS/{mol[m]}_{traj[t]}_{b}_OLSMAE.npy', OLS_MAE, allow_pickle=True)
                #np.save(f'Data/outputsnew/LRR/{mol[m]}_{traj[t]}_{b}_LRRMAE.npy', LRR_MAE, allow_pickle=True)
                #np.save(f'Data/outputsnew/default/{mol[m]}_{traj[t]}_{b}_defMAE.npy', def_MAE, allow_pickle=True)
                
            

if __name__=='__main__':
    
    mol = ['benzene','naphthalene','anthracene']
    traj = ['MD','DFTB']
    sigmas = np.asarray([[715,940],[1300,1200],[2455,2200]])
    fidnames= np.asarray(['sto-3g.dat','3-21g.dat','6-31g.dat','def2-svp.dat','def2-tzvp.dat'])
    
    strides = np.asarray([1,2,4,8]) #for each baseline fidelity
    main()
