import numpy as np
from MFML_OOP import MFML as datadiffMF
from Model_MFML import ModelMFML as modelMF
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def main():
    data_diff_MAE = np.zeros((3,2),dtype=float)
    model_diff_MAE = np.zeros((3,2),dtype=float)
    for m in tqdm(range(3),desc='Loop over molecules', leave=True):
        for t in tqdm(range(2),desc='Loop over trajectories',leave=False):
            X_train = np.load(f'Data/representations/{mol[m]}_{traj[t]}_CM.npy')
            y_trains = np.load(f'Data/energyarrs/{mol[m]}/{mol[m]}_{traj[t]}_0_ytrains.npy',allow_pickle=True)
            indexes = np.load(f'Data/energyarrs/{mol[m]}/{mol[m]}_{traj[t]}_0_indexes.npy',allow_pickle=True)
            differences = np.load(f'Data/energyarrs/{mol[m]}/{mol[m]}_{traj[t]}_0_diffarray.npy',allow_pickle=True)
            Xtemp = np.load(f'Evaluation/EVAL_{mol[m]}_{traj[t]}_EVAL_CM.npy')
            ytemp = np.loadtxt(f'Evaluation/{mol[m]}_{traj[t]}_EVAL_E_def2-tzvp.dat')
            
            X_val,X_test,y_val,y_test = train_test_split(Xtemp, ytemp, 
                                                         random_state=0, 
                                                         train_size=712.0/2712.0)
            
            modeldiff = modelMF(reg=1e-9, kernel='matern', sigma=sigmas[m,t], 
                            order=1, metric='l2', gammas=None, p_bar=False)
            modeldiff.train(X_train_parent=X_train, 
                        y_trains=y_trains, indexes=indexes, 
                        shuffle=False, seed=0)
            pp = modeldiff.predict(X_test = X_test, y_test = y_test, 
                               X_val = X_val, y_val = y_val, 
                               optimiser='default')
            model_diff_MAE[m,t] = modeldiff.mae
            
            
            
            datadiff = datadiffMF(reg=1e-9, kernel='matern', sigma=sigmas[m,t], 
                                order=1, metric='l2', gammas=None)
            datadiff.train(X_train=X_train, fidelities=None, 
                        y_trains=differences, indexes=indexes, 
                        shuffle=False, seed=0)
            ppd = datadiff.predict(X_test = X_test, y_test = y_test)
            data_diff_MAE[m,t] = datadiff.mae
    
    np.savetxt('DataDiffMAE.txt', data_diff_MAE, fmt="%.18f")
    np.savetxt('ModelDiffMAE.txt', model_diff_MAE, fmt="%.18f")


if __name__=='__main__':
    mol = ['benzene','naphthalene','anthracene']
    traj = ['MD','DFTB']
    sigmas = np.asarray([[715,940],[1300,1200],[2455,2200]])
    main()