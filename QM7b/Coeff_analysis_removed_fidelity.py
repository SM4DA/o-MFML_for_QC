import numpy as np
from Model_MFML import ModelMFML as MFML

def main():
    
    y_trains_full = np.load('Data/energies/y_trains_0.npy',allow_pickle=True)
    indexes_full = np.load('Data/energies/indexes_0.npy',allow_pickle=True)
    
    y_trains_removed = np.zeros((5),dtype=object)
    indexes_removed = np.zeros((5),dtype=object)
    
    #remove the CCSD-631G fidelity - call it removed model
    for i in [0,1,2,3,5]:
        if i<5:
            y_trains_removed[i] = np.copy(y_trains_full[i])
            indexes_removed[i] = np.copy(indexes_full[i])
        else:
            y_trains_removed[i-1] = np.copy(y_trains_full[i])
            indexes_removed[i-1] = np.copy(indexes_full[i])
        
    
    model_full = MFML(reg=1e-10, kernel='laplacian', sigma=400.0, p_bar=False)
    model_removed = MFML(reg=1e-10, kernel='laplacian', sigma=400.0, p_bar=False)

    model_full.train(X_train_parent=X_train, 
                     y_trains=y_trains_full, 
                     indexes=indexes_full, 
                     shuffle=True, seed=0, n_trains=np.asarray([2**12, 2**11, 2**10, 2**9, 2**8, 2**7]))
    
    model_removed.train(X_train_parent=X_train, 
                        y_trains=y_trains_removed, 
                        indexes=indexes_removed, 
                        shuffle=True, seed=0, n_trains=np.asarray([2**12, 2**11, 2**10, 2**9, 2**7]))
    
    preds_full = model_full.predict(X_test = X_test, y_test = y_test, 
                               X_val = X_val, y_val = y_val, 
                               optimiser='OLS', copy_X= True, 
                               fit_intercept= False)
    
    preds_removed = model_removed.predict(X_test = X_test, y_test = y_test, 
                                  X_val = X_val, y_val = y_val, 
                                  optimiser='OLS', copy_X= True, 
                                  fit_intercept= False)
    
    print(model_full.mae, model_removed.mae)


if __name__=='__main__':
    X_train = np.load('Data/GlobalSLATMTrain.npy',allow_pickle=True)
    #1067 total - 367 val, 700 test 
    y_val = np.loadtxt('Evaluation/eval_CCSD-ccpvdz.dat')[0:367,1]
    y_test = np.loadtxt('Evaluation/eval_CCSD-ccpvdz.dat')[367:,1]
    
    X_val = np.load('Evaluation/GlobalSLATMEval.npy',allow_pickle=True)[0:367]
    X_test = np.load('Evaluation/GlobalSLATMEval.npy',allow_pickle=True)[367:]
    
    main()