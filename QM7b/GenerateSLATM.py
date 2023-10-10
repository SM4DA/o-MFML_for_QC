import numpy as np
import qml
from qml.representations import get_slatm_mbtypes
from tqdm import tqdm



    
def slatm_glob():
    print('Global SLATM Routine\n')
    compounds = []
    for i in tqdm(range(7211),desc='loading compounds...'):
        comps = qml.Compound(xyz=f"geometry/frag_{str(1+i).zfill(4)}.xyz")
        compounds.append(comps)
    
    mbtypes = get_slatm_mbtypes(np.array([mol.nuclear_charges for mol in tqdm(compounds, desc='get mbtype...')]))
    
    for i, mol in tqdm(enumerate(compounds),desc='Generate Local SLATM...'):
        mol.generate_slatm(mbtypes,local=False)
    
    X_slat_glob = np.asarray([mol.representation for mol in tqdm(compounds,desc='Saving Reps...')])
    
    np.save('GlobalSLATM.npy', X_slat_glob)

if __name__=='__main__':
    slatm_glob()
