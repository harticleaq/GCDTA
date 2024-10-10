import numpy as np
import pandas as pd
from rdkit import Chem


def process_smiles(smiles, max_len=64):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    
    if mol == None:
        return np.eye(max_len)
    
    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    adj_matrix = np.array(adj_matrix)+np.eye(adj_matrix.shape[0])
    pad_width = ((0, max_len - adj_matrix.shape[0]), (0, max_len - adj_matrix.shape[0]))
    adj_matrix = np.pad(adj_matrix, pad_width, mode='constant', constant_values=0)
    return adj_matrix

# process_smiles("c1cc(OC)ccc1S(=O)(=O)N")
phases = ['training', 'validation', 'test']
data_path = "../data"
for phase in phases:
    ligands_df = pd.read_csv(data_path + f"/{phase}_smi.csv")
    smi = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
    adj = {}
    for key, value in smi.items():
        adj[key] = process_smiles(value)
    np.save(data_path+f"{phase}_adj", smi)


