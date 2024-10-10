from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from rdkit import Chem

def process_smiles(smiles, max_len=150, data_type="default"):
    try:
        if data_type == "default":
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
        elif data_type == "GPCR":
            mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")

    if mol == None:
        return np.eye(max_len)
    
    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    adj_matrix = np.array(adj_matrix)+np.eye(adj_matrix.shape[0])
    pad_len = max_len - adj_matrix.shape[0]
    if pad_len > 0:
        pad_width = ((0, pad_len), (0, pad_len))
        adj_matrix = np.pad(adj_matrix, pad_width, mode='constant', constant_values=0)
    else:
        adj_matrix = adj_matrix[:max_len, :max_len]
    return adj_matrix

def _n2t(arr, device="cuda:0"):
    if isinstance(arr, np.ndarray):
        return torch.tensor(arr).to(device=device, dtype=torch.float32)
    else:
        return arr.to(device)

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

CHAR_SEQ_SET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

def label_seq(line, max_seq_len=1000):
    X = np.zeros(max_seq_len, dtype=np.int)
    for i, ch in enumerate(line[:max_seq_len]):
        X[i] = CHAR_SEQ_SET[ch] - 1
    return X

def label_smiles(line, max_smi_len=150):
    X = np.zeros(max_smi_len, dtype=np.int)
    for i, ch in enumerate(line[:max_smi_len]):
        X[i] = CHAR_SMI_SET[ch] - 1
    return X

class MyDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_smi_len, seq_size):
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len
        self.seq_size = seq_size

        data_path = Path(data_path)
        
        affinity = {}
        affinity_df = pd.read_csv(data_path / 'affinity_data.csv')
        for _, row in affinity_df.iterrows():
            affinity[row[0]] = row[1]
        self.affinity = affinity

        ligands_df = pd.read_csv(data_path / f"{phase}_smi.csv")
        ligands = {i["pdbid"]: i["smiles"] for _, i in ligands_df.iterrows()}
        self.smi = ligands

        seq_df = pd.read_csv(data_path / f"{phase}_seq_.csv")
        seq = {i["id"]: i["seq"] for _, i in seq_df.iterrows()}
        self.seq = seq
        self.seq_path = list(self.seq.keys())
        assert len(self.seq) == len(self.smi)

        # seq_path = data_path / phase / 'global'
        # self.seq_path = sorted(list(seq_path.glob('*')))
        # assert len(self.seq_path) == len(self.smi)
        self.length = len(self.smi)

    def __getitem__(self, idx):
        seq_name = self.seq_path[idx]
        _seq_tensor = self.seq[seq_name][:self.max_seq_len]
        # seq_tensor = np.array(_seq_tensor)
        seq_tensor = label_seq(_seq_tensor, self.max_seq_len)
        # seq_tensor = np.zeros((self.max_seq_len, self.seq_size))
        # seq_tensor[:len(_seq_tensor)] = _seq_tensor
        smile = self.smi[seq_name]
        adj = process_smiles(smile, self.max_smi_len)
        smile = label_smiles(smile, self.max_smi_len)

        return (
            smile,
            seq_tensor,
            np.array(self.affinity[seq_name], dtype=np.float32),
            adj
            )
        # seq = self.seq_path[idx]
        # _seq_tensor = pd.read_csv(seq, index_col=0).drop(['idx'], axis=1).values[:self.max_seq_len]
        # seq_tensor = np.zeros((self.max_seq_len, self.seq_size))
        # seq_tensor[:len(_seq_tensor)] = _seq_tensor
        # smile = self.smi[seq.name.split('.')[0]]

        # adj = process_smiles(smile, self.max_smi_len)
        # smile = label_smiles(smile, self.max_smi_len)
        
        # return (
        #     smile,
        #     seq_tensor.astype(np.float32),
        #     np.array(self.affinity[seq.name.split('.')[0]], dtype=np.float32),
        #     adj
        #     )

    def __len__(self):
        return self.length
    

class SpecialDataset(Dataset):
    def __init__(self, data_path, phase, max_seq_len, max_smi_len, seq_size, data_type="GPCR"):
        self.max_seq_len = max_seq_len
        self.max_smi_len = max_smi_len
        self.seq_size = seq_size
        data_path = Path(data_path)

        with open(data_path / f"{data_type}_{phase}.txt","r") as f:
            data_list = f.read().strip().split('\n')
        """Exclude data contains '.' in the SMILES format."""
        data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
        N = len(data_list)
        self.smi, self.seq, self.affinity = [], [], []
        for num, data in enumerate(data_list):
            smi, seq, affinity = data.strip().split(" ")
            self.smi.append(smi)
            self.seq.append(seq)
            self.affinity.append(affinity)
        
        self.length = N

    def __getitem__(self, idx):
        seq_tensor = self.seq[idx]
        seq_tensor = label_seq(seq_tensor, self.max_seq_len)
        seq_tensor = seq_tensor
        # seq_tensor = np.eye(self.seq_size)[seq_tensor].astype(np.float32)
        smile = self.smi[idx]
        adj = process_smiles(smile, self.max_smi_len)
        smile = label_smiles(smile, self.max_smi_len)
        affinity = np.array(float(self.affinity[idx])).astype(np.float32)
        
        return (
            smile,
            seq_tensor,
            affinity,
            adj
            )

    def __len__(self):
        return self.length