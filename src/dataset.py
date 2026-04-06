import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from torch.utils.data import DataLoader, TensorDataset
import neurokit2 as nk

# Load Data
"""
Adapted from example_physionet.py in:
@article{PhysioNet-ptb-xl-1.0.3,
  author = {Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Samek, Wojciech and Schaeffter, Tobias},
  title = {{PTB-XL, a large publicly available electrocardiography dataset}},
  journal = {{PhysioNet}},
  year = {2022},
  month = nov,
  note = {Version 1.0.3},
  doi = {10.13026/kfzx-aw45},
  url = {https://doi.org/10.13026/kfzx-aw45}
}
"""

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]

    data = np.array([signal for signal, meta in data])
    return data

def aggregate_diagnostic(agg_df, y_dic):
   # for each entry take each diagnostic statement and add to entry in temporary lsit which is returned
    tmp = []
    sorted_y_dic = {k:v for k,v in sorted(y_dic.items(), key=lambda item: item[1], reverse=True)}
    for key in sorted_y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(dict.fromkeys(tmp))

def load_data(path, sampling_rate=100):
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0) # takes all scp statements in the file
    agg_df = agg_df[agg_df.diagnostic == 1] # filters them to only include diagnostic statements

    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(agg_df, x))

    return X, Y

def split_folds(X, Y, test_fold=10, valid_fold=9):
    X_train = X[np.where(Y.strat_fold <valid_fold)]
    y_train = Y[(Y.strat_fold <valid_fold)].diagnostic_superclass

    # Validation
    X_valid = X[np.where(Y.strat_fold == valid_fold)]
    y_valid = Y[(Y.strat_fold == valid_fold)].diagnostic_superclass

    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass

    return {'train':[X_train, y_train], 
            'valid':[X_valid, y_valid],
            'test':[X_test, y_test]}

# Create Dataset

def create_dataset(data, labels, clean, split):

  """
  Data formatted as [R, S, L]
  R: record
  S: number of samples or sample index / features
  L: lead

  Each record has 1000 samples (as its 100hz recording over 10 seconds so 100*10)
  Each sample in the record has data for each lead.

  """

  inv_labels_map = {
      0: 'NORM',
      1: 'HYP',
      2: 'MI',
      3: 'CD',
      4: 'STTC',
  }

  labels_map = {v:k for k,v in inv_labels_map.items()}

  final_labels = []
  final_data = []
  removed = []



  for i, item in enumerate(labels.values):
    if item:

      twelve_lead1 = []
      twelve_lead2 = []

      for lead in range(12):

        temp_data = data[i,:,lead]


        if clean:
          temp_data = nk.ecg_clean(data[i,:,lead],100)


        if split:
          twelve_lead1.append(temp_data[:500])
          twelve_lead2.append(temp_data[500:])

        else:
          twelve_lead1.append(temp_data)



      final_labels.append(labels_map[item[0]])
      final_data.append(twelve_lead1)

      if split:
        final_labels.append(labels_map[item[0]])
        final_data.append(twelve_lead2)



    else:
      removed.append(i)

  return removed, TensorDataset(torch.Tensor(np.array(final_data)), torch.LongTensor(final_labels))

def get_weighting(dataset):
  _, counts = np.unique(dataset, return_counts=True)

  total = sum(counts)

  weighting = torch.Tensor([i/total for i in np.unique(dataset, return_counts=True)[1]])
  return weighting

