import numpy as np
import pandas as pd
import sklearn
import jax
import scipy
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import re
import itertools
import os, sys
sys.path.append("./../../")
from latent_shift_adaptation.data.lsa_synthetic import Simulator, MultiWSimulator



folder_id = '/nfs/gatsbystor/williamw/latent_confounder/shared_data_linear_3'
os.makedirs(folder_id, exist_ok=True)



#@title Library functions

def get_squeezed_df(data_dict: dict) -> pd.DataFrame:
  """Converts a dict of numpy arrays into a DataFrame, extracting columns of arrays into separate DataFrame columns."""
  temp = {}
  for key, value in data_dict.items():
    squeezed_array = np.squeeze(value)
    if len(squeezed_array.shape) == 1:
      temp[key] = squeezed_array
    elif len(squeezed_array.shape) > 1:
      for i in range(value.shape[1]):
        temp[f'{key}_{i}'] = np.squeeze(value[:, i])
  return pd.DataFrame(temp)

def process_data(data_dict, w_cols=['w_1', 'w_2', 'w_3']):
  result = data_dict.copy()
  for w_col in w_cols:
    result[f'{w_col}_binary'] = 1.0*(result[f'{w_col}'] > 0)
    result[f'{w_col}_one_hot'] = OneHotEncoder(sparse=False).fit_transform(result[f'{w_col}_binary'])
  result['u_one_hot'] = OneHotEncoder(sparse=False).fit_transform(result['u'].reshape(-1, 1))
  return result

def generate_data(p_u, seed, num_samples, partition_dict, param_dict=None):

  sim = MultiWSimulator(param_dict=param_dict)
  samples_dict = {}
  for i, (partition_key, partition_frac) in enumerate(partition_dict.items()):
    num_samples_partition = int(partition_frac*num_samples)
    sim.update_param_dict(num_samples=num_samples_partition, p_u=p_u)
    samples_dict[partition_key] = process_data(sim.get_samples(seed=seed + 15*i))
  return samples_dict

def tidy_w(data_dict, w_value):
  result = data_dict.copy()
  for key in result.keys():
    result[key]['w'] = result[key][f'w_{w_value}']
    result[key]['w_binary'] = result[key][f'w_{w_value}_binary']
    result[key]['w_one_hot'] = result[key][f'w_{w_value}_one_hot']
  return result


# Convert data to dataframe format
def pack_to_df(samples_dict):
  return pd.concat({key: get_squeezed_df(value) for key, value in samples_dict.items()}).reset_index(level=-1, drop=True).rename_axis('partition').reset_index()

# Extract dataframe format back to dict format
def extract_from_df(samples_df, cols=['u', 'x', 'w', 'c', 'c_logits', 'y', 'y_logits', 'y_one_hot',
                                      'u_one_hot', 'x_scaled',
                                      'w_1', 'w_1_binary', 'w_1_one_hot',
                                      'w_2', 'w_2_binary', 'w_2_one_hot',
                                      'w_2_binary', 'w_2_one_hot',
                                      ]):
  """
  Extracts dict of numpy arrays from dataframe
  """
  result = {}
  for col in cols:
    if col in samples_df.columns:
      result[col] = samples_df[col].values
    else:
      match_str = f"^{col}_\d$"
      r = re.compile(match_str, re.IGNORECASE)
      matching_columns = list(filter(r.match, samples_df.columns))
      if len(matching_columns) == 0:
        continue
      result[col] = samples_df[matching_columns].to_numpy()
  return result

def extract_from_df_nested(samples_df, cols=['u', 'x', 'w', 'c', 'c_logits', 'y', 'y_logits', 'y_one_hot', 'w_binary', 'w_one_hot', 'u_one_hot', 'x_scaled']):
  """
  Extracts nested dict of numpy arrays from dataframe with structure {domain: {partition: data}}
  """
  result = {}
  for domain in samples_df['domain'].unique():
    result[domain] = {}
    domain_df = samples_df.query('domain == @domain')
    for partition in domain_df['partition'].unique():
      partition_df = domain_df.query('partition == @partition')
      result[domain][partition] = extract_from_df(partition_df, cols=cols)
  return result

def write_df_to_drive(filename, data, folder_id, overwrite=True):
  file_id = drive.SaveFile(filename=filename, data=data.to_csv(index=False), overwrite_warning=1-overwrite)
  existing_files = drive.ListFolderWithFileNames(folder_id=folder_id)
  existing_files = {key: value for key, value in existing_files.items() if value == filename}
  if filename in existing_files.values():
    if not overwrite:
      print('File exists in folder. Doing nothing')
    else:
      print('Overwriting file in folder')
      for trash_file_id in existing_files.keys():
        drive.TrashFile(trash_file_id)
      drive.MoveFileToFolder(file_id=file_id, folder_id=folder_id)
  else:
    drive.MoveFileToFolder(file_id=file_id, folder_id=folder_id)



def expand_mu_x_u(u_dim, x_dim):
    mu_x_u = np.zeros((u_dim, x_dim))
    mu_x_u[0, 0] = -1
    mu_x_u[0, 1] = 1
    mu_x_u[1, 0] = 1
    mu_x_u[1, 1] = -1
    return mu_x_u * 0.3

def expand_mu_c_x(u_dim, c_dim, x_dim):
    mu_c_x = np.zeros((u_dim, x_dim, c_dim))
    mu_c_x[0, 0, 0] = -2
    mu_c_x[0, 1, 0] = 2
    mu_c_x[1, 0, 0] = 2
    mu_c_x[1, 1, 0] = -2
    return mu_c_x


param_dict = {
    'num_samples': 10000,
    'k_w': 1,
    'k_x': 2,
    'k_c': 1,
    'k_y': 1,
    'mu_w_u_coeff_list': [1,2,3],
    'mu_x_u_coeff': 1,
    'mu_y_u_coeff': 2,
    'mu_y_c_coeff': 1,
    'mu_c_u_coeff': 1,
    'mu_c_x_coeff': 1,
    'mu_w_u_mat': np.array([[-1, 1]]).T,
    # 'mu_x_u_mat': np.array([[-1, 1], [1, -1]]),  # k_u x k_x
    # 'mu_c_u_mat': np.array([[-2, 2, 2], [-1, 1, 2]]),  # k_u x k_c
    'mu_c_u_mat': np.array([[-2], [-1]]),  # k_u x k_c
    # 'mu_c_x_mat': np.array(
    #     [[[-2, 2, -1], [1, -2, -3]], [[2, -2, 1], [-1, 2, 3]]]
    # ),  # k_u x k_x x k_c
    'mu_y_c_mat': np.array([[-2], [-1]]),  # k_u x k_c
    # 'mu_y_c_mat': np.array([[3, -2, -1], [3, -1, -2]]),  # k_u x k_c
    'mu_y_u_mat': np.array([[1, 2]]).T,  # k_u x 1
    'sd_c': 0.0,
    'sd_y': 0.0,
    'p_u': [0.5, 0.5],
}

x_dims = np.arange(2,42,2)
w_coeff_list = [1,3]
p_u_range = [0.9, 0.1]
num_samples = 100000


if folder_id is not None:
    for x_dim in x_dims:
        param_dict['k_x'] = x_dim
        param_dict['mu_x_u_mat'] = expand_mu_x_u(2,x_dim)
        param_dict['mu_c_x_mat'] = expand_mu_c_x(2,param_dict['k_c'],x_dim)
        print('cur x dim',x_dim,param_dict['mu_x_u_mat'].shape)
        partition_dict = {'train': 0.7, 'val': 0.1, 'test': 0.2}
        seed = 192
        result = {}
        for i, p_u_0 in enumerate(p_u_range):
            p_u = [p_u_0, 1 - p_u_0]
            param_dict['p_u'] = p_u
            samples_dict = generate_data(p_u=p_u, seed=i * seed + i, num_samples=num_samples, partition_dict=partition_dict, param_dict=param_dict)
            for w_coeff in w_coeff_list:
                print(p_u_0, w_coeff)
                samples_dict_tidy = tidy_w(samples_dict, w_value=w_coeff)
                samples_df = pack_to_df(samples_dict_tidy)
                filename = f'{x_dim}_synthetic_multivariate_num_samples_10000_w_coeff_{w_coeff}_p_u_0_{p_u_0}.csv'
                samples_df.to_csv(os.path.join(folder_id, filename), index=False)
                print('output folder ',os.path.join(folder_id, filename))

else:
    print('folder_id not set. Doing nothing')