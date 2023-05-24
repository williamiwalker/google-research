import tensorflow as tf
import numpy as np
import ml_collections as mlc
import scipy
import os, sys,json
sys.path.append("./../../")
import pandas as pd
import sklearn
import re
import io
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import latent_shift_adaptation.methods.algorithms_sknp as algorithms_sknp
from latent_shift_adaptation.methods.algorithms_sknp import get_classifier
from latent_shift_adaptation.utils import gumbelmax_vae_ci, gumbelmax_vae
from latent_shift_adaptation.methods import baseline, erm
from latent_shift_adaptation.methods.vae import gumbelmax_vanilla, gumbelmax_graph
from latent_shift_adaptation.methods.shift_correction import cov, label, bbse, bbsez
# from IPython.display import display




ITERATIONS = 10 # Set to 10 to replicate experiments in paper
EPOCHS = 500 # Set to 200 to replicate experiments in paper
xlabel = 'x'  # or 'x', 'x_scaled'
SEED = 0

###############################################################################
# get arguments
###############################################################################

SLURM_ARRAY_TASK_ID = sys.argv[1]
print('SLURM_ARRAY_TASK_ID ', SLURM_ARRAY_TASK_ID)

folder_id = '/nfs/gatsbystor/williamw/latent_confounder/shared_data_linear_4'
parent_folder = '/nfs/gatsbystor/williamw/latent_confounder/'
# # parent_folder = '/home/william/mnt/gatsbystor/latent_confounder/'


OUTPUT_FOLDER = 'latent_adapt_4/x_dim_' + str(SLURM_ARRAY_TASK_ID)
saveFolder = parent_folder + OUTPUT_FOLDER + '/'


filename_source = str(SLURM_ARRAY_TASK_ID) + "_synthetic_multivariate_num_samples_10000_w_coeff_3_p_u_0_0.9.csv"
filename_target = str(SLURM_ARRAY_TASK_ID) + "_synthetic_multivariate_num_samples_10000_w_coeff_3_p_u_0_0.1.csv"



DEFAULT_LOSS = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def mlp(num_classes, width, input_shape, learning_rate,
        loss=DEFAULT_LOSS, metrics=[]):
  """Multilabel Classification."""
  model_input = tf.keras.Input(shape=input_shape)
  # hidden layer
  if width:
    x = tf.keras.layers.Dense(
        width, use_bias=True, activation='relu'
    )(model_input)
  else:
    x = model_input
  model_outuput = tf.keras.layers.Dense(num_classes,
                                        use_bias=True,
                                        activation="linear")(x)  # get logits
  opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
  model = tf.keras.models.Model(model_input, model_outuput)
  model.build(input_shape)
  model.compile(loss=loss, optimizer=opt, metrics=metrics)

  return model



# Convert data to dataframe format
def pack_to_df(samples_dict):
  return pd.concat({key: get_squeezed_df(value) for key, value in samples_dict.items()}).reset_index(level=-1, drop=True).rename_axis('partition').reset_index()

# Extract dataframe format back to dict format
def extract_from_df(samples_df, cols=['u', 'x', 'w', 'c', 'c_logits', 'y', 'y_logits', 'y_one_hot', 'w_binary', 'w_one_hot', 'u_one_hot', 'x_scaled']):
  """
  Extracts dict of numpy arrays from dataframe
  """
  result = {}
  for col in cols:
    if col in samples_df.columns:
      result[col] = samples_df[col].values
    else:
      match_str = f"^{col}_\d+$"
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
  if 'domain' in samples_df.keys():
    for domain in samples_df['domain'].unique():
      result[domain] = {}
      domain_df = samples_df.query('domain == @domain')
      for partition in domain_df['partition'].unique():
        partition_df = domain_df.query('partition == @partition')
        result[domain][partition] = extract_from_df(partition_df, cols=cols)
  else:
    for partition in samples_df['partition'].unique():
        partition_df = samples_df.query('partition == @partition')
        result[partition] = extract_from_df(partition_df, cols=cols)
  return result



data_df_source = pd.read_csv(os.path.join(folder_id, filename_source))
data_df_target = pd.read_csv(os.path.join(folder_id, filename_target))
data_dict_source = extract_from_df_nested(data_df_source)
data_dict_target = extract_from_df_nested(data_df_target)
data_dict_all = dict(source=data_dict_source, target=data_dict_target)


# Convert to TF dataset
ds_source = {
    key: tf.data.Dataset.from_tensor_slices(
        (value['x'], value['y_one_hot'], np.expand_dims(value['c'],axis=1), value['w_one_hot'], value['u_one_hot']),
    ) for key, value in data_dict_all['source'].items()
}
ds_target = {
    key: tf.data.Dataset.from_tensor_slices(
        (value['x'], value['y_one_hot'], np.expand_dims(value['c'],axis=1), value['w_one_hot'], value['u_one_hot']),
    ) for key, value in data_dict_all['target'].items()
}

#@title Data exploration
batcher = iter(ds_source['train'])
batch = next(batcher)
for ind in range(10):
    print('batch 2',next(batcher))


batch_size = 128
for split in ds_source.keys():
  ds_source[split] = ds_source[split].repeat().shuffle(1000).batch(batch_size)
  ds_target[split] = ds_target[split].repeat().shuffle(1000).batch(batch_size)

batch = next(iter(ds_source['train']))
print('batch shape',batch[0].shape,batch[2].shape,batch[3].shape,batch[4].shape)
x_dim = batch[0].shape[1]
c_dim = batch[2].shape[1]
w_dim = batch[3].shape[1]
u_dim = batch[4].shape[1]
num_classes = 2
test_fract = 0.2
val_fract = 0.1

num_examples = 10000
steps_per_epoch = num_examples // batch_size
steps_per_epoch_test = int(steps_per_epoch * test_fract)
steps_per_epoch_val = int(steps_per_epoch * val_fract)
steps_per_epoch_train = steps_per_epoch - steps_per_epoch_test - steps_per_epoch_val

pos = mlc.ConfigDict()
pos.x, pos.y, pos.c, pos.w, pos.u = 0, 1, 2, 3, 4


x = tf.convert_to_tensor(batch[0])
var_x = tf.math.reduce_variance(x).numpy()
h_y = scipy.stats.entropy(np.argmax(batch[1], axis=1))
h_c = scipy.stats.entropy(batch[2].numpy().reshape((-1,)))
h_w = scipy.stats.entropy(np.argmax(batch[3], axis=1))
weight_x = 1. / var_x
weight_y = 1. / h_y
weight_c = 1. / h_c
weight_w = 1. / h_w


tf.random.set_seed(SEED)
np.random.seed(SEED)

evals = {  # evaluation functions
    "cross-entropy": tf.keras.metrics.CategoricalCrossentropy(),
    "accuracy": tf.keras.metrics.CategoricalAccuracy(),
    "auc": tf.keras.metrics.AUC(multi_label = False)
}

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', min_delta=0.01, factor=0.1, patience=20,
    min_lr=1e-7)

callbacks = [reduce_lr]

do_calib = True
evaluate = tf.keras.metrics.CategoricalCrossentropy()

learning_rate = 0.01  #@param {type:"number"}
width = 100  #@param {type:"number"}

train_kwargs = {
    'epochs': EPOCHS,
    'steps_per_epoch':steps_per_epoch_train,
    'verbose': True,
    'callbacks':callbacks
    }

val_kwargs = {
    'epochs': EPOCHS,
    'steps_per_epoch':steps_per_epoch_val,
    'verbose': False,
    'callbacks':callbacks
    }

test_kwargs = {'verbose': False,
               'steps': steps_per_epoch_test}


result = {}

#########################################################################################
# Evaluation Metrics
#########################################################################################


# Define sklearn evaluation metrics
def soft_accuracy(y_true, y_pred, threshold=0.5, **kwargs):
    return sklearn.metrics.accuracy_score(y_true, y_pred >= threshold, **kwargs)


def soft_balanced_accuracy(y_true, y_pred, threshold=0.5, **kwargs):
    return sklearn.metrics.balanced_accuracy_score(y_true, y_pred >= threshold, **kwargs)


def log_loss64(y_true, y_pred, **kwargs):
    return sklearn.metrics.log_loss(y_true, y_pred.astype(np.float64), **kwargs)


evals_sklearn = {
    'cross-entropy': log_loss64,
    'accuracy': soft_accuracy,
    'balanced_accuracy': soft_balanced_accuracy,
    'auc': sklearn.metrics.roc_auc_score
}


def evaluate_clf():
    result_dict = {}
    for metric in evals_sklearn.keys():
        result_dict[metric] = {}

    y_pred_source = clf.predict(data_dict_all['source']['test']['x'])
    y_pred_target = clf.predict(data_dict_all['target']['test']['x'])
    if 'cbm' in method:
        # hacky workaround for now
        y_pred_source = y_pred_source[1]
        y_pred_target = y_pred_target[1]
    y_pred_source = y_pred_source.numpy()[:, 1] if tf.is_tensor(y_pred_source) else y_pred_source[:, 1]
    # y_pred_source = y_pred_source.numpy()[:, 1]
    y_pred_target = y_pred_target.numpy()[:, 1] if tf.is_tensor(y_pred_target) else y_pred_target[:, 1]
    y_true_source = data_dict_all['source']['test']['y']
    y_true_target = data_dict_all['target']['test']['y']

    # print('y pred source shape ', y_pred_source.shape, y_pred_target.shape, y_true_source.shape, y_true_target.shape)
    for metric in evals_sklearn.keys():
        result_dict[metric]['eval_on_source'] = evals_sklearn[metric](y_true_source, y_pred_source)
        result_dict[metric]['eval_on_target'] = evals_sklearn[metric](y_true_target, y_pred_target)
    return result_dict


def evaluate_sk(model):
    result_dict = {}
    for metric in evals_sklearn.keys():
        result_dict[metric] = {}
    y_pred_source = model.predict_proba(data_dict_all['source']['test']['x'])[:, -1]
    y_pred_target = model.predict_proba(data_dict_all['target']['test']['x'])[:, -1]
    y_true_source = data_dict_all['source']['test']['y']
    y_true_target = data_dict_all['target']['test']['y']
    for metric in evals_sklearn.keys():
        result_dict[metric]['eval_on_source'] = evals_sklearn[metric](y_true_source, y_pred_source)
        result_dict[metric]['eval_on_target'] = evals_sklearn[metric](y_true_target, y_pred_target)
    return result_dict


#########################################################################################
# Sklearn Baselines
#########################################################################################


result_sk = {}
method = 'erm_source_sk'
result_list = []
for seed in range(ITERATIONS):
  print(f'Iteration: {seed} method:{method}')
  np.random.seed(seed)
  model = get_classifier('mlp')
  model.fit(data_dict_all['source']['train']['x'], data_dict_all['source']['train']['y'])
  result_list.append(evaluate_sk(model))
result_sk[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in enumerate(result_list)])
method = 'erm_target_sk'
result_list = []
for seed in range(ITERATIONS):
  print(f'Iteration: {seed} method:{method}')
  np.random.seed(seed)
  model = get_classifier('mlp')
  model.fit(data_dict_all['target']['train']['x'], data_dict_all['target']['train']['y'])
  result_list.append(evaluate_sk(model))
result_sk[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in enumerate(result_list)])
method = 'known_u_sk'
result_list = []
for seed in range(ITERATIONS):
  print(f'Iteration: {seed} method:{method}')
  np.random.seed(seed)
  lsa_pred_probs_target = algorithms_sknp.latent_shift_adaptation(
      x_source=data_dict_all['source']['train']['x'],
      y_source=data_dict_all['source']['train']['y'],
      u_source=data_dict_all['source']['train']['u'],
      x_target=data_dict_all['target']['test']['x'],
      model_type='mlp')[:, -1]
  lsa_result_dict = {}
  for metric in evals_sklearn.keys():
    lsa_result_dict[metric] = {}
    lsa_result_dict[metric]['eval_on_source'] = np.nan
    lsa_result_dict[metric]['eval_on_target'] = evals_sklearn[metric](data_dict_all['target']['test']['y'], lsa_pred_probs_target)
  result_list.append(lsa_result_dict)
result_sk[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in enumerate(result_list)])


metrics = list(evals_sklearn.keys())
result_df_sk=pd.concat(result_sk).reset_index(level=1, drop=True).rename_axis('method').reset_index()
result_df_sk
results_mean_sk = result_df_sk.groupby(['method', 'eval_set'])[metrics].mean()
results_sd_sk = result_df_sk.groupby(['method', 'eval_set'])[metrics].std()
results_formatted_sk = results_mean_sk.applymap(lambda x: '{:.4f}'.format(x)) + u' \u00B1 ' + results_sd_sk.applymap(lambda x: '{:.4f}'.format(x))
results_formatted_sk
results_long_sk = results_formatted_sk.reset_index().melt(id_vars=['method', 'eval_set'], value_vars=metrics, var_name='metric', value_name='performance')
print(results_long_sk.head())
results_pivot_sk = results_long_sk.pivot(index=['method', 'metric'], columns='eval_set')
print(results_pivot_sk.query('metric == "cross-entropy"'))
print(results_pivot_sk.query('metric == "accuracy"'))
print(results_pivot_sk.query('metric == "auc"'))


#########################################################################################
## ERM - source
# This trains using empirical risk minimization on the source distribution.
#########################################################################################

# Choose your input. If it's only X, set `inputs='x'`. If it's both X and C, set `inputs='xc'`, and so on.

inputs = 'x'  #@param {type:"string"}


input_shape = x_dim * ('x' in inputs) + c_dim * ('c' in inputs) \
  + u_dim * ('u' in inputs) + w_dim * ('w' in inputs)
input_shape = (input_shape, )
model = mlp(num_classes=num_classes, width=width,
            input_shape=input_shape, learning_rate=learning_rate,
            metrics=['accuracy'])
model.summary()

method = 'erm_source'
result_list = []
for seed in range(ITERATIONS):
  print(f'Iteration: {seed} method:{method}')
  tf.random.set_seed(seed)
  np.random.seed(seed)
  model = mlp(num_classes=num_classes, width=width,
            input_shape=input_shape, learning_rate=learning_rate,
            metrics=['accuracy'])
  clf = erm.Method(model, evaluate, inputs=inputs, dtype=tf.float32, pos=pos)
  clf.fit(ds_source['train'], ds_source['val'], ds_target['train'],
          steps_per_epoch_val, **train_kwargs)
  result_list.append(evaluate_clf())
result[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in enumerate(result_list)])



#########################################################################################
## ERM - target
# This trains using empirical risk minimization on the target distribution, assuming access to both x and y in the target domain.
#########################################################################################

inputs = 'x'  #@param {type:"string"}

method = 'erm_target'
result_list = []
for seed in range(ITERATIONS):
  print(f'Iteration: {seed} method:{method}')
  tf.random.set_seed(seed)
  np.random.seed(seed)
  model = mlp(num_classes=num_classes, width=width,
            input_shape=input_shape, learning_rate=learning_rate,
            metrics=['accuracy'])
  clf = erm.Method(model, evaluate, inputs=inputs, dtype=tf.float32, pos=pos)
  clf.fit(ds_target['train'], ds_target['val'], ds_target['train'],
          steps_per_epoch_val, **train_kwargs)
  result_list.append(evaluate_clf())
result[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in enumerate(result_list)])

#########################################################################################
## Covariate Shift
#########################################################################################


# Performs adaptation by weighting instances by $q(x)/p(x)$. We estimate the likelihood ratio using a neural network that discriminates between source and target domains.

method = 'covar'
result_list = []
input_shape = (x_dim,)
for seed in range(ITERATIONS):
  print(f'Iteration: {seed} method:{method}')
  tf.random.set_seed(seed)
  np.random.seed(seed)

  model = mlp(num_classes=num_classes, width=width, input_shape=input_shape,
              learning_rate=learning_rate,
              metrics=['accuracy'])

  domain_discriminator = mlp(num_classes=2, width=width, input_shape=input_shape,
                             learning_rate=learning_rate,
                             loss="sparse_categorical_crossentropy",
                             metrics=['accuracy'])

  clf = cov.Method(model, domain_discriminator, evaluate,
                   dtype=tf.float32, pos=pos)
  clf.fit(ds_source['train'], ds_source['val'], ds_target['train'],
          steps_per_epoch_val, **train_kwargs)
  result_list.append(evaluate_clf())
result[method] = pd.concat(
  [pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in
   enumerate(result_list)])


#########################################################################################
## Label Shift (Oracle Weights)

# Label shift adjustment with oracle access to $q(y)$, where $q(y)$ is the probability distribution of y in the target domain.
#########################################################################################


method = 'label_oracle'
result_list = []
input_shape = (x_dim,)
for seed in range(ITERATIONS):
    print(f'Iteration: {seed} method:{method}')
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = mlp(num_classes=num_classes, width=width, input_shape=input_shape,
                learning_rate=learning_rate,
                metrics=['accuracy'])

    clf = label.Method(model, evaluate, num_classes=num_classes,
                       dtype=tf.float32, pos=pos)
    clf.fit(ds_source['train'], ds_source['val'], ds_target['train'],
            steps_per_epoch_val, **train_kwargs)
    result_list.append(evaluate_clf())
result[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in
                            enumerate(result_list)])


#########################################################################################
## Black box label shift adjustment (BBSE)

# Label shift adjustment in which with the likelihood ratio q(y)/p(y) is estimated using the confusion matrix approach, as in Lipton 2018 (https://arxiv.org/abs/1802.03916).
#########################################################################################


method = 'bbse'

result_list = []
input_shape = (x_dim,)
for seed in range(ITERATIONS):
    print(f'Iteration: {seed} method:{method}')
    tf.random.set_seed(seed)
    np.random.seed(seed)

    x2z = mlp(num_classes=num_classes, width=width, input_shape=input_shape,
              learning_rate=learning_rate,
              metrics=['accuracy'])
    model = mlp(num_classes=num_classes, width=width, input_shape=input_shape,
                learning_rate=learning_rate,
                metrics=['accuracy'])

    clf = bbse.Method(model, x2z, evaluate, num_classes=num_classes,
                      dtype=tf.float32, pos=pos)

    clf.fit(ds_source['train'], ds_source['val'], ds_target['train'],
            steps_per_epoch_val, **train_kwargs)

    result_list.append(evaluate_clf())
result[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in
                            enumerate(result_list)])


#########################################################################################
## Latent shift adaptation with known U

# Adaptation using equation (1) in the paper (https://arxiv.org/abs/2212.11254), assuming access to U in the source domain.
#########################################################################################

method = 'known_u'

result_list = []
input_shape = (x_dim,)
for seed in range(ITERATIONS):
    print(f'Iteration: {seed} method:{method}')
    tf.random.set_seed(seed)
    np.random.seed(seed)

    z = 'u'
    x2z_model = mlp(num_classes=u_dim, width=width, input_shape=(x_dim,),
                    learning_rate=learning_rate,
                    metrics=['accuracy'])

    model = mlp(num_classes=num_classes, width=width, input_shape=(x_dim + u_dim,),
                learning_rate=learning_rate,
                metrics=['accuracy'])

    clf = bbsez.Method(model, x2z_model, evaluate, num_classes=num_classes,
                       dtype=tf.float32, pos=pos, confounder=z)
    clf.fit(ds_source['train'], ds_source['val'], ds_target['train'],
            steps_per_epoch_val, **train_kwargs)

    result_list.append(evaluate_clf())
result[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in
                            enumerate(result_list)])


#########################################################################################
## VAE

# When U is unknown, we estimate it using a variational autoencoder. 'Graph-based' VAE enforces the structure of the graph in the paper in the decoder. 'Vanilla' VAE does not enforce any structure.
#########################################################################################

latent_dim = 5  #@param {type:"number"}

### Graph-Based

method = 'vae_graph'
result_list = []
input_shape = (x_dim,)
for seed in range(ITERATIONS):
    print(f'Iteration: {seed} method:{method}')
    tf.random.set_seed(seed)
    np.random.seed(seed)

    encoder = mlp(num_classes=latent_dim, width=width,
                  input_shape=(x_dim + c_dim + w_dim + num_classes,),
                  learning_rate=learning_rate,
                  metrics=['accuracy'])

    model_x2u = mlp(num_classes=latent_dim, width=width, input_shape=(x_dim,),
                    learning_rate=learning_rate,
                    metrics=['accuracy'])
    model_xu2y = mlp(num_classes=num_classes, width=width,
                     input_shape=(x_dim + latent_dim,),
                     learning_rate=learning_rate,
                     metrics=['accuracy'])
    vae_opt = tf.keras.optimizers.RMSprop(learning_rate=1e-4)

    dims = mlc.ConfigDict()
    dims.x = x_dim
    dims.y = num_classes
    dims.c = c_dim
    dims.w = u_dim
    dims.u = u_dim

    clf = gumbelmax_graph.Method(encoder, width, vae_opt,
                                 model_x2u, model_xu2y,
                                 dims, latent_dim, None,
                                 kl_loss_coef=3,
                                 num_classes=num_classes, evaluate=evaluate,
                                 dtype=tf.float32, pos=pos)

    clf.fit(ds_source['train'], ds_source['val'], ds_target['train'],
            steps_per_epoch_val, **train_kwargs)

    result_list.append(evaluate_clf())
result[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in
                            enumerate(result_list)])


### Vanilla

method = 'vae_vanilla'
result_list = []
input_shape = (x_dim,)
for seed in range(ITERATIONS):
    print(f'Iteration: {seed} method:{method}')
    tf.random.set_seed(seed)
    np.random.seed(seed)

    encoder = mlp(num_classes=latent_dim, width=width,
                  input_shape=(x_dim + c_dim + w_dim + num_classes,),
                  learning_rate=learning_rate,
                  metrics=['accuracy'])
    decoder = mlp(num_classes=x_dim + c_dim + w_dim + num_classes,
                  width=width, input_shape=(latent_dim,),
                  learning_rate=learning_rate,
                  metrics=['accuracy'])

    model_x2u = mlp(num_classes=latent_dim, width=width, input_shape=(x_dim,),
                    learning_rate=learning_rate,
                    metrics=['accuracy'])
    model_xu2y = mlp(num_classes=num_classes, width=width,
                     input_shape=(x_dim + latent_dim,),
                     learning_rate=learning_rate,
                     metrics=['accuracy'])
    vae_opt = tf.keras.optimizers.Adam(learning_rate=1e-4)

    clf = gumbelmax_vanilla.Method(encoder, decoder, vae_opt,
                                   model_x2u, model_xu2y, kl_loss_coef=3,
                                   num_classes=num_classes, evaluate=evaluate,
                                   dtype=tf.float32, pos=pos, dims=dims)
    clf.fit(ds_source['train'], ds_source['val'], ds_target['train'],
            steps_per_epoch_val, **train_kwargs)

    result_list.append(evaluate_clf())
result[method] = pd.concat([pd.DataFrame(elem).rename_axis('eval_set').reset_index().assign(iteration=i) for i, elem in
                            enumerate(result_list)])


# Results

result_df=pd.concat(result).reset_index(level=1, drop=True).rename_axis('method').reset_index()

metrics = list(evals_sklearn.keys())

results_mean = result_df.groupby(['method', 'eval_set'])[metrics].mean()
results_sd = result_df.groupby(['method', 'eval_set'])[metrics].std()
results_formatted = results_mean.applymap(lambda x: '{:.4f}'.format(x)) + u' \u00B1 ' + results_sd.applymap(lambda x: '{:.4f}'.format(x))

os.makedirs(saveFolder, exist_ok=True)
result_df.to_csv(os.path.join(saveFolder, 'method_comparison_table.csv'), index=False)

results_long = results_formatted.reset_index().melt(id_vars=['method', 'eval_set'], value_vars=metrics, var_name='metric', value_name='performance')
print(results_long.head())
results_pivot = results_long.pivot(index=['method', 'metric'], columns='eval_set')
print(results_pivot.query('metric == "cross-entropy"'))
print(results_pivot.query('metric == "accuracy"'))
print(results_pivot.query('metric == "auc"'))

results_pivot_concat = pd.concat([results_pivot, results_pivot_sk]).sort_index()
print(results_pivot_concat.query('metric == "cross-entropy"'))
print(results_pivot_concat.query('metric == "accuracy"'))
print(results_pivot_concat.query('metric == "auc"'))

method_format_dict = {
    'erm_source': 'ERM-SOURCE',
    'covar': 'COVAR',
    'label_oracle': 'LABEL',
    'bbse': 'BBSE',
    'vae_graph': 'LSA-WAE (ours)',
    'vae_vanilla': 'LSA-WAE-V',
    'known_u': 'LSA-ORACLE',
    'erm_target': 'ERM-TARGET'
    }
method_format_df = pd.DataFrame(method_format_dict, index = ['Method']).transpose().rename_axis('method').reset_index()
method_order = ['ERM-SOURCE', 'COVAR', 'LABEL', 'BBSE', 'LSA-WAE (ours)', 'LSA-WAE-V', 'LSA-ORACLE', 'ERM-TARGET']

results_to_print = results_pivot_concat.droplevel(0, axis=1).reset_index()
results_to_print = results_to_print.merge(method_format_df).drop('method', axis=1).set_index('Method')
result_to_print = results_to_print.loc[method_order]

filename = "simulation_continuous_10_seeds_{metric}.txt"
caption_text = "{metric}"
for metric in ['auc', 'cross-entropy', 'accuracy']:
  temp = result_to_print.query('metric == @metric').drop('metric', axis=1)
  with open(os.path.join(saveFolder, filename.format(metric=metric)), "w") as handle:
    temp.to_latex(
        buf=handle,
        caption=caption_text.format(metric=metric)
    )