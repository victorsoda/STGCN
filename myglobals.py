
# filepath
distance_filepath = 'data/distance.csv'
params_file_prefix = 'stgcn_params/stgcn.params_'

# dataset related
train_prop = 0.6
valid_prop = 0.2
test_prop = 1 - train_prop - valid_prop
num_points_for_train = 12
num_points_for_predict = 3
num_input_features = 3


# hyper parameters
normalized_k_threshold = 0.1    # in distance weight matrix
cheb_K = 3      # up K in cheb polynomial
learning_rate = 1e-3
optimizer = 'RMSprop'
decay_rate = 0.7
decay_interval = 5
epochs = 20
batch_size = 10

load_epoch = 0


