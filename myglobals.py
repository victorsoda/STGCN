
# filepath
distance_filepath = 'data/distance.csv'

# dataset related
train_prop = 0.6
valid_prop = 0.2
test_prop = 1 - train_prop - valid_prop
num_points_for_train = 12
num_points_for_predict = 9


# hyper parameters
normalized_k_threshold = 0.1    # in distance weight matrix
cheb_K = 3      # up K in cheb polynomial
learning_rate = 1e-3
optimizer = 'RMSprop'
decay_rate = 0.7
decay_interval = 5
epochs = 50
batch_size = 25



