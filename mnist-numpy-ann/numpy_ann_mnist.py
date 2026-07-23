import time
import random
import json
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras.datasets import mnist
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]).astype('float32') / 255.0
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).astype('float32') / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

def sigmoid(z):
  return 1 / (1 + np.exp(-z))


def softmax(Z):
  expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
  return expZ / np.sum(expZ, axis=0, keepdims=True)


def relu(z):
  return np.maximum(z, 0)


def tanh(x):
  return np.tanh(x)


def derivative_relu(z):
  return np.array(z > 0, dtype=np.float32)


def derivative_tanh(x):
  return 1 - np.power(np.tanh(x), 2)


def initialize_parameters(layer_dims):
  parameters = {}
  L = len(layer_dims) - 1

  for l in range(1, L+1):
    #A weight matrix W for a given layer l connects all neurons from the previous layer l-1 to all neurons in the current layer l. If you have N neurons in l-1 and M neurons in l, the weight matrix W will have dimensions (M, N)
    parameters['w'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
    #reason for this scaling is to control the variance of activations throughout network
    parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))

  return parameters


layer_dims = [X_train.shape[1], 100, 200, 10]
params = initialize_parameters(layer_dims)

for l in range(1, len(layer_dims)):
  print("Shape of W" + str(l) + ":", params['w' + str(l)].shape)
  print("Shape of B" + str(l) + ":", params['b' + str(l)].shape, "\n")


def forward_propagation(X, parameters, activation):
  forward_cache = {}
  L = len(parameters) // 2  #total number of layers without input
  forward_cache['A0'] = X.T

  for l in range(1,L):
    forward_cache['z'+str(l)] = parameters['w'+str(l)].dot(forward_cache['A'+str(l-1)]) + parameters['b'+str(l)]  # Z1 = W1.A0 + B1
    #apply activation A1 = activation(Z1)
    if activation == 'relu':
      forward_cache['A'+str(l)] = relu(forward_cache['z'+str(l)])
    else:
      forward_cache['A'+str(l)] = tanh(forward_cache['z'+str(l)])

  #forward propagation for last layer
  forward_cache['z'+str(L)] = parameters['w'+str(L)].dot(forward_cache['A'+str(L-1)]) + parameters['b'+str(L)]
  # for one class classification -> sigmoid, for multiclass -> softmax
  if forward_cache['z'+str(L)].shape[0] == 1:
    forward_cache['A'+str(L)] = sigmoid(forward_cache['z'+str(L)])
  else:
    forward_cache['A'+str(L)] = softmax(forward_cache['z'+str(L)])

  return forward_cache['A'+str(L)], forward_cache


aL, forw_cache = forward_propagation(X_train, params, 'relu')
for l in range(len(params) // 2 + 1):
  print('Shape of A' + str(l) + ':', forw_cache['A'+str(l)].shape, '\n')


def compute_cost(AL, y):
  m = y.shape[0]
  AL = np.clip(AL, 1e-12, 1 - 1e-12)
  # For binary classification: Cost = (−1/m) ∑[y∗log(aL)+(1−y)∗log(1−aL)]
  # For multi-class classification: Cost = (−1/m) ∑∑[yk∗log(ak)]
  if y.shape[1] == 1:
    cost = -(1./m) * (np.dot(y, np.log(AL)) + np.dot(1-y, np.log(1-AL)))
  else:
    cost = -(1/m) * np.sum(y.T * np.log(AL))
  cost = np.squeeze(cost)
  return cost


def backward_propagation(AL, y, parameters, forward_cache, activation):
  grads = {}
  L = len(parameters) // 2
  m = y.shape[0]
  # For last layer,  dZL  will be (dCost/dAL) * (dAL/dZL) = AL−Y 
  grads['dz' + str(L)] = AL - y.T
  grads['dw' + str(L)] = (1/m) * grads['dz' + str(L)].dot(forward_cache['A' + str(L-1)].T)
  grads['db' + str(L)] = (1/m) * np.sum(grads['dz' + str(L)], axis=1, keepdims=True)
  # Except for last layer, we use a loop to implement backprop for other layers
  for l in reversed(range(1, L)):
    # dz_l = d(Loss)/d(Z_l) = (d(Loss)/d(A_l)) * (d(A_l)/d(Z_l))
    dz_prev_factor = np.dot(parameters['w' + str(l+1)].T, grads['dz' + str(l+1)])
    if activation == 'relu':
      grads['dz' + str(l)] = dz_prev_factor * derivative_relu(forward_cache['A' + str(l)])
    elif activation == 'tanh':
      grads['dz' + str(l)] = dz_prev_factor * derivative_tanh(forward_cache['z' + str(l)])
    else:
      raise ValueError("Unsupported activation for intermediate layers in backpropagation")
    
    grads['dw' + str(l)] = (1/m) * np.dot(grads['dz' + str(l)], forward_cache['A' + str(l-1)].T)
    grads['db' + str(l)] = (1/m) * np.sum(grads['dz' + str(l)], axis=1, keepdims=True)

  return grads


grads = backward_propagation(forw_cache['A' + str(3)], y_train, params, forw_cache, 'relu')
for l in reversed(range(1, len(grads) // 3 + 1)):
  print('shape of dz' + str(l) + ':', grads['dz' + str(l)].shape)
  print('shape of dw' + str(l) + ':', grads['dw' + str(l)].shape)
  print('shape of db' + str(l) + ':', grads['db' + str(l)].shape, '\n')


def update_parameters(parameters, grads, learning_rate):
  L = len(parameters) // 2
  for l in range(L):
    parameters['w' + str(l+1)] = parameters['w' + str(l+1)] - learning_rate * grads['dw' + str(l+1)]
    parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]
  return parameters


def predict(X, y, parameters, activation):
  m = X.shape[0]
  y_pred, caches = forward_propagation(X, parameters, activation)

  if y.shape[1] == 1:
    y_pred = np.array(y_pred > 0.5, dtype='float')
  else:
    y_pred = np.argmax(y_pred, axis=0)   # (10, m) -> (m,)
    y = np.argmax(y, axis=1)             # (m, 10) -> (m,)
  
  return np.round(np.sum((y_pred==y)/m), 2)


def model(X, y, layer_dims, learning_rate=0.009, activation='relu', num_iterations=3000):
  np.random.seed(42)
  parameters = initialize_parameters(layer_dims)
  history = {'iter': [], 'cost': [], 'train_acc': [], 'test_acc': []}

  for i in range(0, num_iterations):
    AL, forward_cache = forward_propagation(X, parameters, activation)
    cost = compute_cost(AL, y)
    grads = backward_propagation(AL, y, parameters, forward_cache, activation)
    parameters = update_parameters(parameters, grads, learning_rate)

    if i % (num_iterations/30) == 0:
      train_acc = predict(X_train, y_train, parameters, activation)
      test_acc = predict(X_test, y_test, parameters, activation)
      history['iter'].append(i)
      history['cost'].append(float(cost))
      history['train_acc'].append(float(train_acc))
      history['test_acc'].append(float(test_acc))
      print("\niter:{} \t cost: {} \t train_acc:{} \t test_acc:{}".format(i, np.round(cost, 2), train_acc, test_acc))
    if i % 10 == 0:
      print("==", end = '')

  return parameters, history


def plot_training_history(history, save_path='training_curve.png'):
  surface = '#fcfcfb'
  ink_primary = '#0b0b0b'
  ink_secondary = '#52514e'
  ink_muted = '#898781'
  grid_color = '#e1e0d9'
  axis_color = '#c3c2b7'
  color_train = '#2a78d6'  # categorical slot 1 - blue
  color_test = '#1baf7a'   # categorical slot 2 - aqua

  plt.rcParams['font.family'] = 'sans-serif'
  fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
  fig.patch.set_facecolor(surface)

  ax = axes[0]
  ax.set_facecolor(surface)
  ax.plot(history['iter'], history['cost'], color=color_train, linewidth=2)
  ax.set_title('Training Loss', color=ink_primary, fontsize=13, fontweight='bold', loc='left')
  ax.set_xlabel('Iteration', color=ink_secondary, fontsize=10)
  ax.set_ylabel('Cost', color=ink_secondary, fontsize=10)
  ax.grid(True, color=grid_color, linewidth=0.8)
  ax.set_axisbelow(True)
  for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
  for spine in ('left', 'bottom'):
    ax.spines[spine].set_color(axis_color)
  ax.tick_params(colors=ink_muted, labelsize=9)

  ax = axes[1]
  ax.set_facecolor(surface)
  ax.plot(history['iter'], history['train_acc'], color=color_train, linewidth=2, label='Train')
  ax.plot(history['iter'], history['test_acc'], color=color_test, linewidth=2, label='Test')
  ax.set_title('Accuracy', color=ink_primary, fontsize=13, fontweight='bold', loc='left')
  ax.set_xlabel('Iteration', color=ink_secondary, fontsize=10)
  ax.set_ylabel('Accuracy', color=ink_secondary, fontsize=10)
  ax.set_ylim(0, 1)
  ax.grid(True, color=grid_color, linewidth=0.8)
  ax.set_axisbelow(True)
  for spine in ('top', 'right'):
    ax.spines[spine].set_visible(False)
  for spine in ('left', 'bottom'):
    ax.spines[spine].set_color(axis_color)
  ax.tick_params(colors=ink_muted, labelsize=9)
  ax.legend(frameon=False, fontsize=10, labelcolor=ink_secondary, loc='lower right')

  plt.tight_layout()
  plt.savefig(save_path, dpi=300, facecolor=surface, bbox_inches='tight')
  plt.close(fig)
  print(f"Saved training curve to {save_path}")


lr = 0.0075
iters = 2100

parameters, history = model(X_train, y_train, layer_dims, learning_rate = lr, activation = 'relu', num_iterations = iters)

np.savez('mnist_ann_parameters.npz', **parameters)
with open('training_history.json', 'w') as f:
  json.dump(history, f, indent=2)
plot_training_history(history, 'training_curve.png')
