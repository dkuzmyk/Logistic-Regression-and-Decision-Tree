import numpy as np
np.random.seed(1)

def sigmoid(z):
  """
  sigmoid function that maps inputs into the interval [0,1]
  Your implementation must be able to handle the case when z is a vector (see unit test)
  Inputs:
  - z: a scalar (real number) or a vector
  Outputs:
  - trans_z: the same shape as z, with sigmoid applied to each element of z
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  trans_z = 1/(1+np.exp(-z))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return trans_z

def logistic_regression(X, w):
  """
  logistic regression model that outputs probabilities of positive examples
  Inputs:
  - X: an array of shape (num_sample, num_features)
  - w: an array of shape (num_features,)
  Outputs:
  - logits: a vector of shape (num_samples,)
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  logits = sigmoid(np.dot(X,w))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return logits


def logistic_loss(X, w, y):
  """
  a function that compute the loss value for the given dataset (X, y) and parameter w;
  It also returns the gradient of loss function w.r.t w
  Here (X, y) can be a set of examples, not just one example.
  Inputs:
  - X: an array of shape (num_sample, num_features)
  - w: an array of shape (num_features,)
  - y: an array of shape (num_sample,), it is the ground truth label of data X
  Output:
  - loss: a scalar which is the value of loss function for the given data and parameters
  - grad: an array of shape (num_featues,), the gradient of loss
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  loss = np.sum(-y*np.log(logistic_regression(X,w.T))-((1-y)*np.log(1-logistic_regression(X,w.T))))/X.shape[0]
  fun = logistic_regression(X,w.T)
  grad = np.dot(-y*(1-fun),X) + np.dot((1-y)*fun,X)
  grad = grad/X.shape[0]
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return loss, grad

def softmax(x):
  """
  Convert logits for each possible outcomes to probability values.
  In this function, we assume the input x is a 2D matrix of shape (num_sample, num_classes).
  So we need to normalize each row by applying the softmax function.
  Inputs:
  - x: an array of shape (num_sample, num_classse) which contains the logits for each input
  Outputs:
  - probability: an array of shape (num_sample, num_classes) which contains the
                 probability values of each class for each input
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  probability = np.exp(x.T) / (sum(np.exp(x.T)))
  probability = probability.T
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return probability

def MLR(X, W):
  """
  performs logistic regression on given inputs X
  Inputs:
  - X: an array of shape (num_sample, num_feature)
  - W: an array of shape (num_feature, num_class)
  Outputs:
  - probability: an array of shape (num_sample, num_classes)
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  probability = np.exp(np.dot(W.T,X.T))/sum(np.exp(np.dot(W.T,X.T)))
  probability = probability.T
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return probability

def cross_entropy_loss(X, W, y):
  """
  Inputs:
  - X: an array of shape (num_sample, num_feature)
  - W: an array of shape (num_feature, num_class)
  - y: an array of shape (num_sample,)
  Ouputs:
  - loss: a scalar which is the value of loss function for the given data and parameters
  - grad: an array of shape (num_featues, num_class), the gradient of the loss function
  """
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  first =  np.log(MLR(X,W))
  loss = 0
  Xlength = X.shape[0]
  for i in range(Xlength):
    loss+= first[i][y[i]]
  loss = -loss/Xlength

  rvector = np.tile(y.T,(4,1))
  eq = MLR(X,W)
  print(eq.shape)
  grad = np.dot(X.T,(rvector.T-eq))
  grad = grad/Xlength
  print(rvector)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return loss, grad

def gini_score(groups, classes):
  '''
  Inputs:
  groups: 2 lists of examples. Each example is a list, where the last element is the label.
  classes: a list of different class labels (it's simply [0.0, 1.0] in this problem)
  Outputs:
  gini: gini score, a real number
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  # count all samples at split point
  gini_idx = np.zeros(len(groups))
  num_examples_by_group = np.zeros(len(groups))
  for idx, group in enumerate(groups):
    group = np.array(group)
    if group.size == 0:
      gini_idx[idx] = 1
      number_examples = 0
    else:
      labels = group[:,-1]
      number_examples = labels.shape[0]
      number_zeros = np.count_nonzero(labels == classes[0])
      number_ones = number_examples - number_zeros
      gini_idx[idx] = 1 - ((number_zeros/number_examples)**2 + (number_ones/number_examples)**2)

    num_examples_by_group[idx] = number_examples

  # sum weighted Gini index for each group

  gini = (1/np.sum(num_examples_by_group))*(num_examples_by_group.dot(gini_idx))
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return gini

def create_split(index, threshold, datalist):
  '''
  Inputs:
  index: The index of the feature used to split data. It starts from 0.
  threshold: The threshold for the given feature based on which to split the data.
        If an example's feature value is < threshold, then it goes to the left group.
        Otherwise (>= threshold), it goes to the right group.
  datalist: A list of samples.
  Outputs:
  left: List of samples
  right: List of samples
  '''

  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  right = []
  left = []
  for each in datalist:
    if each[index] < threshold:
      left.append(each)
    else:
      right.append(each)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return left, right

def get_best_split(datalist):
  '''
  Inputs:
  datalist: A list of samples. Each sample is a list, the last element is the label.
  Outputs:
  node: A dictionary contains 3 key value pairs, such as: node = {'index': integer, 'value': float, 'groups': a tuple contains two lists of examples}
  Pseudo-code:
  for index in range(#feature): # index is the feature index
    for example in datalist:
      use create_split with (index, example[index]) to divide datalist into two groups
      compute the Gini index for this division
  construct a node with the (index, example[index], groups) that corresponds to the lowest Gini index
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  node = {'index': '', 'value': '', 'groups': ''}

  classes = np.array(datalist)[:,-1]
  lowest_gini = 1
  for index in range(len(datalist[0])-1):
    for example in datalist:
      group1, group2 = create_split(index, example[index], datalist)
      result = gini_score((group1, group2), classes)
      if result<lowest_gini:
        lowest_gini = result
        node['index'] = index
        node['value'] = example[index]
        node['groups'] = (group1, group2)

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return node

def to_terminal(group):
  '''
  Input:
    group: A list of examples. Each example is a list, whose last element is the label.
  Output:
    label: the label indicating the most common class value in the group
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  labels = np.array(group)[:, -1]
  counts = np.bincount(labels.astype(int))  # returns count of each index (takes index as the value)
  label = np.argmax(counts)

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return label

def recursive_split(node, max_depth, min_size, depth):
  '''
  Inputs:
  node:  A dictionary contains 3 key value pairs, node =
         {'index': integer, 'value': float, 'groups': a tuple contains two lists fo samples}
  max_depth: maximum depth of the tree, an integer
  min_size: minimum size of a group, an integer
  depth: tree depth for current node
  Output:
  no need to output anything, the input node should carry its own subtree once this function ternimate
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  # check for a no split

  # check for max depth

  # process left child

  # process right child

  left_g, right_g = node['groups']
  del(node['groups'])

  if not left_g or not right_g:
    node['left'] = node['right'] = to_terminal(left_g + right_g)
    return

  if depth >= max_depth-1:
    node['left'] = to_terminal(left_g)
    node['right'] = to_terminal(right_g)
    return

  if len(left_g) <= min_size:
    node['left'] = to_terminal(left_g)
  else:
    node['left'] = get_best_split(left_g)
    recursive_split(node['left'], max_depth, min_size, depth+1)

  if len(right_g) <= min_size:
    node['right'] = to_terminal(right_g)
  else:
    node['right'] = get_best_split(right_g)
    recursive_split(node['right'], max_depth, min_size, depth+1)


  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY


def build_tree(train, max_depth, min_size):
  '''
  Inputs:
    - train: Training set, a list of examples. Each example is a list, whose last element is the label.
    - max_depth: maximum depth of the tree, an integer (root has depth 1)
    - min_size: minimum size of a group, an integer
  Output:
    - root: The root node, a recursive dictionary that should carry the whole tree
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  root = get_best_split(train)
  recursive_split(root, max_depth, min_size, depth=1)
  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
  return root

def predict(root, sample):
  '''
  Inputs:
  root: the root node of the tree. a recursive dictionary that carries the whole tree.
  sample: a list
  Outputs:
  '''
  # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

  if sample[root['index']] < root['value']:
    if isinstance(root['left'], dict):
      r = predict(root['left'], sample)
    else:
      return root['left']

  else:
    if isinstance(root['right'], dict):
      r = predict(root['right'], sample)
    else:
      return root['right']
  return r

  # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
