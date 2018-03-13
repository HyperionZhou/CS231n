import numpy as np
import operator
from past.builtins import xrange
from collections import Counter

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      for j in xrange(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
         #dists[i][j] = np.sqrt(np.sum(np.square(self.X_train[j, :] - X[i, :]), axis = 0))
            dists[i][j] = np.linalg.norm(X[i] - self.X_train[j])
            #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
        #dists[i] = np.sqrt(np.sum(np.square(self.X_train - X[i,:]), axis = 1))
        cloned_array = np.array([X[i]] * num_train)
        # [X[i]]: clone vertically; trade time with space allocation
        dists[i] = np.linalg.norm(self.X_train - cloned_array, axis = 1)
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    '''
    train_2 = self.X_train * self.X_train
    train_2 = np.sum(train_2, axis = 1)
    
    test_2 = X * X
    test_2 = np.sum(test_2, axis = 1)
    
    test_train = np.dot(X, np.transpose(self.X_train))
    
    dists = (2 * test_train - train_2)
    dists = np.transpose(dists) - test_2
    dists = -1 * dists
    dists = np.transpose(dists)
    dists = np.sqrt(dists)
   ''' 
    X_train_2 = self.X_train*self.X_train
    X_train_2 = np.sum(X_train_2, axis = 1)
    
    X_train_2_repeat = np.array([X_train_2]*X.shape[0])
    
    X_2 = X*X
    X_2 = np.sum(X_2, axis = 1)
    X_2_repeat = np.array( [X_2]*self.X_train.shape[0]).transpose()
    
    X_dot_X_train = X.dot(self.X_train.T)
    
#     print X_train_2_repeat.shape
#     print X_2_repeat.shape
#     print X_dot_X_train.shape
#     
#     
#     print X_train_2_repeat ,"\n"
#     print X_2_repeat ,"\n"
#     print 2*X_dot_X_train
    dists = X_train_2_repeat + X_2_repeat - 2*X_dot_X_train
    dists = np.sqrt(dists)

    
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
        dists_i = dists[i]
        #closest_y = np.argsort(dists_i[i,:])[0:k]
        closest_y = self.y_train[dists_i.argsort()[:k]]
         #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
        '''
        a = {}
        for j in xrange(k):
            closest_y[j] = self.y_train[closest_y[j]]
            if closest_y[j] in a:
                a[closest_y[j]] = a[closest_y[j]] + 1
            else:
                a[closest_y[j]] = 1
        y_pred[i] = max(a.iteritems(), key = operator.itemgetter(1))[0]\
        '''
        #unique, counts = np.unique(closest_y, return_counts = True)
        #y_pred[i] = dict(zip(unique, counts))
        y_pred[i] = Counter(closest_y).most_common(1)[0][0]
        
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################
    return y_pred

