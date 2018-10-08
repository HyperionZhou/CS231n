import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  score = X.dot(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  
  for i in range(num_train):
    #numeric stability
    score[i] -= np.max(score[i])
    exp_score = np.exp(score[i])
    prob = exp_score / np.sum(exp_score)
    
    #loss += - np.log(np.max(score[i])/np.sum(score[i]))
    
    
    for j in range(num_class):
        if j == y[i]:
            dW[:,j] += np.transpose(X[i]) * (prob[j] - 1)
            loss += - np.log(prob[j])
        else:
            dW[:,j] += np.transpose(X[i]) * prob[j]
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  max_scores = np.reshape(np.max(scores, axis = 1), [-1,1])
  scores -= max_scores
  scores_exp = np.exp(scores)
  prob = scores_exp / np.reshape(np.sum(scores_exp, axis = 1),[-1,1])
  loss += np.sum(-1 * np.log(prob[np.arange(num_train), y]))
  
  correct_ans = np.zeros_like(prob)
  correct_ans[np.arange(num_train), y] = 1
  prob_dw = prob - correct_ans
  dW = np.transpose(X).dot(prob_dw)
  dW /= num_train
  dW += 2 * reg * W

  loss /= num_train
  loss += 2 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

