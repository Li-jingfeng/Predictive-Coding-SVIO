import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def hard_sigmoid(x):
    '''
    - hard sigmoid function by zcr.
    - Computes element-wise hard sigmoid of x.
    - what is hard sigmoid?
        Segment-wise linear approximation of sigmoid. Faster than sigmoid.
        Returns 0. if x < -2.5, 1. if x > 2.5. In -2.5 <= x <= 2.5, returns 0.2 * x + 0.5.
    - See e.g. https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/sigm.py#L279
    '''
    slope = 0.2
    shift = 0.5
    x = (slope * x) + shift
    x = F.threshold(-x, -1, -1)
    x = F.threshold(-x, 0, 0)
    return x

def get_activationFunc(act_str):
    act = act_str.lower()
    if act == 'relu':
        # return nn.ReLU(True)
        return nn.ReLU()
        # return F.relu()
    elif act == 'tanh':
        # return F.tanh
        return nn.Tanh()
    # elif act == 'hard_sigmoid':
    #     return hard_sigmoid
    else:
        raise(RuntimeError('cannot obtain the activation function named %s' % act_str))

def batch_flatten(x):
    '''
    equal to the `batch_flatten` in keras.
    x is a Variable in pytorch
    '''
    shape = [*x.size()]
    dim = np.prod(shape[1:])
    dim = int(dim)      # 不加这步的话, dim是<class 'numpy.int64'>类型, 不能在view中用. 加上这步转成<class 'int'>类型.
    return x.view(-1, dim)