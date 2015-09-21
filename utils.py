__author__ = 'Frederic Godin  (frederic.godin@ugent.be / www.fredericgodin.com)'

from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
import lasagne.updates


# Adapted from Lasagne

def adagrad(loss_or_grads, params, learning_rate=1.0, epsilon=1e-6):
    """Adagrad updates
    Scale learning rates by dividing with the square root of accumulated
    squared gradients. See [1]_ for further description.
    Parameters
    ----------
    loss_or_grads : symbolic expression or list of expressions
        A scalar loss expression, or a list of gradient expressions
    params : list of shared variables
        The variables to generate update expressions for
    learning_rate : float or symbolic scalar
        The learning rate controlling the size of update steps
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    Returns
    -------
    OrderedDict
        A dictionary mapping each parameter to its update expression
    Notes
    -----
    Using step size eta Adagrad calculates the learning rate for feature i at
    time step t as:
    .. math:: \\eta_{t,i} = \\frac{\\eta}
       {\\sqrt{\\sum^t_{t^\\prime} g^2_{t^\\prime,i}+\\epsilon}} g_{t,i}
    as such the learning rate is monotonically decreasing.
    Epsilon is not included in the typical formula, see [2]_.
    References
    ----------
    .. [1] Duchi, J., Hazan, E., & Singer, Y. (2011):
           Adaptive subgradient methods for online learning and stochastic
           optimization. JMLR, 12:2121-2159.
    .. [2] Chris Dyer:
           Notes on AdaGrad. http://www.ark.cs.cmu.edu/cdyer/adagrad.pdf
    """

    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()
    accus = []

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                             broadcastable=param.broadcastable)
        accu_new = accu + grad ** 2
        updates[accu] = accu_new
        accus.append((accu,value.shape))
        updates[param] = param - (learning_rate * grad /
                                  T.sqrt(accu_new + epsilon))

    return updates,accus

def reset_grads(accus):
    for accu in accus:
        accu[0].set_value(np.zeros(accu[1], dtype=accu[0].dtype))