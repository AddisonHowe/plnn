"""Tests for Non-Euclidean DeepPhiPLNN methods.

"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom

from plnn.models import NEDeepPhiPLNN

#####################
##  Configuration  ##
#####################

W1 = np.array([
    [1, 3],
    [2, 2],
    [3, 1],
], dtype=float)

W2 = np.array([
    [1, 1, -2],
    [0, 1, 0],
    [-1, 2, 1],
], dtype=float)

W3 = np.array([
    [2, 3, 1]
], dtype=float)

WT1 = np.array([
    [2, 4],
    [-1, 1],
], dtype=float)

###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################
