import pytest
import numpy as np
from plnn.data_generation.signals import get_binary_function
from plnn.data_generation.signals import get_sigmoid_function
from plnn.models import PLNN

#####################
##  Configuration  ##
#####################


###############################################################################
###############################   BEGIN TESTS   ###############################
###############################################################################

class TestBinarySignal:
    
    @pytest.mark.parametrize("tc, pi, pf, test_times, expected", [
        [
            [2, 5], 
            [0, 3], 
            [4, 2],
            [1, 2, 3, 5, 6], 
            [[0, 3], [4, 3], [4, 3], [4, 2], [4, 2]]
        ],
    ])
    def test_binary_signal_direct(self, tc, pi, pf, test_times, expected):
        f = get_binary_function(tc, pi, pf)
        ps = f(test_times)
        errors = []
        for i, t in enumerate(test_times):
            if not np.allclose(expected[i], ps[i]):
                msg = f"f(t={t:.5f}) =/= {expected[i]}. Got {ps[i]}"
                errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
        

class TestSigmoidSignal:
    
    @pytest.mark.parametrize("tc, pi, pf, r, test_times, expected", [
        [
            [2, 3], 
            [0, 3], 
            [4, 2],
            [1, 2],
            [1, 2, 3, 5, 6], 
            [[0.476811688088, 2.99966464987], 
             [2, 2.98201379004], 
             [3.52318831191, 2.5], 
             [3.99010950737, 2.00033535013], 
             [3.99865859948, 2.00000614417]]
        ],
    ])
    def test_sigmoid_signal_direct(self, tc, pi, pf, r, test_times, expected):
        f = get_sigmoid_function(*[np.array(xx) for xx in [tc, pi, pf, r]])
        ps = f(test_times)
        errors = []
        for i, t in enumerate(test_times):
            if not np.allclose(expected[i], ps[i]):
                msg = f"f(t={t:.5f}) =/= {expected[i]}. Got {ps[i]}"
                errors.append(msg)
        assert not errors, "Errors occurred:\n{}".format("\n".join(errors))
