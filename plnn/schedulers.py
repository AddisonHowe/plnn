"""Training hyperparameter schedulers.

"""

from abc import ABC, abstractmethod
import numpy as np


class AbstractTimestepSchedule(ABC):

    def __init__(self, schedule):
        self.schedule = schedule

    @abstractmethod
    def get_dt(self, epoch, **kwargs):
        """Return the value of dt at the given epoch."""
        raise NotImplementedError()
    

class ConstantTimestepSchedule(AbstractTimestepSchedule):

    def __init__(self, value):
        super().__init__(schedule='constant')
        self.value = value

    def get_dt(self, epoch, **kwargs):
        return self.value
    

class SteppedTimestepSchedule(AbstractTimestepSchedule):

    def __init__(self, change_epochs, values):
        super().__init__(schedule='stepped')
        change_epochs = np.sort(change_epochs)
        self._check_args(change_epochs, values)
        self.change_epochs = change_epochs
        self.values = values

    def get_dt(self, epoch, **kwargs):
        
    
    def _check_args(self, change_epochs, values):
        assert change_epochs[0] == 0, "First change epoch must be 0. " \
            f"Got {change_epochs}."
        assert len(change_epochs) == len(values), "Length of change_epochs " \
            f"and values must match. Got {change_epochs} and {values}."
        assert np.all(change_epochs >= 0), "All change epochs must be " \
            f"nonnegative. Got {change_epochs}."
        assert len(change_epochs) > 0, "Got empty change_epochs."
        return True

TIMESTEP_SCHEDULERS = {
    'constant' : ConstantTimestepSchedule,
    'stepped' : SteppedTimestepSchedule,
}