
from enum import Enum

class TrainExitStatus( Enum ):
    success = 0
    error = 1 # for consistency with python's default error exit status
    reached_update_limit = 2
    interrupted = 3
    nan_loss = 4
    overfitting = 5
