
from enum import Enum

class TrainExitStatus( Enum ):
    success = 0
    error = 1 # for consistency with python's default error exit status
    malformed_command = 2
    reached_update_limit = 3
    interrupted = 4
    nan_loss = 5
    overfitting = 6
