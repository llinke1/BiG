import numpy as np

class fileloader:
    def __init__(self, filetype="numpy"):
        self.filetype=filetype


    def load(self, filename):
        
        if self.filetype=="numpy":
            field = np.load(filename)
        else:
            raise NotImplementedError("File loader can currently only handle numpy arrays")
        
        return field