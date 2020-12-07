import numpy as np

class Model:
    def __init__(self):
        pass

    def predict(self, obs):
        return np.random.choice([0,1], size=3)
