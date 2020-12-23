import numpy as np
import random
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class FeatureTransformer:
    def __init__(self, env):
        observation_samples = np.array([[random.uniform(-1.0, 1.7), random.uniform(-2.5, 2.5), random.uniform(-0.26, 0.26), random.uniform(-3.0, 3.0)] for _ in range(1000)])
        scaler = StandardScaler()
        scaler.fit(observation_samples)
        feature_union = FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=500)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=500)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=500)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=500))
        ])
        feature_union.fit(scaler.transform(observation_samples))
        self.scaler = scaler
        self.feature_union = feature_union

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        return self.feature_union.transform(scaled)
