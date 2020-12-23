import numpy as np
import random
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler

class FeatureTransformer:
    def __init__(self, env):
        observation_samples = np.array([[random.uniform(-10, 17), random.uniform(-25, 25), random.uniform(-2.6, 2.6), random.uniform(-30, 30)] for _ in range(1000)])
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
