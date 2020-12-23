import numpy as np
from FeatureTransformer import FeatureTransformer
from sklearn.linear_model import SGDRegressor


class Agent:

    def __init__(self, env):
        self.env = env
        self.models = []
        self.feature_transformer = FeatureTransformer(env)
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit(self.feature_transformer.transform([env.reset()]), [0]) #(X,Y) = (State, Reward) for this action
            self.models.append(model)

    def update_policy(self, observation, action, reward):
        X = self.feature_transformer.transform([observation])
        self.models[action].partial_fit(X, [reward])

    def act(self, observation):
        X = self.feature_transformer.transform([observation])
        result = np.stack([m.predict(X) for m in self.models]).T
        return np.argmax(result)

