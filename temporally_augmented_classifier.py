from abc import ABC
from river import base
from collections import deque


class TemporallyAugmentedClassifier(base.Classifier, ABC):

    def __init__(self, base_learner:base.Classifier=None, num_old_labels:int=0, num_old_features: int = 0):
        self._base_learner = base_learner
        self._num_old_labels = num_old_labels
        self._old_labels = deque([0] * self._num_old_labels)
        # additional features
        self._num_old_features = num_old_features
        self._old_features = deque([0] * self._num_old_features * 6) # 6 is the number of features

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> "Classifier":
        x = self._extend_with_old_labels(x)
        x = self._extend_with_old_features(x) # addition
        self._base_learner.learn_one(x, y)
        self._update_past_labels(y)
        self._update_past_features(x) # addition
        return self._base_learner

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        x = self._extend_with_old_labels(x)
        x = self._extend_with_old_features(x) # addition
        return self._base_learner.predict_one(x)

    def _update_past_labels(self, y):
        self._old_labels.append(y)
        self._old_labels.popleft()

    def _extend_with_old_labels(self, x):
        x_ext = x.copy()
        ext = range(len(x_ext.keys()), len(x_ext.keys()) + self._num_old_labels)
        for el, old_label in zip(ext, list(self._old_labels)):
            # check on type of keys, if string or int
            if isinstance(list(x_ext.keys())[0], type('str')):
                x_ext[str(el)] = old_label
            else:
                x_ext[el] = old_label
        return x_ext
    
    # additional function
    def _update_past_features(self, x):
        for i in range(1, 7):
            self._old_features.append(x["feature" + str(i)])
            self._old_features.popleft()

    # additional function
    def _extend_with_old_features(self, x):
        x_ext = x.copy()
        ext = range(len(x_ext.keys()), len(x_ext.keys()) + self._num_old_features * 6)
       
        for el, old_feature in zip(ext, list(self._old_features)):
            # check on type of keys, if string or int
            if isinstance(list(x_ext.keys())[0], type('str')):
                x_ext[str(el)] = old_feature
            else:
                x_ext[el] = old_feature
        return x_ext
