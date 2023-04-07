from abc import ABC
from river import base
from collections import deque


class TemporallyAugmentedClassifier(base.Classifier, ABC):

    def __init__(self, base_learner:base.Classifier=None, num_old_labels:int=0):
        self._base_learner = base_learner
        self._num_old_labels = num_old_labels
        self._old_labels = deque([0]*self._num_old_labels)

    def learn_one(self, x: dict, y: base.typing.ClfTarget, **kwargs) -> "Classifier":
        x = self._extend_with_old_labels(x)
        self._base_learner.learn_one(x, y)
        self._update_past_labels(y)
        return self._base_learner

    def predict_one(self, x: dict) -> base.typing.ClfTarget:
        x = self._extend_with_old_labels(x)
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
