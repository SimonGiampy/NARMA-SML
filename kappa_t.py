from river.metrics import base

__all__ = ["CohenKappaTemporal"]


class CohenKappaTemporal(base.MultiClassMetric):
    r"""Cohen's Kappa score.

    Cohen's Kappa expresses the level of agreement between two annotators on a classification
    problem. It is defined as

    $$
    \kappa = (p_o - p_e) / (1 - p_e)
    $$

    where $p_o$ is the empirical probability of agreement on the label
    assigned to any sample (prequential accuracy), and $p_e$ is
    the expected agreement when both annotators assign labels randomly.

    Parameters
    ----------
    cm
        This parameter allows sharing the same confusion
        matrix between multiple metrics. Sharing a confusion matrix reduces the amount of storage
        and computation time.

    """

    def get(self):

        try:
            p0 = (self.cm.total_true_positives + self.cm.total_true_negatives) / self.cm.n_samples  # same as accuracy
        except ZeroDivisionError:
            p0 = 0

        if self.cm.n_samples != 0:
            pe = (
                (self.cm.total_true_positives + self.cm.total_false_negatives)
                * (self.cm.total_true_positives + self.cm.total_false_positives)
                + (self.cm.total_true_negatives + self.cm.total_false_negatives)
                * (self.cm.total_true_negatives + self.cm.total_false_positives)
            ) / (self.cm.n_samples ** 2)
        else:
            pe = 0

        try:
            return (p0 - pe) / (1 - pe)
        except ZeroDivisionError:
            return 0.0
