import base

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

    Examples
    --------

    >>> from river import metrics

    >>> y_true = ['cat', 'ant', 'cat', 'cat', 'ant', 'bird']
    >>> y_pred = ['ant', 'ant', 'cat', 'cat', 'ant', 'cat']

    >>> metric = metrics.CohenKappa()

    >>> for yt, yp in zip(y_true, y_pred):
    ...     metric = metric.update(yt, yp)

    >>> metric
    CohenKappa: 42.86%

    References
    ----------
    [^1]: J. Cohen (1960). "A coefficient of agreement for nominal scales". Educational and Psychological Measurement 20(1):37-46. doi:10.1177/001316446002000104.

    """

    def get(self):

        try:
            p0 = self.cm.total_true_positives / self.cm.n_samples  # same as accuracy
        except ZeroDivisionError:
            p0 = 0

        if self.cm.n_samples != 0:
            pe = self.cm.correct_no_change / self.cm.n_samples
        else:
            pe = 0

        try:
            return (p0 - pe) / (1 - pe)
        except ZeroDivisionError:
            return 0.0
