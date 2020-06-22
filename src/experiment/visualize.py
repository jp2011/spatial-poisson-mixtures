import corner
import matplotlib.pyplot as plt
import pandas as pd


def plot_traceplots(samples, names, layout=None):
    """
    A multi-plot for the traceplot of N samples of M random variables
    :param samples: array_like[N, M]
    :param names:  a list of labels of size M
    :param layout: (x, y): 2D layout with x rows, y coulmns
    :return:
    """

    n_samples, ndim = samples.shape
    assert ndim == len(names)

    plot_layout = layout if layout else (int((ndim + 1) / 2), 2)

    plt.figure()
    ts = pd.DataFrame(samples, columns=names)
    return ts.plot(
        subplots=True,
        linewidth=0.5,
        color='k',
        layout=plot_layout,
        figsize=(12,12))


def plots_mcmc_histograms(samples, names=None):
    """
    A summary plot for MCMC samples as provided by the 'corner' package.
    """
    return corner.corner(samples, labels=names)
