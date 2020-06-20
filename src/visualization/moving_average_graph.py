import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def moving_average_graph(df, ind_var, n, x_label='', y_label='', tick_pos=[], tick_label=[]):
    """
    Returns a rolling average line graph of the independent variable vs the probability of being a one-hit wonder
    :param df: The dataframe of potential one-hit wonders
    :param ind_var: The name of the column that you want to be the independent variable
    :param n: How many datapoints to include in the rolling average
    :param x_label: The name of the x-label on the graph
    :param y_label: The name of the y-label on the graph
    :param tick_pos: The tick positions you want to label as a list (default no ticks)
    :param tick_label: The labels of the tick positions (default no labels)
    :return: A line graph
    """
    sns.set_style("dark")
    year_graph = df.sort_values(by=ind_var)['one_hit']
    year_graph = np.convolve(year_graph, np.ones((n,)) / n, mode='valid') * 100
    sns.lineplot(x=range(len(year_graph)), y=year_graph)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(tick_pos, tick_label);