import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def plot_hist_p_value(list_p_value):
    fig, ax = plt.subplots(figsize = (6, 6))

    ax.hist(list_p_value, bins = 10, edgecolor = 'black', color = 'black', alpha = 0.68)
    
    ax.axvline(x = 0.05, color = 'pink', linestyle = '--', linewidth = 2, label = 'alpha = 0.05')
    
    ax.set_title("P-value Distribution")
    ax.set_xlabel("P-value")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_ecdf_p_value(list_p_value, label = "P_value"):
    plt.rcParams.update({'font.size': 18})
    grid = np.linspace(0, 1, 101)

    fig, ax = plt.subplots(figsize = (6, 6))

    ecdf = sm.distributions.ECDF(np.array(list_p_value))
    
    ax.plot(grid, ecdf(grid), color = 'magenta', linestyle = '-', linewidth = 3, label = label)
    
    ax.plot([0, 1], [0, 1], 'k--')
    
    ax.set_title(label)
    ax.legend()

    plt.tight_layout()
    plt.show()