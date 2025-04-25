# -*- coding: utf-8 -*-
# @Time     : 4/20/2025 20:06
# @Author   : Junyi
# @FileName: Parrot.py
# @Software  : PyCharm
# Observing PEP 8 coding style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from Reality import Reality

class Parrot:
    def __init__(self, N=10):
        self.N = N
        self.Q_table = np.zeros((2 ** self.N, self.N))
        self.knowledge = 0

    def aggregate_from_data(self, Q_table_list=None):
        # Stack all Q_tables and take mean along axis 0 (across tables)
        self.Q_table = np.mean(np.stack(Q_table_list), axis=0)
        self.knowledge = np.count_nonzero(np.any(self.Q_table != 0, axis=1)) / (2 ** self.N)

    def visualize_1(self):
        # Custom colormap: Light gray for zero, then blue → red for positive values
        colors = [(0.9, 0.9, 0.9), "blue", "red"]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)

        # Use a logarithmic scale to enhance contrast if needed
        norm = mcolors.SymLogNorm(linthresh=1, linscale=0.5, vmin=0, vmax=np.max(self.Q_table))

        plt.figure(figsize=(8, 6))
        plt.imshow(self.Q_table, cmap=cmap, aspect='auto', norm=norm)
        plt.colorbar(label="Q-value")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.title("Heatmap of Q-table (Zero-dominant)")
        plt.show()





