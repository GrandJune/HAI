import numpy as np

# Example: Define N
N = 5  # Example value
Q_table = np.zeros((2 ** N, N + 1))

# Modify some rows to be non-zero for demonstration
Q_table[3, 2] = 1
Q_table[7, 3] = 2

# Count non-zero rows
non_zero_rows = np.count_nonzero(np.any(Q_table != 0, axis=1))

# Calculate percentage
percentage = (non_zero_rows / Q_table.shape[0]) * 100

print(f"Percentage of non-zero rows: {percentage:.2f}%")
print(2 / 2 ** 5)
# -*- coding: utf-8 -*-
# @Time     : 3/13/2025 19:44
# @Author   : Junyi
# @FileName: Test.py
# @Software  : PyCharm
# Observing PEP 8 coding style
