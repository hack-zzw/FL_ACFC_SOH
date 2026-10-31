"""Shared helpers retained for final experiments and validation.

Historical search/runner code is recoverable from the organization backup.
"""
import numpy as np
from configs.schema import 特征列

class RobustSourceScaler:

    def __init__(self, record):
        self.record = record

    def 变换(self, frame):
        columns = 特征列 + ['cycle_index']
        values = frame[columns].to_numpy(float)
        for name in self.record['log_features']:
            i = columns.index(name)
            values[:, i] = np.log1p(np.maximum(values[:, i], 0))
        values[:, :16] = np.clip(values[:, :16], self.record['lower'], self.record['upper'])
        return ((values - np.array(self.record['mean'])) / np.array(self.record['std'])).astype(np.float32)
