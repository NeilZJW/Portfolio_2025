# -*- coding: utf-8 -*-
# @Author  : Neil
# @Time    : 2024/5/6 15:20

import numpy as np
from sklearn.decomposition import IncrementalPCA


class IPCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.whiten = False
        self.transformer = IncrementalPCA(
            n_components,
            whiten=self.whiten,
            batch_size=max(100, 5 * n_components)
        )
        self.batch_support = True

    def get_param_str(self):
        return "IPCA_c{}{}".format(
            self.n_components, "_w" if self.whiten else ""
        )

    def fit(self, X):
        self.transformer.fit(X)

    def fit_partial(self, X):
        try:
            self.transformer.partial_fit(X)
            # avoid overflow
            self.transformer.n_samples_seen_ = \
                self.transformer.n_samples_seen_.astype(np.int64)
            return True
        except ValueError as e:
            print(f"\nIPCA ERROR: {e}")
            return False

    def get_component(self):
        # already sorted
        stdev = np.sqrt(self.transformer.explained_variance_)
        var_ratio = self.transformer.explained_variance_ratio_
        # PCA outputs are normalized
        return self.transformer.components_, stdev, var_ratio
