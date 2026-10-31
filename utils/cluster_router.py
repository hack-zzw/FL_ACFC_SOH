import numpy as np
import pandas as pd

def causal_routes(target_observations, router):
    """Only the first 20 observed feature records are read for each battery."""
    allowed = ['battery_id', 'cycle_index'] + list(router['features'])
    obs = target_observations[list(dict.fromkeys(allowed))]
    prefix = obs.sort_values(['battery_id', 'cycle_index']).groupby('battery_id').head(20)
    assert prefix.groupby('battery_id').size().eq(20).all()
    means = prefix.groupby('battery_id')[router['features']].mean()
    z = (means.to_numpy() - np.array(router['source_scaler_mean'])) / np.array(router['source_scaler_scale'])
    d2 = ((z[:, None, :] - np.array(router['cluster_centers'])[None, :, :]) ** 2).sum(axis=2)
    return pd.Series(['cluster_' + str(int(i)) for i in d2.argmin(axis=1)], index=means.index)
