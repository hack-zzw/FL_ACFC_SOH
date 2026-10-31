"""回归指标：MAE、MAPE、RMSE 和分层 R²。"""
import numpy as np

def 回归指标(真实值: np.ndarray, 预测值: np.ndarray) -> dict[str, float]:
    y = np.asarray(真实值, dtype=float).reshape(-1)
    pred = np.asarray(预测值, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(pred):
        raise ValueError('真实值和预测值必须为相同的非空长度')
    error = pred - y
    variance = float(np.square(y - y.mean()).sum())
    r2 = float('nan') if variance <= 1e-12 else 1.0 - float(np.square(error).sum()) / variance
    return {'mae': float(np.abs(error).mean()), 'mape': float((np.abs(error) / np.maximum(np.abs(y), 1e-08)).mean() * 100.0), 'rmse': float(np.sqrt(np.square(error).mean())), 'r2': r2}

def 分层回归指标(真实值: np.ndarray, 预测值: np.ndarray, 分组: np.ndarray) -> dict[str, float]:
    """返回记录汇总和电池内分层的 R²。

    ``r2`` 保留原有的记录汇总口径，便于兼容历史结果。
    ``r2_within_battery`` 使用每个分组自己的真值均值作为基线，
    ``r2_battery_macro`` 是各分组 R² 的等权平均。分组内真值方差为零的
    组不参与宏平均，并通过 ``r2_battery_valid`` 显式报告有效组数。
    """
    y = np.asarray(真实值, dtype=float).reshape(-1)
    pred = np.asarray(预测值, dtype=float).reshape(-1)
    groups = np.asarray(分组).reshape(-1)
    if len(y) == 0 or len(y) != len(pred) or len(y) != len(groups):
        raise ValueError('真实值、预测值和分组必须为相同的非空长度')
    result = 回归指标(y, pred)
    within_sse = 0.0
    within_sst = 0.0
    group_r2: list[float] = []
    for group in pd_unique(groups):
        mask = groups == group
        gy = y[mask]
        gp = pred[mask]
        error = gp - gy
        sse = float(np.square(error).sum())
        sst = float(np.square(gy - gy.mean()).sum())
        within_sse += sse
        within_sst += sst
        if sst > 1e-12:
            group_r2.append(1.0 - sse / sst)
    result['r2_within_battery'] = float(1.0 - within_sse / within_sst) if within_sst > 1e-12 else float('nan')
    result['r2_battery_macro'] = float(np.mean(group_r2)) if group_r2 else float('nan')
    result['r2_battery_valid'] = float(len(group_r2))
    result['r2_battery_total'] = float(len(pd_unique(groups)))
    result['sst_within_battery'] = float(within_sst)
    result['sse'] = float(np.square(pred - y).sum())
    return result

def pd_unique(values: np.ndarray) -> np.ndarray:
    """Return stable unique values without making pandas a metric dependency."""
    return np.unique(values)
