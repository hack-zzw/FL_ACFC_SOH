"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
import numpy as np
import pandas as pd
from models.federated import 仅训练预测头 as 仅训练预测头
from utils.regression import 分层回归指标 as 分层回归指标
from models.federated import 加载状态 as 加载状态
from utils.regression import 回归指标 as 回归指标
from models.encoder import 按配置创建模型 as 按配置创建模型
from models.federated import 支持集残差趋势校准 as 支持集残差趋势校准
from models.federated import 本地训练 as 本地训练
from models.federated import 预测 as 预测
from models.federated import 预测头仿射校准 as 预测头仿射校准

def _来源级方法(method: str) -> bool:
    """Methods with one lightweight head per source/protocol domain."""
    return method.startswith(('material_source', 'protocol_source', 'global_head_protocol_adapter', 'protocol_adapter'))

def _评估(method, global_state, material_states, target, scaler, config, device, seed, shots):
    all_true, all_pred, details, predictions = ([], [], [], [])
    for battery_id, battery in target.groupby('battery_id', sort=True):
        battery = battery.sort_values('cycle_index').reset_index(drop=True)
        fixed_query_start = config.额外参数.get('固定查询起点', None)
        query_start = shots if fixed_query_start is None else max(shots, int(fixed_query_start))
        support_window = str(config.额外参数.get('支持集窗口', 'prefix'))
        if support_window == 'recent':
            support_start = max(0, query_start - shots)
            support = battery.iloc[support_start:query_start]
        elif support_window == 'prefix':
            support_start = 0
            support = battery.iloc[:shots]
        else:
            raise ValueError(f'未知支持集窗口：{support_window}')
        query = battery.iloc[query_start:]
        if query.empty:
            continue
        model = 按配置创建模型(config).to(device)
        model.退化率尺度 = float(config.物理退化率尺度)
        material = str(battery['material'].iloc[0])
        state_key = f"{material}|{battery['source'].iloc[0]}" if _来源级方法(method) else material
        state = material_states.get(state_key, global_state) if material_states else global_state
        domain_policy = config.额外参数.get('来源预测头适配策略', {})
        if state is not None:
            加载状态(model, state, device)
        if shots > 0 and (not support.empty):
            if method.startswith('material_balanced') or _来源级方法(method) or method.startswith('centralized_acfc') or method.startswith('centralized_material_balanced') or (method in {'global_head_protocol_adapter_acfc_pinn', 'protocol_adapter_acfc_pinn', 'protocol_metadata_conditioned_acfc_pinn', 'episodic_federated_acfc_pinn', 'episodic_federated_local10_acfc_pinn', 'episodic_federated_protocol_balanced_acfc_pinn', 'episodic_federated_adapter_acfc_pinn'}):
                domain_policy = config.额外参数.get('来源预测头适配策略', {})
                policy = domain_policy.get(state_key, {}).get(str(shots), {}) if _来源级方法(method) else {}
                policy = policy or config.额外参数.get('预测头适配策略', {}).get(str(shots), {})
                old_lr = config.目标学习率
                config.目标学习率 = float(policy.get('learning_rate', config.目标学习率))
                epochs = int(policy.get('epochs', config.目标训练轮数))
                config.额外参数['few_shot适配模块'] = 'adapter_predictor' if method == 'episodic_federated_adapter_acfc_pinn' else 'adapter' if method in {'global_head_protocol_adapter_acfc_pinn', 'protocol_adapter_acfc_pinn'} else 'predictor'
                仅训练预测头(model, support, scaler, config, device, epochs)
                config.目标学习率 = old_lr
            else:
                old = config.学习率
                config.学习率 = config.目标学习率
                old_physics = config.启用物理约束
                config.启用物理约束 = False
                本地训练(model, support, scaler, config, device, config.目标训练轮数)
                config.启用物理约束 = old_physics
                config.学习率 = old
        affine_policy = domain_policy.get(state_key, {}).get(str(shots), {}) if _来源级方法(method) else {}
        if bool(affine_policy.get('affine_calibration', False)):
            slope, intercept = 预测头仿射校准(model, support, scaler, device, float(affine_policy.get('affine_ridge', 0.1)), float(affine_policy.get('affine_min_variance', 1e-08)), int(affine_policy.get('affine_min_samples', 4)))
        else:
            slope, intercept = (1.0, 0.0)
        pred = 预测(model, query, scaler, device, 上下文=support)
        pred = slope * pred + intercept
        trend_policy = config.额外参数.get('来源预测头适配策略', {})
        trend_policy = trend_policy.get(state_key, {}).get(str(shots), {}) if _来源级方法(method) else {}
        if bool(trend_policy.get('residual_trend_calibration', False)):
            if bool(trend_policy.get('affine_calibration', False)):
                raise ValueError('残差趋势与仿射校准不能叠加，请分别运行消融实验')
            pred = pred + 支持集残差趋势校准(model, support, query, scaler, device, 正则系数=float(trend_policy.get('residual_trend_ridge', 0.1)), 阶数=int(trend_policy.get('residual_trend_degree', 1)), 最大修正=float(trend_policy.get('residual_trend_max_abs', 0.03)))
        raw_pred = np.asarray(pred, dtype=float).copy()
        domain_policy = config.额外参数.get('来源预测头适配策略', {})
        projection_policy = domain_policy.get(state_key, {}).get(str(shots), {}) if _来源级方法(method) else {}
        use_projection = bool(projection_policy.get('monotonic_projection', config.额外参数.get('单调预测投影', False)))
        if use_projection:
            if not support.empty:
                pred = np.minimum.accumulate(np.concatenate(([float(support['soh'].iloc[-1])], pred)))[1:]
            else:
                pred = np.minimum.accumulate(pred)
        pred = np.clip(pred, config.SOH下限, config.SOH上限)
        true = query['soh'].to_numpy(float)
        metric = 回归指标(true, pred)
        details.append({'method': method, 'seed': seed, 'shots': shots, 'battery_id': battery_id, 'material': material, 'source': query['source'].iloc[0], **metric, 'support_samples': len(support), 'query_start': query_start, 'support_start': support_start, 'query_samples': len(query)})
        all_true.extend(true)
        all_pred.extend(pred)
        predictions.extend(({'method': method, 'seed': seed, 'shots': shots, 'battery_id': battery_id, 'material': material, 'source': row['source'], 'cycle_index': int(row['cycle_index']), 'true_soh': float(y), 'pred_soh': float(p), 'raw_pred_soh': float(raw)} for (_, row), y, p, raw in zip(query.iterrows(), true, pred, raw_pred)))
    if not all_true:
        raise ValueError(f'shots={shots} 后没有可评估的目标样本')
    pooled = 分层回归指标(np.array(all_true), np.array(all_pred), np.array([row['battery_id'] for row in details for _ in range(row['query_samples'])]))
    detail_frame = pd.DataFrame(details)
    pooled.update({'client_macro_mae': float(detail_frame['mae'].mean()), 'client_macro_mape': float(detail_frame['mape'].mean()), 'client_macro_rmse': float(detail_frame['rmse'].mean()), 'client_macro_r2': float(detail_frame['r2'].dropna().mean()) if detail_frame['r2'].notna().any() else float('nan'), 'r2_battery_valid': float(detail_frame['r2'].notna().sum()), 'r2_battery_total': float(len(detail_frame)), 'battery_count': len(detail_frame), 'query_samples': len(all_true)})
    return (pooled, details, predictions)
