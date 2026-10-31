"""参数共享策略完整对比实验

对比四种联邦学习策略：
1. Centralized（集中式上界）- 所有数据集中训练
2. FedAvg（全共享）- 编码器和预测头全部聚合
3. Material-level（材料级）- 预测头按材料分组
4. Material-Source（材料-来源级）- 预测头按材料|来源分组
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from configs.schema import 数据目录, 输出目录, 实验配置, 特征列
from dataloader.loader import 读取数据, 按电池划分训练验证
from models.federated import 训练联邦, 训练集中式, 训练材料平衡共享编码器, 训练材料来源共享编码器, 训练交叉共享策略, 加载状态, 预测, 仅训练预测头
from utils.regression import 分层回归指标, 回归指标
from models.encoder import 按配置创建模型, 创建模型
from models import encoder as model_module
from dataloader.preprocessing import 拟合材料标准化器, 拟合标准化器
from utils.runtime import 保存环境信息, 新建输出目录, 设置随机种子, 选择设备

def _评估策略(strategy_name, global_state, material_states, target, scaler, config, device, seed, shots, 域策略对所有策略生效=True):
    """评估单个策略在目标集上的性能"""
    all_true, all_pred, details, predictions = ([], [], [], [])
    for battery_id, battery in target.groupby('battery_id', sort=True):
        battery = battery.sort_values('cycle_index').reset_index(drop=True)
        fixed_query_start = config.额外参数.get('固定查询起点', None)
        query_start = shots if fixed_query_start is None else max(shots, int(fixed_query_start))
        support, query = (battery.iloc[:shots], battery.iloc[query_start:])
        if query.empty:
            continue
        model = 按配置创建模型(config).to(device)
        model.退化率尺度 = float(config.物理退化率尺度)
        material = str(battery['material'].iloc[0])
        source = str(battery['source'].iloc[0])
        原始source = str(battery['原始source'].iloc[0]) if '原始source' in battery.columns else source
        if strategy_name == 'centralized':
            state = global_state
        elif strategy_name in {'fedavg', 'fedprox', 'flat_global'}:
            state = global_state
        elif strategy_name == 'material_balanced':
            state = material_states.get(material, global_state) if material_states else global_state
        elif strategy_name in {'material_source', 'flat_domain', 'hierarchical_domain', 'clustered'}:
            state_key = f'{material}|{source}'
            state = material_states.get(state_key, global_state) if material_states else global_state
        elif strategy_name == 'hierarchical_global':
            state = global_state
        else:
            raise ValueError(f'未知策略：{strategy_name}')
        if state is not None:
            加载状态(model, state, device)
        domain_policy = config.额外参数.get('来源预测头适配策略', {})
        适配键 = f'{material}|{原始source}'
        if shots > 0 and (not support.empty):
            if strategy_name == 'material_source' or 域策略对所有策略生效:
                policy = domain_policy.get(适配键, {}).get(str(shots), {})
            else:
                policy = {}
            policy = policy or config.额外参数.get('预测头适配策略', {}).get(str(shots), {})
            old_lr = config.目标学习率
            config.目标学习率 = float(policy.get('learning_rate', config.目标学习率))
            epochs = int(policy.get('epochs', config.目标训练轮数))
            仅训练预测头(model, support, scaler, config, device, epochs)
            config.目标学习率 = old_lr
        pred = 预测(model, query, scaler, device, 上下文=support)
        raw_pred = np.asarray(pred, dtype=float).copy()
        projection_policy = domain_policy.get(适配键, {}).get(str(shots), {}) if strategy_name == 'material_source' or 域策略对所有策略生效 else {}
        use_projection = bool(projection_policy.get('monotonic_projection', config.额外参数.get('单调预测投影', False)))
        if use_projection:
            if not support.empty:
                pred = np.minimum.accumulate(np.concatenate(([float(support['soh'].iloc[-1])], pred)))[1:]
            else:
                pred = np.minimum.accumulate(pred)
        pred = np.clip(pred, config.SOH下限, config.SOH上限)
        true = query['soh'].to_numpy(float)
        metric = 回归指标(true, pred)
        details.append({'strategy': strategy_name, 'seed': seed, 'shots': shots, 'battery_id': battery_id, 'material': material, 'source': source, **metric, 'support_samples': len(support), 'support_start': 0, 'query_start': query_start, 'query_samples': len(query)})
        all_true.extend(true)
        all_pred.extend(pred)
        predictions.extend(({'strategy': strategy_name, 'seed': seed, 'shots': shots, 'battery_id': battery_id, 'material': material, 'source': source, 'cycle_index': int(row['cycle_index']), 'true_soh': float(y), 'pred_soh': float(p), 'raw_pred_soh': float(raw)} for (_, row), y, p, raw in zip(query.iterrows(), true, pred, raw_pred)))
    if not all_true:
        raise ValueError(f'策略 {strategy_name}，shots={shots} 后没有可评估的目标样本')
    pooled = 分层回归指标(np.array(all_true), np.array(all_pred), np.array([row['battery_id'] for row in details for _ in range(row['query_samples'])]))
    detail_frame = pd.DataFrame(details)
    pooled.update({'client_macro_mae': float(detail_frame['mae'].mean()), 'client_macro_mape': float(detail_frame['mape'].mean()), 'client_macro_rmse': float(detail_frame['rmse'].mean()), 'client_macro_r2': float(detail_frame['r2'].dropna().mean()) if detail_frame['r2'].notna().any() else float('nan'), 'r2_battery_valid': float(detail_frame['r2'].notna().sum()), 'r2_battery_total': float(len(detail_frame)), 'battery_count': len(detail_frame), 'query_samples': len(all_true)})
    return (pooled, details, predictions)

def _电池特征均值(frame):
    """每块电池的特征均值向量（用于聚类）。"""
    return frame.groupby('battery_id')[特征列].mean()

def _拟合聚类(source, k, 随机种子=0):
    """在源电池特征上拟合 k-means，返回（标准化器, 聚类器）。"""
    feats = _电池特征均值(source)
    sc = StandardScaler().fit(feats)
    km = KMeans(n_clusters=k, n_init=10, random_state=随机种子).fit(sc.transform(feats))
    return (sc, km)

def _簇标签(frame, sc, km):
    """对任意电池集合预测簇标签，返回 battery_id -> 标签 的 Series。"""
    feats = _电池特征均值(frame)
    return pd.Series(km.predict(sc.transform(feats)), index=feats.index).astype(str)
