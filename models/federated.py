"""ACfC-PINN 的本地训练、FedAvg 和材料感知联邦聚合。"""
import dataclasses
from collections import defaultdict
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from configs.schema import 实验配置, 特征列
from models.losses import ACfC数据损失, 序列ACfC数据损失
from models.encoder import 按配置创建模型

def _创建配置模型(配置: 实验配置, device):
    return 按配置创建模型(配置).to(device)

def _状态(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def _加载(model, state):
    if any((str(k).endswith('退化尺度参数') for k in state)) and hasattr(model, '启用可学习尺度'):
        if not getattr(model, '可学习退化尺度', False):
            model.启用可学习尺度()
    model.load_state_dict(state, strict=True)

def _创建优化器(model, 配置: 实验配置):
    """纯 ACfC 优化器；所有可训练模块使用同一学习率。"""
    return torch.optim.AdamW(model.parameters(), lr=配置.学习率, weight_decay=0.0001)

def _单调参数(配置: 实验配置) -> dict:
    """单调约束参数：C 方案（容差 + 软阈值）与迁移项 2（单调形式）。"""
    return {'单调容差': float(配置.额外参数.get('单调容差', 0.0005)), '单调软阈值': 配置.额外参数.get('单调软阈值', None), '单调形式': str(配置.额外参数.get('单调形式', 'uncond')), '标签死区': float(配置.额外参数.get('标签死区', 0.0))}

def _权重表(配置: 实验配置, 名称: str) -> dict:
    """取出材料门控权重表；既支持 dataclass 字段，也支持放在 额外参数 里。"""
    表 = getattr(配置, 名称, None)
    if not isinstance(表, dict) or not 表:
        表 = 配置.额外参数.get(名称)
    return dict(表) if isinstance(表, dict) else {}

def _查权重表(表: dict, material, source=None):
    """按 域键（材料|来源）→ 材料键 → 通配键 "*" 的顺序查找，找不到返回 None。"""
    if not 表:
        return None
    候选 = []
    if material is not None and source is not None:
        候选.append(f'{material}|{source}')
    if material is not None:
        候选.append(str(material))
    候选.append('*')
    for 键 in 候选:
        if 键 in 表:
            return float(表[键])
    return None

def _材料物理权重(配置: 实验配置, material, source=None) -> float:
    值 = _查权重表(_权重表(配置, '物理权重表'), material, source)
    return float(配置.物理损失权重 if 值 is None else 值)

def _材料门控配置(配置: 实验配置, material, source=None) -> 实验配置:
    """按客户端材料返回带覆盖权重的配置副本（放大器：材料门控物理/单调权重）。

    原项目是单一材料场景，"全局恒定权重"等价于"正确权重"；本项目是四类材料，
    全局恒定权重必然被平均掉——LFP 上 −10% 的收益与 NCM 上 +4% 的劣化抵消成 0。

    **必须用 ``dataclasses.replace`` 复制**：所有客户端循环共享同一个配置对象，
    原地改 ``配置.物理损失权重`` 会把某个材料的权重串给后续所有材料。
    """
    覆盖 = {}
    物理 = _查权重表(_权重表(配置, '物理权重表'), material, source)
    if 物理 is not None:
        覆盖['物理损失权重'] = max(0.0, 物理)
    单调 = _查权重表(_权重表(配置, '单调权重表'), material, source)
    if 单调 is not None:
        覆盖['单调损失权重'] = max(0.0, 单调)
    if not 覆盖:
        return 配置
    return dataclasses.replace(配置, **覆盖)

def _物理聚合权重(配置: 实验配置, materials: list[str]) -> list[int]:
    """共享物理网的跨材料聚合权重：只让"物理权重 > 0"的材料参与。

    若不做这一步，关闭物理的材料仍会以"不训练物理网"的陈旧状态参与等权聚合，
    把 LFP 训练出的物理网向初始值稀释。全为 0 时退回等权，避免聚合权重和为零。
    """
    if not bool(配置.额外参数.get('物理聚合仅启用材料', False)):
        return [1] * len(materials)
    权重 = [1 if _材料物理权重(配置, material) > 0 else 0 for material in materials]
    return 权重 if sum(权重) > 0 else [1] * len(materials)

def _张量常驻(data, device):
    """把 TensorDataset 的四元组一次性搬到设备，避免逐批 H2D。

    返回 (tensors, total) 或 None。
    """
    if data is None:
        return None
    tensors = [t.to(device, non_blocking=True) for t in data.tensors]
    return (tensors, tensors[0].shape[0])

def _迭代批次(tensors, total, batch_size, device, shuffle: bool):
    """在常驻设备张量上手动切批，绕开 DataLoader 与逐批 H2D 的主机开销。"""
    if shuffle:
        perm = torch.randperm(total, device=device)
    for start in range(0, total, batch_size):
        if shuffle:
            index = perm[start:start + batch_size]
            yield tuple((t[index] for t in tensors))
        else:
            stop = min(start + batch_size, total)
            yield tuple((t[start:stop] for t in tensors))

def _后期样本权重(labels: torch.Tensor, 配置: 实验配置) -> torch.Tensor | None:
    policy = 配置.额外参数.get('后期SOH加权', {})
    if not bool(policy.get('enabled', False)):
        return None
    threshold = float(policy.get('threshold', 0.9))
    maximum = max(1.0, float(policy.get('max_weight', 2.0)))
    lower = min(threshold - 1e-06, float(policy.get('lower', 0.7)))
    progress = ((threshold - labels.float()) / max(threshold - lower, 1e-06)).clamp(0.0, 1.0)
    return 1.0 + (maximum - 1.0) * progress

def _成对数据(frame: pd.DataFrame, 标准化器, 返回电池权重: bool=False):
    """构造相邻循环样本对；可选返回客户端内电池等权抽样权重。"""
    frame = frame.sort_values(['battery_id', 'cycle_index'])
    x1, x2, y1, y2, weights = ([], [], [], [], [])
    for _, group in frame.groupby('battery_id', sort=True):
        if len(group) < 2:
            continue
        values = 标准化器.变换(group)
        labels = group['soh'].to_numpy(np.float32)
        x1.append(values[:-1])
        x2.append(values[1:])
        y1.append(labels[:-1])
        y2.append(labels[1:])
        weights.extend([1.0 / (len(group) - 1)] * (len(group) - 1))
    if not x1:
        return (None, None) if 返回电池权重 else None
    dataset = TensorDataset(torch.from_numpy(np.concatenate(x1)), torch.from_numpy(np.concatenate(x2)), torch.from_numpy(np.concatenate(y1)).view(-1, 1), torch.from_numpy(np.concatenate(y2)).view(-1, 1))
    if 返回电池权重:
        return (dataset, torch.tensor(weights, dtype=torch.double))
    return dataset

def _序列数据(frame: pd.DataFrame, 标准化器, 窗口长度: int):
    """按电池切分固定长度连续循环窗口，保留 CfC 的时间递推关系。"""
    features, labels = ([], [])
    for _, group in frame.sort_values(['battery_id', 'cycle_index']).groupby('battery_id', sort=True):
        values = 标准化器.变换(group)
        target = group['soh'].to_numpy(np.float32)
        for start in range(0, len(group) - 窗口长度 + 1, 窗口长度):
            features.append(values[start:start + 窗口长度])
            labels.append(target[start:start + 窗口长度])
    if not features:
        return None
    return TensorDataset(torch.from_numpy(np.stack(features)), torch.from_numpy(np.stack(labels)).unsqueeze(-1))

def _本地采样对数(frame, 配置, 默认对数, battery_balanced):
    """Optional per-client epoch budget for matched-compute data ablations."""
    budgets = 配置.额外参数.get('客户端每本地轮采样对数')
    if budgets is None:
        return 默认对数
    if not battery_balanced:
        raise ValueError('固定采样预算要求客户端内电池等权采样')
    clients = frame['client_id'].astype(str).unique()
    if len(clients) != 1 or str(clients[0]) not in budgets:
        raise ValueError('固定采样预算必须包含当前唯一客户端')
    count = budgets[str(clients[0])]
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
        raise ValueError('客户端采样对数必须是正整数')
    return int(count)

def 本地训练(model, frame, 标准化器, 配置: 实验配置, device, 训练轮数=None, 参考状态=None):
    epochs = 配置.本地训练轮数 if 训练轮数 is None else 训练轮数
    if bool(配置.额外参数.get('Episode完整查询更新', False)):
        for parameter in model.parameters():
            parameter.requires_grad = True
    if getattr(model, '序列模式', False):
        if 配置.额外参数.get('客户端每本地轮采样对数') is not None:
            raise ValueError('固定相邻样本对预算不适用于序列模式')
        window = int(配置.额外参数.get('序列窗口长度', 16))
        dataset = _序列数据(frame, 标准化器, window)
        if dataset is None or epochs <= 0:
            return []
        model.to(device).train()
        batch_size = max(1, 配置.批大小 // window)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = _创建优化器(model, 配置)
        logs = []
        for epoch in range(epochs):
            losses = []
            for features, targets in loader:
                features, targets = (features.to(device), targets.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss, _ = 序列ACfC数据损失(model, features, targets)
                if 参考状态 is not None:
                    penalty = sum(((p - 参考状态[n].to(device)).square().sum() for n, p in model.named_parameters()))
                    loss = loss + 0.5 * 配置.FedProx系数 * penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            logs.append({'epoch': epoch + 1, 'loss': float(np.mean(losses))})
        return logs
    battery_balanced = bool(配置.额外参数.get('客户端内电池等权采样', False))
    dataset, weights = _成对数据(frame, 标准化器, 返回电池权重=True) if battery_balanced else (_成对数据(frame, 标准化器), None)
    if dataset is None or epochs <= 0:
        return []
    sample_count = _本地采样对数(frame, 配置, len(dataset), battery_balanced)
    model.to(device).train()
    optimizer = _创建优化器(model, 配置)
    常驻 = bool(配置.额外参数.get('常驻显存训练', True))
    if not 常驻:
        if weights is None:
            loader = DataLoader(dataset, batch_size=配置.批大小, shuffle=True)
        else:
            sampler = WeightedRandomSampler(weights, num_samples=sample_count, replacement=True)
            loader = DataLoader(dataset, batch_size=配置.批大小, sampler=sampler)
        logs = []
        for epoch in range(epochs):
            losses = []
            for x1, x2, y1, y2 in loader:
                x1, x2, y1, y2 = (x1.to(device), x2.to(device), y1.to(device), y2.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss, detail = ACfC数据损失(model, x1, x2, y1, y2, _后期样本权重(y1, 配置), _后期样本权重(y2, 配置), float(配置.额外参数.get('训练单调权重', 0.0)))
                if 参考状态 is not None:
                    penalty = sum(((p - 参考状态[n].to(device)).square().sum() for n, p in model.named_parameters()))
                    loss = loss + 0.5 * 配置.FedProx系数 * penalty
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            logs.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'sample_count': sample_count, 'optimizer_steps': len(losses), 'available_pairs': len(dataset)})
        return logs
    搬移 = _张量常驻(dataset, device)
    if 搬移 is None:
        return []
    常驻张量, 总数 = 搬移
    抽样权重 = weights.to(device) if weights is not None else None
    logs = []
    for epoch in range(epochs):
        losses = []
        if 抽样权重 is not None:
            索引 = torch.multinomial(抽样权重, num_samples=sample_count, replacement=True)
        else:
            索引 = None
        for start in range(0, sample_count, 配置.批大小):
            if 索引 is not None:
                批索引 = 索引[start:start + 配置.批大小]
                x1, x2, y1, y2 = (t[批索引] for t in 常驻张量)
            else:
                stop = min(start + 配置.批大小, 总数)
                x1, x2, y1, y2 = (t[start:stop] for t in 常驻张量)
            optimizer.zero_grad(set_to_none=True)
            loss, detail = ACfC数据损失(model, x1, x2, y1, y2, _后期样本权重(y1, 配置), _后期样本权重(y2, 配置), float(配置.额外参数.get('训练单调权重', 0.0)))
            if 参考状态 is not None:
                penalty = sum(((p - 参考状态[n].to(device)).square().sum() for n, p in model.named_parameters()))
                loss = loss + 0.5 * 配置.FedProx系数 * penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        logs.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'sample_count': sample_count, 'optimizer_steps': len(losses), 'available_pairs': len(dataset)})
    return logs

def _材料平衡成对数据(frame: pd.DataFrame, 标准化器):
    """构造相邻循环样本对及材料标签，用于集中式材料等权采样。"""
    frame = frame.sort_values(['battery_id', 'cycle_index'])
    x1, x2, y1, y2, materials = ([], [], [], [], [])
    for _, group in frame.groupby('battery_id', sort=True):
        if len(group) < 2:
            continue
        values = 标准化器.变换(group)
        labels = group['soh'].to_numpy(np.float32)
        count = len(group) - 1
        x1.append(values[:-1])
        x2.append(values[1:])
        y1.append(labels[:-1])
        y2.append(labels[1:])
        materials.extend([str(group['material'].iloc[0])] * count)
    if not x1:
        return (None, None)
    dataset = TensorDataset(torch.from_numpy(np.concatenate(x1)), torch.from_numpy(np.concatenate(x2)), torch.from_numpy(np.concatenate(y1)).view(-1, 1), torch.from_numpy(np.concatenate(y2)).view(-1, 1))
    counts = pd.Series(materials).value_counts()
    weights = torch.tensor([1.0 / float(counts[material]) for material in materials], dtype=torch.double)
    return (dataset, weights)

def 材料平衡集中式训练(model, frame, 标准化器, 配置: 实验配置, device, 训练轮数: int):
    """集中式参考：全量 source 可见，但每个训练轮对三种材料等权抽样。

    注意：材料门控权重（``物理权重表``/``单调权重表``）只作用于按客户端划分的联邦路径
    （训练联邦 / 训练材料平衡共享编码器 / 训练材料平衡FedAdam / 训练材料来源共享编码器），
    因为只有那里才有"每个本地更新属于哪一类材料"的边界。集中式视角下单个模型同时看
    多类材料，逐样本门控需要改损失签名，暂不支持。
    """
    dataset, weights = _材料平衡成对数据(frame, 标准化器)
    if dataset is None or 训练轮数 <= 0:
        return []
    model.to(device).train()
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(dataset, batch_size=配置.批大小, sampler=sampler)
    optimizer = _创建优化器(model, 配置)
    logs = []
    for epoch in range(训练轮数):
        losses = []
        for x1, x2, y1, y2 in loader:
            x1, x2, y1, y2 = (x1.to(device), x2.to(device), y1.to(device), y2.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss, _ = ACfC数据损失(model, x1, x2, y1, y2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        logs.append({'epoch': epoch + 1, 'loss': float(np.mean(losses))})
    return logs

def 仅训练预测头(model, frame, 标准化器, 配置: 实验配置, device, 训练轮数: int):
    """冻结 ACfC 编码器，只使用支持样本更新预测头，避免 few-shot 灾难性遗忘。"""
    if frame.empty or 训练轮数 <= 0:
        return []
    module_name = str(配置.额外参数.get('few_shot适配模块', 'predictor'))
    prefixes = ['solution.protocol_adapter.'] if module_name == 'adapter' else ['solution.predictor.'] if module_name == 'predictor' else ['solution.predictor.', 'solution.protocol_adapter.']
    trainable = [parameter for name, parameter in model.named_parameters() if any((name.startswith(prefix) for prefix in prefixes))]
    if not trainable:
        raise ValueError(f'模型没有可用于 few-shot 适配的模块：{module_name}')
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any((name.startswith(prefix) for prefix in prefixes))
    model.to(device)
    model.solution.encoder.eval()
    if model.solution.protocol_adapter is not None:
        model.solution.protocol_adapter.train(module_name in {'adapter', 'adapter_predictor'})
    model.solution.predictor.train(module_name in {'predictor', 'adapter_predictor'})
    x = torch.from_numpy(标准化器.变换(frame))
    y = torch.from_numpy(frame['soh'].to_numpy(np.float32)).view(-1, 1)
    if getattr(model, '序列模式', False):
        loader = [(x.unsqueeze(0), y.unsqueeze(0))]
    else:
        loader = DataLoader(TensorDataset(x, y), batch_size=min(配置.批大小, len(frame)), shuffle=True)
    optimizer = torch.optim.AdamW(trainable, lr=配置.目标学习率, weight_decay=0.0001)
    logs = []
    for epoch in range(训练轮数):
        losses = []
        for features, targets in loader:
            features, targets = (features.to(device), targets.to(device))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features)
            loss = torch.nn.functional.huber_loss(prediction, targets, delta=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        logs.append({'epoch': epoch + 1, 'loss': float(np.mean(losses))})
    return logs

def 聚合(states, weights, noise=0.0, reference_state=None, clip_norm=None, secure=True):
    """聚合客户端更新。

    ``secure=True`` 在单进程原型中用成对随机掩码模拟安全聚合：掩码在求和前
    加到不同客户端、在求和时抵消，服务器只使用掩码后的贡献。真实部署时应将
    此接口替换为由客户端/服务器通信实现的安全多方求和协议。
    ``reference_state`` 用于按客户端更新做 L2 裁剪；噪声按裁剪范数缩放。
    """
    if not states or len(states) != len(weights):
        raise ValueError('states 和 weights 必须非空且长度一致')
    if not any(states):
        return {}
    total = float(sum(weights))
    if total <= 0:
        raise ValueError('聚合权重之和必须为正')
    names = tuple(states[0])
    contributions = [{name: states[i][name].float() * (float(weights[i]) / total) for name in names} for i in range(len(states))]
    if reference_state is not None and clip_norm is not None and (clip_norm > 0):
        for contribution, state, weight in zip(contributions, states, weights):
            delta = torch.cat([(state[name].float() - reference_state[name].float()).reshape(-1) for name in names])
            norm = torch.linalg.vector_norm(delta)
            scale = min(1.0, float(clip_norm) / max(float(norm), 1e-12))
            for name in names:
                contribution[name] = (state[name].float() - reference_state[name].float()) * (float(weight) / total) * scale
    if secure and len(contributions) > 1:
        for left in range(len(contributions) - 1):
            for right in range(left + 1, len(contributions)):
                for name in names:
                    mask = torch.randn_like(contributions[left][name])
                    contributions[left][name] += mask
                    contributions[right][name] -= mask
    result = {}
    for name in names:
        value = sum((item[name] for item in contributions))
        if reference_state is not None and clip_norm is not None and (clip_norm > 0):
            value = reference_state[name].float() + value
        if noise > 0 and value.is_floating_point():
            sensitivity = float(clip_norm) if clip_norm is not None and clip_norm > 0 else 1.0
            sensitivity *= max((float(w) for w in weights)) / total
            value = value + torch.randn_like(value) * (float(noise) * sensitivity)
        result[name] = value
    return result

def 训练联邦(source, 标准化器, 配置, device, 材料感知=False, 使用FedProx=False, round_callback=None):
    base = _创建配置模型(配置, device)
    global_state = _状态(base)
    material_states, logs = ({}, [])
    for round_id in range(配置.联邦轮数):
        all_states, all_weights = ([], [])
        grouped = defaultdict(list)
        for client_id, client in source.groupby('client_id', sort=True):
            grouped[str(client['material'].iloc[0])].append((str(client_id), client))
        for material, clients in grouped.items():
            mat_states, mat_weights = ([], [])
            material_reference = material_states.get(material, global_state) if 材料感知 else global_state
            材料配置 = _材料门控配置(配置, material)
            for client_id, client in clients:
                model = _创建配置模型(材料配置, device)
                start = material_reference
                _加载(model, start)
                local = 本地训练(model, client, 标准化器, 材料配置, device, 参考状态=start if 使用FedProx else None)
                state = _状态(model)
                count = max(1, client['battery_id'].nunique()) if bool(配置.额外参数.get('客户端内电池等权采样', False)) else max(1, len(client))
                all_states.append(state)
                all_weights.append(count)
                mat_states.append(state)
                mat_weights.append(count)
                if local:
                    logs.append({'round': round_id + 1, 'client_id': client_id, 'loss': local[-1]['loss']})
            if 材料感知 and mat_states:
                material_states[material] = 聚合(mat_states, mat_weights, 配置.隐私噪声系数, reference_state=material_reference, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        if all_states:
            global_state = 聚合(all_states, all_weights, 配置.隐私噪声系数, reference_state=None if 材料感知 else global_state, clip_norm=None if 材料感知 else 配置.隐私裁剪范数, secure=配置.启用安全聚合)
        if round_callback is not None:
            round_callback(round_id + 1, global_state, material_states)
    return (global_state, material_states, logs)

def _构造情景(frame: pd.DataFrame, support_cycles: int, query_min_cycles: int):
    """Split each source battery into an early support and later query set."""
    support_parts, query_parts = ([], [])
    for _, battery in frame.sort_values(['battery_id', 'cycle_index']).groupby('battery_id', sort=True):
        if len(battery) < support_cycles + query_min_cycles:
            continue
        support_parts.append(battery.iloc[:support_cycles].copy())
        query_parts.append(battery.iloc[support_cycles:].copy())
    if not support_parts:
        return (frame.iloc[0:0].copy(), frame.iloc[0:0].copy())
    return (pd.concat(support_parts, ignore_index=True), pd.concat(query_parts, ignore_index=True))

def 训练情景联邦(source, 标准化器, 配置, device):
    """First-order episodic federated ACfC-PINN.

    Each client adapts a copied model on early cycles, then performs the
    ordinary ACfC-PINN update on later cycles.  Only source-domain query labels
    are used here; target-domain labels enter evaluation through the existing
    few-shot support path.
    """
    base = _创建配置模型(配置, device)
    global_state = _状态(base)
    logs = []
    support_cycles = int(配置.额外参数.get('Episode支持周期', 5))
    query_min_cycles = int(配置.额外参数.get('Episode查询最小周期', 5))
    if support_cycles < 1 or query_min_cycles < 1:
        raise ValueError('Episode支持周期和Episode查询最小周期必须大于 0')
    for round_id in range(配置.联邦轮数):
        states, weights = ([], [])
        domain_state_map = defaultdict(list)
        domain_weight_map = defaultdict(list)
        for client_id, client in source.groupby('client_id', sort=True):
            support, query = _构造情景(client, support_cycles, query_min_cycles)
            if query.empty:
                continue
            model = _创建配置模型(配置, device)
            _加载(model, global_state)
            support_config = dataclasses.replace(配置, 额外参数={**配置.额外参数, 'few_shot适配模块': 'predictor'})
            仅训练预测头(model, support, 标准化器, support_config, device, int(配置.额外参数.get('Episode支持训练轮数', 1)))
            query_config = dataclasses.replace(配置, 额外参数={**配置.额外参数, 'Episode完整查询更新': True})
            local = 本地训练(model, query, 标准化器, query_config, device, 配置.本地训练轮数)
            states.append(_状态(model))
            state = states[-1]
            weight = max(1, client['battery_id'].nunique()) if bool(配置.额外参数.get('客户端内电池等权采样', False)) else max(1, len(query))
            weights.append(weight)
            domain = f"{client['material'].iloc[0]}|{client['source'].iloc[0]}"
            domain_state_map[domain].append(state)
            domain_weight_map[domain].append(weight)
            if local:
                logs.append({'round': round_id + 1, 'client_id': str(client_id), 'support_samples': len(support), 'query_samples': len(query), 'loss': local[-1]['loss']})
        if states:
            if bool(配置.额外参数.get('情景协议等权聚合', False)):
                domain_states = []
                for domain in sorted(domain_state_map):
                    domain_states.append(聚合(domain_state_map[domain], domain_weight_map[domain], 配置.隐私噪声系数, reference_state=global_state, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合))
                global_state = 聚合(domain_states, [1] * len(domain_states), 配置.隐私噪声系数, reference_state=global_state, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
            else:
                global_state = 聚合(states, weights, 配置.隐私噪声系数, reference_state=global_state, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
    return (global_state, {}, logs)

def _子状态(state, prefix):
    return {name: value.clone() for name, value in state.items() if name.startswith(prefix)}

def _完整状态(base_state, encoder_state, predictor_state):
    state = {name: value.clone() for name, value in base_state.items()}
    state.update(encoder_state)
    state.update(predictor_state)
    return state

def 训练协议适配器联邦(source, 标准化器, 配置, device):
    """Aggregate a shared encoder/head while keeping one adapter per source protocol."""
    base_state = _状态(_创建配置模型(配置, device))
    adapter_prefix = 'solution.protocol_adapter.'
    shared_names = [name for name in base_state if not name.startswith(adapter_prefix)]
    if len(shared_names) == len(base_state):
        raise ValueError('协议适配器联邦训练需要启用协议适配器')
    global_shared = {name: base_state[name].clone() for name in shared_names}
    zero_adapter = _子状态(base_state, adapter_prefix)
    domain_key = source['material'].astype(str) + '|' + source['source'].astype(str)
    domains = sorted(domain_key.unique())
    domain_adapters = {domain: {name: value.clone() for name, value in zero_adapter.items()} for domain in domains}
    logs = []
    for round_id in range(1, 配置.联邦轮数 + 1):
        shared_states, shared_weights = ([], [])
        next_adapters = {}
        for domain, domain_frame in source.groupby(domain_key, sort=True):
            adapter_states, adapter_weights = ([], [])
            for client_id, client in domain_frame.groupby('client_id', sort=True):
                model = _创建配置模型(配置, device)
                start = {**global_shared, **domain_adapters[str(domain)]}
                _加载(model, start)
                local = 本地训练(model, client, 标准化器, 配置, device)
                state = _状态(model)
                weight = max(1, client['battery_id'].nunique()) if bool(配置.额外参数.get('客户端内电池等权采样', False)) else max(1, len(client))
                shared_states.append({name: state[name] for name in shared_names})
                shared_weights.append(weight)
                adapter_states.append(_子状态(state, adapter_prefix))
                adapter_weights.append(weight)
                if local:
                    logs.append({'round': round_id, 'client_id': str(client_id), 'domain': str(domain), 'loss': local[-1]['loss']})
            next_adapters[str(domain)] = 聚合(adapter_states, adapter_weights, 配置.隐私噪声系数, reference_state=domain_adapters[str(domain)], clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        if shared_states:
            global_shared = 聚合(shared_states, shared_weights, 配置.隐私噪声系数, reference_state=global_shared, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        domain_adapters = next_adapters
    global_state = {**global_shared, **zero_adapter}
    domain_states = {domain: {**global_shared, **adapter} for domain, adapter in domain_adapters.items()}
    return (global_state, domain_states, logs)

def _FedAdam更新(current, target, first_moment, second_moment, step: int, learning_rate: float=0.5, beta1: float=0.9, beta2: float=0.99, epsilon: float=1e-08):
    """服务器端 FedAdam：对客户端聚合增量做自适应更新。"""
    updated, new_first, new_second = ({}, {}, {})
    for name, value in current.items():
        delta = target[name].float() - value.float()
        m = beta1 * first_moment.get(name, torch.zeros_like(delta)) + (1.0 - beta1) * delta
        v = beta2 * second_moment.get(name, torch.zeros_like(delta)) + (1.0 - beta2) * delta.square()
        m_hat = m / (1.0 - beta1 ** step)
        v_hat = v / (1.0 - beta2 ** step)
        updated[name] = value.float() + learning_rate * m_hat / (torch.sqrt(v_hat) + epsilon)
        new_first[name], new_second[name] = (m, v)
    return (updated, new_first, new_second)

def 训练材料平衡共享编码器(source, 标准化器, 配置, device):
    """分层联邦训练：共享编码器跨材料等权聚合，预测头仅在材料内聚合。"""
    base_state = _状态(_创建配置模型(配置, device))
    global_encoder = _子状态(base_state, 'solution.encoder.')
    materials = sorted(source['material'].astype(str).unique())
    material_heads = {material: _子状态(base_state, 'solution.predictor.') for material in materials}
    logs = []
    for round_id in range(配置.联邦轮数):
        material_encoder_states = {}
        for material in materials:
            clients = list(source[source['material'] == material].groupby('client_id', sort=True))
            local_encoders, local_heads, weights = ([], [], [])
            for client_id, client in clients:
                model = _创建配置模型(配置, device)
                start_state = _完整状态(base_state, global_encoder, material_heads[material])
                _加载(model, start_state)
                local_log = 本地训练(model, client, 标准化器, 配置, device)
                state = _状态(model)
                weight = max(1, client['battery_id'].nunique()) if bool(配置.额外参数.get('客户端内电池等权采样', False)) else max(1, len(client))
                local_encoders.append(_子状态(state, 'solution.encoder.'))
                local_heads.append(_子状态(state, 'solution.predictor.'))
                weights.append(weight)
                if local_log:
                    logs.append({'round': round_id + 1, 'client_id': str(client_id), 'material': material, 'loss': local_log[-1]['loss']})
            material_encoder_states[material] = 聚合(local_encoders, weights, 配置.隐私噪声系数, reference_state=global_encoder, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
            material_heads[material] = 聚合(local_heads, weights, 配置.隐私噪声系数, reference_state=material_heads[material], clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        equal_weights = [1] * len(materials)
        global_encoder = 聚合([material_encoder_states[m] for m in materials], equal_weights, 配置.隐私噪声系数, reference_state=global_encoder, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
    global_head = 聚合([material_heads[m] for m in materials], [1] * len(materials))
    global_state = _完整状态(base_state, global_encoder, global_head)
    material_states = {material: _完整状态(base_state, global_encoder, material_heads[material]) for material in materials}
    return (global_state, material_states, logs)

def 训练材料平衡FedAdam(source, 标准化器, 配置, device):
    """材料平衡分层联邦 + FedAdam 服务器更新（纯 ACfC）。"""
    base_state = _状态(_创建配置模型(配置, device))
    global_encoder = _子状态(base_state, 'solution.encoder.')
    materials = sorted(source['material'].astype(str).unique())
    material_heads = {material: _子状态(base_state, 'solution.predictor.') for material in materials}
    encoder_m, encoder_v = ({}, {})
    server_lr = float(配置.额外参数.get('服务器学习率', 0.5))
    logs = []
    for round_id in range(1, 配置.联邦轮数 + 1):
        material_encoders = {}
        for material in materials:
            clients = list(source[source['material'] == material].groupby('client_id', sort=True))
            local_encoders, local_heads, weights = ([], [], [])
            for client_id, client in clients:
                model = _创建配置模型(配置, device)
                _加载(model, _完整状态(base_state, global_encoder, material_heads[material]))
                local_log = 本地训练(model, client, 标准化器, 配置, device)
                weight = max(1, client['battery_id'].nunique()) if bool(配置.额外参数.get('客户端内电池等权采样', False)) else max(1, len(client))
                state, weight = (_状态(model), weight)
                local_encoders.append(_子状态(state, 'solution.encoder.'))
                local_heads.append(_子状态(state, 'solution.predictor.'))
                weights.append(weight)
                if local_log:
                    logs.append({'round': round_id, 'client_id': str(client_id), 'material': material, 'loss': local_log[-1]['loss']})
            material_encoders[material] = 聚合(local_encoders, weights, 配置.隐私噪声系数, reference_state=global_encoder, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
            material_heads[material] = 聚合(local_heads, weights, 配置.隐私噪声系数, reference_state=material_heads[material], clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        target_encoder = 聚合([material_encoders[m] for m in materials], [1] * len(materials), reference_state=global_encoder, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        global_encoder, encoder_m, encoder_v = _FedAdam更新(global_encoder, target_encoder, encoder_m, encoder_v, round_id, server_lr)
    global_head = 聚合([material_heads[m] for m in materials], [1] * len(materials))
    global_state = _完整状态(base_state, global_encoder, global_head)
    material_states = {m: _完整状态(base_state, global_encoder, material_heads[m]) for m in materials}
    return (global_state, material_states, logs)

def 训练材料来源共享编码器(source, 标准化器, 配置, device, round_callback=None):
    """共享编码器材料等权聚合，预测头在“材料-来源”域内独立聚合。"""
    base_state = _状态(_创建配置模型(配置, device))
    global_encoder = _子状态(base_state, 'solution.encoder.')
    materials = sorted(source['material'].astype(str).unique())
    domain_key = source['material'].astype(str) + '|' + source['source'].astype(str)
    domains = sorted(domain_key.unique())
    domain_heads = {domain: _子状态(base_state, 'solution.predictor.') for domain in domains}
    material_heads = {material: _子状态(base_state, 'solution.predictor.') for material in materials}
    head_shrinkage = float(配置.额外参数.get('来源头收缩系数', 0.0))
    if not 0.0 <= head_shrinkage <= 1.0:
        raise ValueError('来源头收缩系数必须位于 [0, 1]')

    def _收缩来源头(local_head, material_head):
        """Shrink a source head toward its material-level parent head."""
        if head_shrinkage <= 0.0:
            return {name: value.clone() for name, value in local_head.items()}
        result = {}
        for name, value in local_head.items():
            parent = material_head[name]
            if value.is_floating_point():
                result[name] = ((1.0 - head_shrinkage) * value.float() + head_shrinkage * parent.float()).to(value.dtype)
            else:
                result[name] = value.clone()
        return result
    域等权共享编码器 = bool(配置.额外参数.get('域等权共享编码器', False))
    使用FedProx = bool(配置.额外参数.get('来源级FedProx', False))
    FedProx域 = set((str(item) for item in 配置.额外参数.get('来源级FedProx域', [])))
    logs = []
    for round_id in range(1, 配置.联邦轮数 + 1):
        material_encoder_states = {}
        for material in materials:
            material_frame = source[source['material'] == material]
            local_encoders, material_weights = ([], [])
            domain_encoder_states = []
            for domain, domain_frame in material_frame.groupby(material_frame['material'].astype(str) + '|' + material_frame['source'].astype(str), sort=True):
                local_heads, domain_encoders, domain_weights = ([], [], [])
                for client_id, client in domain_frame.groupby('client_id', sort=True):
                    model = _创建配置模型(配置, device)
                    start_state = _完整状态(base_state, global_encoder, domain_heads[str(domain)])
                    _加载(model, start_state)
                    启用当前域FedProx = 使用FedProx and (not FedProx域 or str(domain) in FedProx域)
                    local_log = 本地训练(model, client, 标准化器, 配置, device, 参考状态=start_state if 启用当前域FedProx else None)
                    weight = max(1, client['battery_id'].nunique()) if bool(配置.额外参数.get('客户端内电池等权采样', False)) else max(1, len(client))
                    state = _状态(model)
                    local_encoders.append(_子状态(state, 'solution.encoder.'))
                    material_weights.append(weight)
                    domain_encoders.append(_子状态(state, 'solution.encoder.'))
                    local_heads.append(_子状态(state, 'solution.predictor.'))
                    domain_weights.append(weight)
                    if local_log:
                        logs.append({'round': round_id, 'client_id': str(client_id), 'domain': str(domain), 'loss': local_log[-1]['loss'], **({'sample_count': sum((item['sample_count'] for item in local_log)), 'optimizer_steps': sum((item['optimizer_steps'] for item in local_log)), 'available_pairs': local_log[-1]['available_pairs']} if 配置.额外参数.get('客户端每本地轮采样对数') is not None else {})})
                raw_domain_head = 聚合(local_heads, domain_weights, 配置.隐私噪声系数, reference_state=domain_heads[str(domain)], clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
                domain_heads[str(domain)] = _收缩来源头(raw_domain_head, material_heads[material])
                domain_encoder_states.append(聚合(domain_encoders, domain_weights, 配置.隐私噪声系数, reference_state=global_encoder, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合))
            if head_shrinkage > 0.0:
                material_domains = [str(domain) for domain in domain_key[material_frame.index].unique()]
                material_heads[material] = 聚合([domain_heads[domain] for domain in sorted(material_domains)], [1] * len(material_domains))
            if 域等权共享编码器:
                material_encoder_states[material] = 聚合(domain_encoder_states, [1] * len(domain_encoder_states), 配置.隐私噪声系数, reference_state=global_encoder, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
            else:
                material_encoder_states[material] = 聚合(local_encoders, material_weights, 配置.隐私噪声系数, reference_state=global_encoder, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        global_encoder = 聚合([material_encoder_states[m] for m in materials], [1] * len(materials), reference_state=global_encoder, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        if round_callback is not None:
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                current_head = 聚合([domain_heads[d] for d in domains], [1] * len(domains))
                current_global = _完整状态(base_state, global_encoder, current_head)
                current_domains = {domain: _完整状态(base_state, global_encoder, domain_heads[domain]) for domain in domains}
                round_callback(round_id, current_global, current_domains)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
    global_head = 聚合([domain_heads[d] for d in domains], [1] * len(domains))
    global_state = _完整状态(base_state, global_encoder, global_head)
    domain_states = {domain: _完整状态(base_state, global_encoder, domain_heads[domain]) for domain in domains}
    return (global_state, domain_states, logs)

def 训练交叉共享策略(source, 标准化器, 配置, device, head_level: str='global', encoder_mode: str='flat', round_callback=None):
    """在同一训练协议下交叉改变共享头粒度和编码器聚合方式。

    head_level 可取 global、material 或 domain；
    encoder_mode 可取 flat、material 或 domain。编码器聚合的层级会同时用于
    全局头的聚合，从而使四格比较只改变预先登记的共享轴。
    """
    if head_level not in {'global', 'material', 'domain'}:
        raise ValueError(f'未知 head_level：{head_level}')
    if encoder_mode not in {'flat', 'material', 'domain'}:
        raise ValueError(f'未知 encoder_mode：{encoder_mode}')
    base_state = _状态(_创建配置模型(配置, device))
    global_encoder = _子状态(base_state, 'solution.encoder.')
    materials = sorted(source['material'].astype(str).unique())
    domain_values = source['material'].astype(str) + '|' + source['source'].astype(str)
    domains = sorted(domain_values.unique())

    def _head_key(material: str, domain: str) -> str:
        if head_level == 'global':
            return '__global__'
        return material if head_level == 'material' else domain
    head_states = {}
    head_keys = sorted({_head_key(str(m), str(d)) for m, d in zip(source['material'], domain_values)})
    for key in head_keys:
        head_states[key] = _子状态(base_state, 'solution.predictor.')
    logs = []

    def _aggregate_records(records, field, mode, reference):
        """Aggregate a record field with flat/material/domain hierarchy."""
        if not records:
            return reference
        if mode == 'flat':
            return 聚合([r[field] for r in records], [r['weight'] for r in records], 配置.隐私噪声系数, reference_state=reference, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        by_material = defaultdict(list)
        by_domain = defaultdict(list)
        for record in records:
            by_material[str(record['material'])].append(record)
            by_domain[str(record['domain'])].append(record)
        if mode == 'material':
            material_states = [聚合([r[field] for r in by_material[m]], [r['weight'] for r in by_material[m]], 配置.隐私噪声系数, reference_state=reference, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合) for m in sorted(by_material)]
            return 聚合(material_states, [1] * len(material_states), 配置.隐私噪声系数, reference_state=reference, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
        material_states = []
        for material in sorted(by_material):
            material_domains = sorted({str(r['domain']) for r in by_material[material]})
            domain_states = []
            for domain in material_domains:
                rows = by_domain[domain]
                domain_states.append(聚合([r[field] for r in rows], [r['weight'] for r in rows], 配置.隐私噪声系数, reference_state=reference, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合))
            material_states.append(聚合(domain_states, [1] * len(domain_states), 配置.隐私噪声系数, reference_state=reference, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合))
        return 聚合(material_states, [1] * len(material_states), 配置.隐私噪声系数, reference_state=reference, clip_norm=配置.隐私裁剪范数, secure=配置.启用安全聚合)
    for round_id in range(1, 配置.联邦轮数 + 1):
        records = []
        for domain, domain_frame in source.groupby(source['material'].astype(str) + '|' + source['source'].astype(str), sort=True):
            material = str(domain_frame['material'].iloc[0])
            for client_id, client in domain_frame.groupby('client_id', sort=True):
                key = _head_key(material, str(domain))
                start_state = _完整状态(base_state, global_encoder, head_states[key])
                model = _创建配置模型(配置, device)
                _加载(model, start_state)
                local_log = 本地训练(model, client, 标准化器, 配置, device)
                state = _状态(model)
                weight = max(1, client['battery_id'].nunique()) if bool(配置.额外参数.get('客户端内电池等权采样', False)) else max(1, len(client))
                records.append({'encoder': _子状态(state, 'solution.encoder.'), 'head': _子状态(state, 'solution.predictor.'), 'weight': weight, 'material': material, 'domain': str(domain)})
                if local_log:
                    logs.append({'round': round_id, 'client_id': str(client_id), 'domain': str(domain), 'loss': local_log[-1]['loss']})
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        global_encoder = _aggregate_records(records, 'encoder', encoder_mode, global_encoder)
        if head_level == 'global':
            head_states['__global__'] = _aggregate_records(records, 'head', encoder_mode, head_states['__global__'])
        elif head_level == 'material':
            for material in materials:
                rows = [r for r in records if str(r['material']) == material]
                head_states[material] = _aggregate_records(rows, 'head', 'domain' if encoder_mode == 'domain' else 'flat', head_states[material])
        else:
            for domain in domains:
                rows = [r for r in records if str(r['domain']) == domain]
                head_states[domain] = _aggregate_records(rows, 'head', 'flat', head_states[domain])
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if round_callback is not None:
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            try:
                round_callback(round_id, global_encoder, head_states, base_state)
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
    if head_level == 'global':
        global_head = head_states['__global__']
        states = {}
    elif head_level == 'material':
        global_head = 聚合([head_states[m] for m in materials], [1] * len(materials))
        states = {m: _完整状态(base_state, global_encoder, head_states[m]) for m in materials}
    else:
        global_head = 聚合([head_states[d] for d in domains], [1] * len(domains))
        states = {d: _完整状态(base_state, global_encoder, head_states[d]) for d in domains}
    global_state = _完整状态(base_state, global_encoder, global_head)
    return (global_state, states, logs)

def 训练集中式(source, 标准化器, 配置, device):
    model = _创建配置模型(配置, device)
    本地训练(model, source, 标准化器, 配置, device, 配置.联邦轮数 * 配置.本地训练轮数)
    return _状态(model)

def 训练集中式材料平衡(source, 标准化器, 配置, device):
    """材料等权集中式参考，仅改变抽样风险目标，不改变 ACfC-PINN 结构。"""
    model = _创建配置模型(配置, device)
    材料平衡集中式训练(model, source, 标准化器, 配置, device, 配置.联邦轮数 * 配置.本地训练轮数)
    return _状态(model)

def 加载状态(model, state, device):
    _加载(model, state)
    return model.to(device)

def 预测(model, frame, 标准化器, device, 上下文=None):
    """预测 SOH；序列模型可使用已观测的支持样本预热隐藏状态。"""
    model.eval()
    with torch.no_grad():
        if getattr(model, '序列模式', False):
            combined = pd.concat([上下文, frame], ignore_index=True) if 上下文 is not None and (not 上下文.empty) else frame
            x = torch.from_numpy(标准化器.变换(combined)).to(device).unsqueeze(0)
            prediction = model(x).cpu().numpy().reshape(-1)
            return prediction[-len(frame):]
        x = torch.from_numpy(标准化器.变换(frame)).to(device)
        return model(x).cpu().numpy().reshape(-1)

def 预测头仿射校准(model, support, 标准化器, device, 正则系数=0.1, 最小方差=1e-08, 最小样本数=4):
    """用支持集拟合冻结模型输出的尺度和偏置，不更新编码器参数。"""
    if support is None or len(support) < int(最小样本数):
        return (1.0, 0.0)
    observed = 预测(model, support, 标准化器, device)
    target = support['soh'].to_numpy(float)
    if not np.isfinite(observed).all() or float(np.var(observed)) <= float(最小方差):
        return (1.0, 0.0)
    x_mean, y_mean = (float(observed.mean()), float(target.mean()))
    centered = observed - x_mean
    denominator = float(np.dot(centered, centered) + max(0.0, 正则系数))
    slope = (float(np.dot(centered, target - y_mean)) + max(0.0, 正则系数)) / denominator
    slope = max(0.0, slope)
    intercept = y_mean - slope * x_mean
    return (slope, float(intercept))

def 支持集残差趋势校准(model, support, query, 标准化器, device, 正则系数=0.1, 阶数=1, 最大修正=0.03):
    """Fit a support-only low-dimensional residual trend for a query battery."""
    if support is None or query is None or len(support) < 4 or query.empty:
        return np.zeros(0 if query is None else len(query), dtype=float)
    degree = int(阶数)
    if degree not in (1, 2):
        raise ValueError('残差趋势校准阶数必须为 1 或 2')
    support_prediction = 预测(model, support, 标准化器, device)
    target = support['soh'].to_numpy(float)
    residual = target - support_prediction
    if not np.isfinite(residual).all():
        return np.zeros(len(query), dtype=float)
    lower = float(support['cycle_index'].min())
    scale = max(float(query['cycle_index'].max()) - lower, 1.0)
    ts = (support['cycle_index'].to_numpy(float) - lower) / scale
    tq = (query['cycle_index'].to_numpy(float) - lower) / scale
    design = np.column_stack([ts ** power for power in range(degree + 1)])
    query_design = np.column_stack([tq ** power for power in range(degree + 1)])
    penalty = np.diag([0.0] + [max(0.0, float(正则系数))] * degree)
    try:
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ residual)
    except np.linalg.LinAlgError:
        return np.zeros(len(query), dtype=float)
    correction = query_design @ coefficients
    return np.clip(correction, -abs(float(最大修正)), abs(float(最大修正)))
