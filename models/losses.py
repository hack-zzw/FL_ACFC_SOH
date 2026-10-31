"""纯 ACfC 的单步与序列监督 Huber 损失。"""
import torch
from torch import nn

def ACfC数据损失(model: nn.Module, x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor, 权重1: torch.Tensor | None=None, 权重2: torch.Tensor | None=None, 单调权重: float=0.0) -> tuple[torch.Tensor, dict[str, float]]:
    """只使用相邻循环的监督 Huber 损失。"""
    u1, u2 = (model(x1), model(x2))
    loss1 = nn.functional.huber_loss(u1.reshape(-1), y1.reshape(-1), delta=0.05, reduction='none')
    loss2 = nn.functional.huber_loss(u2.reshape(-1), y2.reshape(-1), delta=0.05, reduction='none')
    if 权重1 is None:
        loss1 = loss1.mean()
    else:
        weight1 = 权重1.reshape(-1).to(loss1)
        loss1 = (loss1 * weight1).sum() / weight1.sum().clamp_min(1e-08)
    if 权重2 is None:
        loss2 = loss2.mean()
    else:
        weight2 = 权重2.reshape(-1).to(loss2)
        loss2 = (loss2 * weight2).sum() / weight2.sum().clamp_min(1e-08)
    data_loss = 0.5 * loss1 + 0.5 * loss2
    mono_loss = torch.relu(u2.reshape(-1) - u1.reshape(-1)).mean()
    data_loss = data_loss + max(0.0, float(单调权重)) * mono_loss
    value = float(data_loss.detach())
    return (data_loss, {'data_loss': value, 'physics_loss': 0.0, 'mono_loss': float(mono_loss.detach()), 'range_loss': 0.0})

def 序列ACfC数据损失(model: nn.Module, features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """只使用窗口内的监督 Huber 损失。"""
    prediction = model(features)
    data_loss = nn.functional.huber_loss(prediction.reshape(-1), targets.reshape(-1), delta=0.05)
    value = float(data_loss.detach())
    return (data_loss, {'data_loss': value, 'physics_loss': 0.0, 'mono_loss': 0.0, 'range_loss': 0.0})
