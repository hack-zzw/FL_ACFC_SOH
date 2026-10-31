"""通用工具：随机种子、设备选择和运行环境记录。"""
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

def 设置随机种子(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def 选择设备(配置值: str) -> torch.device:
    if 配置值 == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('配置要求使用 CUDA，但当前环境没有可用 GPU')
        return torch.device('cuda')
    if 配置值 == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def 新建输出目录(root: Path) -> Path:
    path = root / f"实验_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def 保存环境信息(path: Path, device: torch.device) -> None:
    info = {'Python': __import__('sys').version, 'PyTorch': torch.__version__, 'CUDA可用': torch.cuda.is_available(), 'CUDA版本': torch.version.cuda, '设备': str(device), 'GPU': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

def 估算高斯隐私预算(噪声乘子: float, 裁剪范数: float, 通信轮数: int, 客户端采样率: float=1.0, delta: float=1e-05) -> dict[str, float | str | None]:
    """给出高斯机制的保守基本组合上界，供实验记录使用。

    该上界假设每轮客户端都可能参与（采样率默认为 1），不会冒充完整 RDP
    accountant；正式部署应使用服务器端采样率和实际参与记录重新计算。
    """
    if 噪声乘子 <= 0 or 裁剪范数 <= 0 or 通信轮数 <= 0:
        return {'epsilon': None, 'delta': float(delta), '说明': '未启用可报告的高斯差分隐私'}
    q = min(max(float(客户端采样率), 0.0), 1.0)
    sigma = float(噪声乘子)
    epsilon = float(通信轮数) * q * (2.0 * (2.0 * np.log(1.25 / delta)) ** 0.5) / sigma
    return {'epsilon': epsilon, 'delta': float(delta), '采样率': q, '噪声乘子': sigma, '裁剪范数': float(裁剪范数), '说明': '高斯机制基本组合保守上界；非正式 RDP 结果'}
