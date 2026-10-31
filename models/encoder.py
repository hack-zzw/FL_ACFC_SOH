"""无 PINN 的 ACfC 模型：连续时间单元、注意力编码器和 SOH 预测头。"""
import torch
from torch import nn
from configs.schema import 协议元数据列

class 加权注意力层(nn.Module):
    """对隐藏状态的每个通道学习一个权重。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.权重 = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.权重)

class 固定半权重层(nn.Module):
    """保留 sigmoid(0)=0.5 的初始缩放，消融通道权重的可学习性。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5

class CfC单元(nn.Module):
    """简化的闭式连续时间单元，支持原 ACfC-PINN 的 tau 初始化。"""

    def __init__(self, input_dim: int, hidden_dim: int, dt: float=1.0, tau_init: str='softplus_zero'):
        super().__init__()
        self.输入变换 = nn.Linear(input_dim, hidden_dim)
        self.状态变换 = nn.Linear(hidden_dim, hidden_dim)
        self.dt = float(dt)
        self.tau_init = str(tau_init)
        if self.tau_init == 'original_uniform':
            self.tau = nn.Parameter(torch.empty(hidden_dim))
            nn.init.uniform_(self.tau, 0.5, 0.98)
        elif self.tau_init == 'softplus_zero':
            self.tau_raw = nn.Parameter(torch.zeros(hidden_dim))
        else:
            raise ValueError(f'未知 tau 初始化方式：{tau_init}')
        self.激活 = nn.Tanh()
        nn.init.xavier_uniform_(self.输入变换.weight)
        nn.init.xavier_uniform_(self.状态变换.weight)
        nn.init.zeros_(self.输入变换.bias)
        nn.init.zeros_(self.状态变换.bias)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, dt: float | None=None) -> torch.Tensor:
        tau = self.tau if self.tau_init == 'original_uniform' else torch.nn.functional.softplus(self.tau_raw) + 0.05
        step = self.dt if dt is None else float(dt)
        decay = torch.exp(-step / tau).unsqueeze(0)
        proposal = self.激活(self.输入变换(x) + self.状态变换(hidden))
        return decay * hidden + (1.0 - decay) * proposal

class CfC编码器(nn.Module):
    """逐样本 CfC 编码器；可选的静态可学习通道加权构成 ACfC。"""

    def __init__(self, input_dim: int, hidden_dim: int=32, embedding_dim: int=36, layers: int=2, dropout: float=0.05, cfc_dt: float=1.0, tau_init: str='softplus_zero', channel_fusion: bool=True):
        super().__init__()
        self.输入层 = nn.Linear(input_dim, hidden_dim)
        self.单元 = nn.ModuleList([CfC单元(hidden_dim, hidden_dim, dt=cfc_dt, tau_init=tau_init) for _ in range(layers)])
        fusion_mode = channel_fusion if isinstance(channel_fusion, str) else 'learned' if channel_fusion else 'identity'
        if fusion_mode not in {'learned', 'identity', 'fixed_half'}:
            raise ValueError(f'未知通道加权模式：{fusion_mode}')
        self.启用通道加权 = fusion_mode == 'learned'
        factory = {'learned': lambda: 加权注意力层(hidden_dim), 'identity': nn.Identity, 'fixed_half': 固定半权重层}[fusion_mode]
        self.注意力 = nn.ModuleList([factory() for _ in range(layers)])
        self.投影 = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.输入层.weight)
        nn.init.xavier_uniform_(self.投影.weight)
        nn.init.zeros_(self.输入层.bias)
        nn.init.zeros_(self.投影.bias)

    def _单步前向(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.输入层(x))
        h = self.dropout(h)
        hidden = torch.zeros_like(h)
        for cell, attention in zip(self.单元, self.注意力):
            hidden = attention(cell(h, hidden))
            h = self.dropout(hidden)
        return self.投影(hidden)

    def _序列前向(self, x: torch.Tensor) -> torch.Tensor:
        """沿循环维递推隐藏状态，x 的形状为 [批量, 序列长度, 特征维度]。"""
        h = self.dropout(torch.relu(self.输入层(x)))
        for cell, attention in zip(self.单元, self.注意力):
            hidden = torch.zeros_like(h[:, 0, :])
            outputs = []
            for step in range(h.shape[1]):
                hidden = attention(cell(h[:, step, :], hidden))
                outputs.append(hidden)
            h = self.dropout(torch.stack(outputs, dim=1))
        return self.投影(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self._单步前向(x)
        if x.ndim == 3:
            return self._序列前向(x)
        raise ValueError(f'CfC编码器仅接受二维或三维输入，收到形状：{tuple(x.shape)}')

class ACfCSolutionNet(nn.Module):
    """SOH 解网络。"""

    def __init__(self, input_dim: int=17, embedding_dim: int=36, cfc_dt: float=1.0, tau_init: str='softplus_zero', protocol_adapter: bool=False, adapter_hidden_dim: int=12, channel_fusion: bool=True):
        super().__init__()
        encoder_args = dict(embedding_dim=embedding_dim, cfc_dt=cfc_dt, tau_init=tau_init)
        if channel_fusion is not True:
            encoder_args['channel_fusion'] = channel_fusion
        self.encoder = CfC编码器(input_dim, **encoder_args)
        self.predictor = nn.Sequential(nn.Linear(embedding_dim, 32), nn.SiLU(), nn.Linear(32, 1))
        self.protocol_adapter = None
        if protocol_adapter:
            self.protocol_adapter = nn.Sequential(nn.Linear(embedding_dim, adapter_hidden_dim), nn.Tanh(), nn.Linear(adapter_hidden_dim, 1))
            nn.init.zeros_(self.protocol_adapter[-1].weight)
            nn.init.zeros_(self.protocol_adapter[-1].bias)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        prediction = self.predictor(embedding)
        if self.protocol_adapter is not None:
            prediction = prediction + self.protocol_adapter(embedding)
        return prediction

class ACfC(nn.Module):
    """纯数据驱动 ACfC 模型。

    模型只包含共享 ACfC 编码器、预测头和可选的协议适配器。物理残差网络、
    自动微分约束和训练期单调项均已移除；联邦训练只聚合这些数据驱动参数。
    """

    def __init__(self, input_dim: int=17, hidden_dim: int=32, 序列模式: bool=False, cfc_dt: float=1.0, tau_init: str='softplus_zero', 残差模式: str='disabled', protocol_adapter: bool=False, adapter_hidden_dim: int=12, channel_fusion: bool=True):
        super().__init__()
        self.input_dim = input_dim
        self.序列模式 = 序列模式
        self.solution = ACfCSolutionNet(input_dim, cfc_dt=cfc_dt, tau_init=tau_init, protocol_adapter=protocol_adapter, adapter_hidden_dim=adapter_hidden_dim, channel_fusion=channel_fusion)

    def forward_with_residual(self, xt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """兼容旧调用方；纯 ACfC 不计算物理残差。"""
        prediction = self.solution(xt)
        return (prediction, torch.zeros_like(prediction))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.solution(x)

def 残差模式_从配置(配置) -> str:
    """兼容旧配置；新实验统一使用禁用物理分支的 ACfC。"""
    return 'disabled'

def 创建模型(模型类型: str='acfc', 输入维度: int=17, cfc_dt: float=1.0, tau_init: str='softplus_zero', 残差模式: str='disabled', protocol_adapter: bool=False, adapter_hidden_dim: int=12, channel_fusion: bool=True) -> ACfC:
    if 模型类型.lower() not in {'acfc', 'acfc-pinn', 'cfc', 'acfc_sequence', 'acfc-sequence'}:
        raise ValueError(f'本项目仅支持 ACfC 模型，收到：{模型类型}')
    return ACfC(输入维度, 序列模式=模型类型.lower() in {'acfc_sequence', 'acfc-sequence'}, cfc_dt=cfc_dt, tau_init=tau_init, 残差模式=残差模式, protocol_adapter=protocol_adapter, adapter_hidden_dim=adapter_hidden_dim, channel_fusion=channel_fusion)

def 按配置创建模型(配置) -> ACfC:
    """按实验配置构造纯 ACfC 模型。"""
    metadata = bool(配置.额外参数.get('启用协议元数据', False))
    adapter = bool(配置.额外参数.get('启用协议适配器', False))
    return 创建模型(配置.模型类型, 输入维度=17 + (len(协议元数据列) if metadata else 0), cfc_dt=float(配置.额外参数.get('CfC时间步长', 1.0)), tau_init=str(配置.额外参数.get('CfC时间常数初始化', 'softplus_zero')), 残差模式=残差模式_从配置(配置), protocol_adapter=adapter, adapter_hidden_dim=int(配置.额外参数.get('协议适配器隐藏维度', 12)), channel_fusion=配置.额外参数.get('通道加权模式', bool(配置.额外参数.get('启用通道加权', True))))
