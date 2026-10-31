"""项目配置：集中保存数据路径、模型超参数和实验设置。"""
from utils.paths import ROOT as PACKAGE_ROOT
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
项目目录 = PACKAGE_ROOT
_数据目录候选 = [Path(os.environ['ACFC_PINN_DATA_DIR']).expanduser() if os.environ.get('ACFC_PINN_DATA_DIR') else None, 项目目录 / 'data', 项目目录.parent / 'FL_KAN_PINN_SOH' / 'data', 项目目录.parent / 'data']
数据目录 = next((path.resolve() for path in _数据目录候选 if path is not None and path.exists()), 项目目录 / 'data')
输出目录 = 项目目录 / 'results/generated'
特征列 = ['voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness', 'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy', 'current mean', 'current std', 'current kurtosis', 'current skewness', 'CV Q', 'CV charge time', 'current slope', 'current entropy']
协议元数据列 = ['protocol_temperature', 'protocol_charge_current', 'protocol_discharge_current', 'protocol_cutoff_voltage', 'protocol_profile_code', 'protocol_metadata_known']
额定容量 = {'HUST': 1.1, 'MIT': 1.1, 'TJU_NCA': 3.5, 'TJU_NCM': 3.5, 'XJTU_NCM': 2.0}

@dataclass
class 实验配置:
    方法: tuple[str, ...] = ('centralized_acfc', 'local', 'fedavg', 'material_source_acfc')
    shots: tuple[int, ...] = (0, 5, 10, 20)
    随机种子: tuple[int, ...] = (20260903,)
    联邦轮数: int = 5
    本地训练轮数: int = 1
    目标训练轮数: int = 10
    批大小: int = 256
    学习率: float = 0.001
    目标学习率: float = 0.0005
    物理损失权重: float = 0.0
    单调损失权重: float = 0.0
    范围损失权重: float = 0.0
    FedProx系数: float = 0.0001
    隐私噪声系数: float = 0.0
    隐私裁剪范数: float = 1.0
    启用安全聚合: bool = True
    SOH下限: float = 0.0
    SOH上限: float = 1.1
    物理退化率尺度: float = 0.05
    设备: str = 'auto'
    最大文件数: int | None = None
    每文件最大行数: int | None = None
    数据划分种子: int = 20260903
    保存电池明细: bool = True
    模型类型: str = 'acfc'
    启用物理约束: bool = False
    额外参数: dict[str, object] = field(default_factory=dict)
    物理权重表: dict[str, float] = field(default_factory=dict)
    单调权重表: dict[str, float] = field(default_factory=dict)

    def 转字典(self) -> dict[str, object]:
        return asdict(self)
