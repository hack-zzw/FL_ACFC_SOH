"""训练集统计量标准化，确保目标客户端不参与特征统计。"""
from dataclasses import dataclass
import re
import numpy as np
import pandas as pd
from configs.schema import 特征列, 协议元数据列

def 补充协议元数据(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric protocol metadata.

    External manifests may provide these columns directly.  When they do not,
    protocol names are parsed conservatively; unknown fields remain zero and
    ``protocol_metadata_known`` marks whether at least one physical quantity
    was recovered.  This keeps the conditioned model usable on legacy data.
    """
    result = frame.copy()
    for column in 协议元数据列:
        if column not in result.columns:
            result[column] = 0.0
    if 'source' not in result.columns:
        return result
    sources = result['source'].astype(str)

    def _number(pattern, text):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return float(match.group(1)) if match else 0.0
    temperatures = sources.map(lambda value: _number('(?<![0-9])([0-9]+(?:\\.[0-9]+)?)\\s*C', value) if 'room_temperature' not in value.lower() else 25.0)
    charge = sources.map(lambda value: _number('(?:charge|CC)[^0-9]*([0-9]+(?:\\.[0-9]+)?)\\s*A', value))
    discharge = sources.map(lambda value: _number('(?:discharge|CC)[^0-9]*([0-9]+(?:\\.[0-9]+)?)\\s*A', value))
    known = (temperatures != 0) | (charge != 0) | (discharge != 0)
    for name, values in (('protocol_temperature', temperatures), ('protocol_charge_current', charge), ('protocol_discharge_current', discharge)):
        original = pd.to_numeric(result[name], errors='coerce')
        result[name] = original.where(original.notna() & (original != 0), values).astype(np.float32)
    result['protocol_metadata_known'] = pd.to_numeric(result['protocol_metadata_known'], errors='coerce').fillna(0.0)
    result['protocol_metadata_known'] = np.maximum(result['protocol_metadata_known'].to_numpy(float), known.to_numpy(float))

    def _profile(value):
        text = value.lower()
        if 'square_wave' in text:
            return 2.0
        if 'mixed' in text:
            return 3.0
        if 'urban' in text or 'artemis' in text:
            return 4.0
        if 'cc' in text:
            return 1.0
        return 0.0
    result['protocol_profile_code'] = result['protocol_profile_code'].where(pd.to_numeric(result['protocol_profile_code'], errors='coerce').notna() & (pd.to_numeric(result['protocol_profile_code'], errors='coerce') != 0), sources.map(_profile)).astype(np.float32)
    return result

@dataclass
class 标准化器:
    均值: np.ndarray
    标准差: np.ndarray
    元数据列: tuple[str, ...] = ()

    def 变换(self, frame: pd.DataFrame) -> np.ndarray:
        work = 补充协议元数据(frame) if self.元数据列 else frame
        columns = 特征列 + list(self.元数据列) + ['cycle_index']
        values = work[columns].to_numpy(dtype=np.float32)
        return ((values - self.均值) / self.标准差).astype(np.float32)

def 拟合标准化器(frame: pd.DataFrame, 启用协议元数据: bool=False) -> 标准化器:
    元数据 = tuple(协议元数据列) if 启用协议元数据 else ()
    work = 补充协议元数据(frame) if 元数据 else frame
    columns = 特征列 + list(元数据) + ['cycle_index']
    values = work[columns].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-08] = 1.0
    return 标准化器(mean.astype(np.float32), std.astype(np.float32), 元数据)

@dataclass
class 材料标准化器:
    """按材料分别拟合统计量，减少跨材料特征尺度差异。"""
    全局: 标准化器
    分材料: dict[str, 标准化器]

    def 变换(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty or 'material' not in frame.columns:
            return self.全局.变换(frame)
        result = np.empty((len(frame), len(self.全局.元数据列) + len(特征列) + 1), dtype=np.float32)
        materials = frame['material'].astype(str).to_numpy()
        for material in pd.unique(materials):
            positions = np.flatnonzero(materials == material)
            rows = frame.iloc[positions]
            scaler = self.分材料.get(str(material), self.全局)
            result[positions, :] = scaler.变换(rows)
        return result

def 拟合材料标准化器(frame: pd.DataFrame, 启用协议元数据: bool=False) -> 材料标准化器:
    """仅使用源客户端数据，分别拟合 LFP、NCA、NCM 的特征统计量。"""
    global_scaler = 拟合标准化器(frame, 启用协议元数据=启用协议元数据)
    material_scalers = {str(material): 拟合标准化器(group, 启用协议元数据=启用协议元数据) for material, group in frame.groupby('material', sort=True)}
    return 材料标准化器(global_scaler, material_scalers)
