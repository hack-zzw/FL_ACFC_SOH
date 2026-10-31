"""通过 JSON 清单读取项目外的循环级电池数据集。"""
from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from configs.schema import 特征列, 协议元数据列

def _安全标识(value: object) -> str:
    text = re.sub('[^0-9A-Za-z_#-]+', '_', str(value)).strip('_')
    return text or 'unknown'

def _必填文本(entry: dict, key: str) -> str:
    value = str(entry.get(key, '')).strip()
    if not value:
        raise ValueError(f'外部数据集配置缺少非空字段：{key}')
    return value

def 解析外部数据目录(entry: dict, manifest_dir: Path) -> Path:
    """Resolve a dataset root relative to its manifest, including for registration."""
    name = _必填文本(entry, 'name')
    root_value = _必填文本(entry, 'root')
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = Path(manifest_dir) / root
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'外部数据集 {name} 的目录不存在：{root}')
    return root

def _读取单个数据集(entry: dict, manifest_dir: Path, 每文件最大行数: int | None=None) -> pd.DataFrame:
    name = _必填文本(entry, 'name')
    material = _必填文本(entry, 'material')
    if '|' in name or '|' in material:
        raise ValueError(f'外部数据集 {name} 的 name 和 material 不能包含域分隔符 |')
    root = 解析外部数据目录(entry, manifest_dir)
    pattern = str(entry.get('glob', '**/*.csv'))
    files = sorted((path for path in root.glob(pattern) if path.is_file()))
    if not files:
        raise FileNotFoundError(f'外部数据集 {name} 在 {root} 中没有匹配 {pattern} 的文件')
    feature_map = entry.get('feature_columns', {})
    if feature_map is None:
        feature_map = {}
    if not isinstance(feature_map, dict):
        raise ValueError(f'外部数据集 {name} 的 feature_columns 必须是 JSON 对象')
    actual_features = {canonical: str(feature_map.get(canonical, canonical)) for canonical in 特征列}
    metadata_map = entry.get('protocol_metadata_columns', {})
    if metadata_map is None:
        metadata_map = {}
    if not isinstance(metadata_map, dict):
        raise ValueError(f'外部数据集 {name} 的 protocol_metadata_columns 必须是 JSON 对象')
    metadata_values = entry.get('protocol_metadata', {})
    if metadata_values is None:
        metadata_values = {}
    if not isinstance(metadata_values, dict):
        raise ValueError(f'外部数据集 {name} 的 protocol_metadata 必须是 JSON 对象')
    soh_column = str(entry.get('soh_column', '')).strip()
    capacity_column = str(entry.get('capacity_column', 'capacity')).strip()
    if not soh_column and (not capacity_column):
        raise ValueError(f'外部数据集 {name} 必须配置 soh_column 或 capacity_column')
    nominal = entry.get('nominal_capacity')
    if not soh_column and (nominal is None or float(nominal) <= 0):
        raise ValueError(f'外部数据集 {name} 使用容量计算 SOH 时必须给出正 nominal_capacity')
    cycle_column = str(entry.get('cycle_column', '')).strip()
    battery_column = str(entry.get('battery_id_column', '')).strip()
    source_column = str(entry.get('source_column', '')).strip()
    delimiter = str(entry.get('delimiter', ','))
    encoding = str(entry.get('encoding', 'utf-8'))
    rows: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path, sep=delimiter, encoding=encoding, nrows=每文件最大行数)
        required = list(actual_features.values())
        required.append(soh_column or capacity_column)
        if cycle_column:
            required.append(cycle_column)
        if battery_column:
            required.append(battery_column)
        if source_column:
            required.append(source_column)
        metadata_actual = {canonical: str(metadata_map.get(canonical, canonical)) for canonical in 协议元数据列 if metadata_map.get(canonical, canonical) in frame.columns}
        missing = sorted({column for column in required if column not in frame.columns})
        if missing:
            raise ValueError(f'{path} 缺少外部数据清单声明的字段：{missing}')
        selected = pd.DataFrame({canonical: pd.to_numeric(frame[actual], errors='coerce') for canonical, actual in actual_features.items()})
        for canonical, actual in metadata_actual.items():
            selected[canonical] = pd.to_numeric(frame[actual], errors='coerce')
        for canonical, value in metadata_values.items():
            if canonical in 协议元数据列 and canonical not in selected:
                selected[canonical] = float(value)
        if (metadata_actual or metadata_values) and 'protocol_metadata_known' not in selected:
            selected['protocol_metadata_known'] = 1.0
        if soh_column:
            selected['soh'] = pd.to_numeric(frame[soh_column], errors='coerce')
            selected['capacity'] = selected['soh'] * float(nominal) if nominal is not None else np.nan
        else:
            selected['capacity'] = pd.to_numeric(frame[capacity_column], errors='coerce')
            selected['soh'] = selected['capacity'] / float(nominal)
        if battery_column:
            selected['battery_id'] = [f'{_安全标识(name)}_{_安全标识(value)}' for value in frame[battery_column]]
        else:
            relative = path.relative_to(root).with_suffix('')
            selected['battery_id'] = f'{_安全标识(name)}_{_安全标识(relative)}'
        if cycle_column:
            selected['cycle_index'] = pd.to_numeric(frame[cycle_column], errors='coerce')
        else:
            selected['cycle_index'] = selected.groupby('battery_id', sort=False).cumcount()
        selected['material'] = material
        if source_column:
            selected['source'] = frame[source_column].astype(str).map(str.strip)
            if selected['source'].eq('').any() or selected['source'].str.contains('|', regex=False).any():
                raise ValueError(f'{path} 的 {source_column} 含空协议标签或域分隔符 |')
        else:
            selected['source'] = name
        selected = selected.replace([np.inf, -np.inf], np.nan)
        selected = selected.dropna(subset=特征列 + ['soh', 'cycle_index', 'battery_id'])
        rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    if result.empty:
        raise ValueError(f'外部数据集 {name} 清洗后没有有效样本')
    result['cycle_index'] = result['cycle_index'].astype(np.float32)
    return result

def 读取外部数据清单(path: Path, 每文件最大行数: int | None=None) -> pd.DataFrame:
    """读取清单中的一个或多个数据集，并转换为项目统一字段。"""
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f'外部数据清单不存在：{manifest_path}')
    with manifest_path.open('r', encoding='utf-8') as file:
        manifest = json.load(file)
    entries = manifest.get('datasets') if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError('外部数据清单必须包含非空 datasets 数组')
    names = [_必填文本(entry, 'name') for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError('外部数据清单中的 name 必须唯一')
    frames = [_读取单个数据集(entry, manifest_path.parent, 每文件最大行数) for entry in entries]
    result = pd.concat(frames, ignore_index=True)
    duplicated = result.groupby('battery_id')['source'].nunique()
    if (duplicated > 1).any():
        raise ValueError('外部数据集生成了跨来源重复的 battery_id')
    return result.reset_index(drop=True)
