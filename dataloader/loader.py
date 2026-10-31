"""读取循环级 CSV，并构造材料、来源和虚拟客户端信息。"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from configs.schema import 特征列, 额定容量

def _识别来源(path: Path) -> tuple[str, str, float] | None:
    parts = [item.lower() for item in path.parts]
    if 'hust data' in parts:
        return ('LFP', 'HUST', 额定容量['HUST'])
    if 'mit data' in parts:
        return ('LFP', 'MIT', 额定容量['MIT'])
    if 'xjtu data' in parts:
        return ('NCM', 'XJTU', 额定容量['XJTU_NCM'])
    if 'dataset_1_nca_battery' in parts:
        return ('NCA', 'TJU', 额定容量['TJU_NCA'])
    if 'dataset_2_ncm_battery' in parts:
        return ('NCM', 'TJU', 额定容量['TJU_NCM'])
    return None

def _电池编号(path: Path, source: str) -> str:
    stem = re.sub('[^0-9A-Za-z#-]+', '_', path.stem)
    parent = re.sub('[^0-9A-Za-z#-]+', '_', path.parent.name)
    return f'{source}_{parent}_{stem}'

def 读取数据(数据根目录: Path, 最大文件数: int | None=None, 每文件最大行数: int | None=None, 划分种子: int=20260903) -> pd.DataFrame:
    files = sorted(数据根目录.rglob('*.csv'))
    rows: list[pd.DataFrame] = []
    for path in files:
        识别结果 = _识别来源(path)
        if 识别结果 is None:
            continue
        if 最大文件数 is not None and len(rows) >= 最大文件数:
            break
        material, source, nominal = 识别结果
        frame = pd.read_csv(path, nrows=每文件最大行数)
        缺少 = [column for column in 特征列 + ['capacity'] if column not in frame.columns]
        if 缺少:
            raise ValueError(f'{path} 缺少字段：{缺少}')
        frame = frame[特征列 + ['capacity']].apply(pd.to_numeric, errors='coerce').dropna()
        if frame.empty:
            continue
        frame['material'] = material
        frame['source'] = source
        frame['battery_id'] = _电池编号(path, source)
        frame['cycle_index'] = np.arange(len(frame), dtype=np.float32)
        frame['soh'] = frame['capacity'] / nominal
        rows.append(frame)
    if not rows:
        raise FileNotFoundError(f'在 {数据根目录} 中没有找到可用的特征 CSV')
    result = pd.concat(rows, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return _构造客户端(result, 划分种子=划分种子)

def _构造客户端(frame: pd.DataFrame, 划分种子: int=20260903) -> pd.DataFrame:
    result = frame.copy()
    group_batteries = {(str(material), str(source)): sorted(group['battery_id'].unique()) for (material, source), group in result.groupby(['material', 'source'], sort=True)}
    client_map: dict[str, int] = {}
    rng = np.random.default_rng(划分种子)
    for (material, source), batteries in group_batteries.items():
        test_count = max(1, int(round(len(batteries) * 0.2)))
        shuffled = list(np.asarray(batteries, dtype=object)[rng.permutation(len(batteries))])
        test_batteries = shuffled[:test_count]
        train_batteries = shuffled[test_count:]
        if source == 'HUST':
            train_clients = (1, 2)
        elif source == 'MIT':
            train_clients = (3, 4)
        elif source == 'XJTU':
            train_clients = (3, 4)
        else:
            train_clients = (1, 2) if material == 'NCM' else (1, 2, 3, 4)
        for index, battery in enumerate(train_batteries):
            client_map[battery] = train_clients[index % len(train_clients)]
        for battery in test_batteries:
            client_map[battery] = 5
    result['client_number'] = [client_map[battery] for battery in result['battery_id']]
    result['client_id'] = result['material'].astype(str) + '|' + result['source'].astype(str) + '|客户端' + result['client_number'].astype(str).str.zfill(2)
    result['partition'] = 'target_test'
    result.loc[result['client_number'] < 5, 'partition'] = 'source'
    return result

def 应用历史客户端身份(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate the pre-2026-10-22 client key for immutable historical replay."""
    result = frame.copy()
    result['client_id'] = result['material'].astype(str) + '_客户端' + result['client_number'].astype(str).str.zfill(2)
    return result

def 导出数据清单(frame: pd.DataFrame, output_dir: Path) -> None:
    """导出电池级和客户端级清单，作为论文实验协议的冻结记录。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    battery_manifest = frame.groupby(['battery_id', 'material', 'source', 'client_id', 'partition'], as_index=False).agg(样本数=('soh', 'size'), 起始循环=('cycle_index', 'min'), 结束循环=('cycle_index', 'max'))
    client_manifest = battery_manifest.groupby(['material', 'source', 'client_id', 'partition'], as_index=False).agg(电池数=('battery_id', 'nunique'), 样本数=('样本数', 'sum'))
    battery_manifest.to_csv(output_dir / '电池清单.csv', index=False, encoding='utf-8-sig')
    client_manifest.to_csv(output_dir / '客户端划分清单.csv', index=False, encoding='utf-8-sig')

def 按电池划分训练验证(frame: pd.DataFrame, 训练比例: float=0.85) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只对训练客户端按时间切分，目标客户端保持完整以支持 few-shot 测试。"""
    source_parts, validation_parts = ([], [])
    for _, group in frame[frame['partition'] == 'source'].groupby('battery_id', sort=True):
        group = group.sort_values('cycle_index')
        split = max(2, int(len(group) * 训练比例))
        split = min(split, len(group) - 1) if len(group) > 1 else len(group)
        source_parts.append(group.iloc[:split])
        validation_parts.append(group.iloc[split:])
    source = pd.concat(source_parts, ignore_index=True) if source_parts else frame.iloc[0:0].copy()
    validation = pd.concat(validation_parts, ignore_index=True) if validation_parts else frame.iloc[0:0].copy()
    return (source, validation)
