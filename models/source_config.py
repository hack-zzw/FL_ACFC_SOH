"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from pathlib import Path
from dataclasses import fields as fields
import json as json
from configs.schema import 实验配置 as 实验配置

def read_config(path: Path) -> 实验配置:
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    names = {field.name for field in fields(实验配置)}
    config = 实验配置(**{key: value for key, value in data.items() if key in names})
    config.启用物理约束 = False
    if any((getattr(config, name) != 0 for name in ('物理损失权重', '单调损失权重', '范围损失权重'))):
        raise ValueError('Frozen source requires zero training constraint weights')
    return config
