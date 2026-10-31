"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.source_config import read_config as read_config
from utils.state import restore_scaler as restore_scaler
from models.encoder import 按配置创建模型 as 按配置创建模型
OLD = ASSET_ROOT / 'backbone'
