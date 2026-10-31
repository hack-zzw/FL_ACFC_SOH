"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.adaptation import _评估 as _评估
from utils.state import restore_rng as restore_rng
from utils.state import restore_scaler as restore_scaler
from models.domain.reference_head import apply_reference as apply_reference
