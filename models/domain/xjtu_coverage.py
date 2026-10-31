"""Selected inference definitions extracted without experiment orchestration."""
from __future__ import annotations
from utils.paths import ASSET_ROOT
from models.domain import xjtu_conditional as original

def head_from_record(record):
    head = original.Head(record['dimension'], 16).to('cuda').eval()
    head.load_state_dict(record['state'], strict=True)
    return head
