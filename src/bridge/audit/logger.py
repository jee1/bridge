"""감사 로그 헬퍼."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from bridge.utils import json as bridge_json

AUDIT_LOG_DIR = Path("logs/audit")
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def write_audit_event(actor: str, action: str, metadata: Dict[str, Any]) -> Path:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "action": action,
        "metadata": metadata,
    }
    AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    filename = AUDIT_LOG_DIR / f"audit-{datetime.now().strftime('%Y%m%d')}.jsonl"
    with filename.open("a", encoding="utf-8") as fh:
        fh.write(bridge_json.dumps(event) + "\n")
    return filename
