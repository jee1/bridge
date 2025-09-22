from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from uuid import uuid4

from bridge.utils import json as bridge_json


def test_dumps_handles_decimal_and_datetime() -> None:
    payload = {
        "amount": Decimal("10.50"),
        "count": Decimal("3"),
        "created_at": datetime(2025, 1, 2, 3, 4, 5),
        "identifier": uuid4(),
        "tags": {"서울", "데이터"},
    }

    serialized = bridge_json.dumps(payload)
    data = json.loads(serialized)

    assert data["amount"] == 10.5
    assert data["count"] == 3
    assert data["created_at"] == "2025-01-02T03:04:05"
    assert sorted(data["tags"]) == ["데이터", "서울"]
    assert isinstance(data["identifier"], str)


def test_loads_accepts_bytes_with_utf8() -> None:
    payload = {"message": "안녕하세요"}
    serialized = bridge_json.dumps(payload).encode("utf-8")

    data = bridge_json.loads(serialized)
    assert data == payload
