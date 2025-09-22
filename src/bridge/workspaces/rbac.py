"""워크스페이스 RBAC 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set


@dataclass(frozen=True)
class Role:
    name: str
    permissions: frozenset[str]


class AccessController:
    """간단한 역할 기반 접근 제어."""

    def __init__(self) -> None:
        self._assignments: Dict[str, Set[str]] = {}
        self._roles: Dict[str, Role] = {}

    def register_role(self, role: Role) -> None:
        self._roles[role.name] = role

    def assign_role(self, user_id: str, role_name: str) -> None:
        if role_name not in self._roles:
            raise ValueError(f"Unknown role: {role_name}")
        self._assignments.setdefault(user_id, set()).add(role_name)

    def has_permission(self, user_id: str, permission: str) -> bool:
        role_names = self._assignments.get(user_id, set())
        return any(permission in self._roles[name].permissions for name in role_names)
