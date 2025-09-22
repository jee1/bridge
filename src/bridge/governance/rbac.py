"""RBAC (Role-Based Access Control) 시스템

역할 기반 접근 제어를 구현합니다.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class PermissionType(Enum):
    """권한 타입"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(Enum):
    """리소스 타입"""
    DATABASE = "database"
    TABLE = "table"
    COLUMN = "column"
    CONNECTOR = "connector"
    ANALYTICS = "analytics"
    GOVERNANCE = "governance"
    SYSTEM = "system"


@dataclass
class Permission:
    """권한 정의"""
    id: str
    name: str
    resource_type: ResourceType
    resource_id: Optional[str] = None  # None이면 해당 타입의 모든 리소스
    permission_type: PermissionType = PermissionType.READ
    conditions: Optional[Dict[str, Any]] = None  # 추가 조건
    description: Optional[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Role:
    """역할 정의"""
    id: str
    name: str
    description: Optional[str] = None
    permissions: List[str] = None  # Permission ID 목록
    is_system_role: bool = False  # 시스템 역할 여부
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class User:
    """사용자 정의"""
    id: str
    username: str
    email: str
    full_name: Optional[str] = None
    roles: List[str] = None  # Role ID 목록
    is_active: bool = True
    last_login: Optional[datetime] = None
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class AccessToken:
    """접근 토큰"""
    token: str
    user_id: str
    expires_at: datetime
    permissions: List[str] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []
        if self.created_at is None:
            self.created_at = datetime.now()

    def is_expired(self) -> bool:
        """토큰 만료 여부 확인"""
        return datetime.now() > self.expires_at


class RBACManager:
    """RBAC 관리자"""
    
    def __init__(self):
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.access_tokens: Dict[str, AccessToken] = {}
        self.logger = logging.getLogger(__name__)
        
        # 기본 시스템 역할 생성
        self._create_default_roles()
    
    def _create_default_roles(self):
        """기본 시스템 역할 생성"""
        # 관리자 역할
        admin_permissions = [
            self._create_permission("admin_all", "모든 권한", ResourceType.SYSTEM, None, PermissionType.ADMIN)
        ]
        
        admin_role = Role(
            id="admin",
            name="Administrator",
            description="시스템 관리자",
            permissions=[p.id for p in admin_permissions],
            is_system_role=True
        )
        
        # 읽기 전용 역할
        read_permissions = [
            self._create_permission("read_all", "모든 읽기 권한", ResourceType.SYSTEM, None, PermissionType.READ)
        ]
        
        read_role = Role(
            id="reader",
            name="Reader",
            description="읽기 전용 사용자",
            permissions=[p.id for p in read_permissions],
            is_system_role=True
        )
        
        # 분석가 역할
        analyst_permissions = [
            self._create_permission("analytics_read", "분석 읽기", ResourceType.ANALYTICS, None, PermissionType.READ),
            self._create_permission("analytics_execute", "분석 실행", ResourceType.ANALYTICS, None, PermissionType.EXECUTE)
        ]
        
        analyst_role = Role(
            id="analyst",
            name="Data Analyst",
            description="데이터 분석가",
            permissions=[p.id for p in analyst_permissions],
            is_system_role=True
        )
        
        # 권한 등록
        for perm in admin_permissions + read_permissions + analyst_permissions:
            self.permissions[perm.id] = perm
        
        # 역할 등록
        self.roles[admin_role.id] = admin_role
        self.roles[read_role.id] = read_role
        self.roles[analyst_role.id] = analyst_role
    
    def _create_permission(self, id: str, name: str, resource_type: ResourceType, 
                          resource_id: Optional[str], permission_type: PermissionType) -> Permission:
        """권한 생성"""
        return Permission(
            id=id,
            name=name,
            resource_type=resource_type,
            resource_id=resource_id,
            permission_type=permission_type
        )
    
    def create_permission(self, permission: Permission) -> bool:
        """권한 생성"""
        try:
            self.permissions[permission.id] = permission
            self.logger.info(f"권한 생성 완료: {permission.id}")
            return True
        except Exception as e:
            self.logger.error(f"권한 생성 실패: {e}")
            return False
    
    def create_role(self, role: Role) -> bool:
        """역할 생성"""
        try:
            # 권한 존재 여부 확인
            for perm_id in role.permissions:
                if perm_id not in self.permissions:
                    self.logger.error(f"권한이 존재하지 않습니다: {perm_id}")
                    return False
            
            self.roles[role.id] = role
            self.logger.info(f"역할 생성 완료: {role.id}")
            return True
        except Exception as e:
            self.logger.error(f"역할 생성 실패: {e}")
            return False
    
    def create_user(self, user: User) -> bool:
        """사용자 생성"""
        try:
            # 역할 존재 여부 확인
            for role_id in user.roles:
                if role_id not in self.roles:
                    self.logger.error(f"역할이 존재하지 않습니다: {role_id}")
                    return False
            
            self.users[user.id] = user
            self.logger.info(f"사용자 생성 완료: {user.id}")
            return True
        except Exception as e:
            self.logger.error(f"사용자 생성 실패: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[User]:
        """사용자 조회"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """사용자명으로 사용자 조회"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """역할 조회"""
        return self.roles.get(role_id)
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """권한 조회"""
        return self.permissions.get(permission_id)
    
    def assign_role_to_user(self, user_id: str, role_id: str) -> bool:
        """사용자에게 역할 할당"""
        user = self.get_user(user_id)
        role = self.get_role(role_id)
        
        if not user or not role:
            return False
        
        if role_id not in user.roles:
            user.roles.append(role_id)
            user.updated_at = datetime.now()
            self.logger.info(f"역할 할당 완료: {user_id} -> {role_id}")
            return True
        
        return False
    
    def remove_role_from_user(self, user_id: str, role_id: str) -> bool:
        """사용자에서 역할 제거"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        if role_id in user.roles:
            user.roles.remove(role_id)
            user.updated_at = datetime.now()
            self.logger.info(f"역할 제거 완료: {user_id} -> {role_id}")
            return True
        
        return False
    
    def check_permission(self, user_id: str, resource_type: ResourceType, 
                        resource_id: Optional[str], permission_type: PermissionType) -> bool:
        """권한 확인"""
        user = self.get_user(user_id)
        if not user or not user.is_active:
            return False
        
        # 사용자의 모든 권한 수집
        user_permissions = self._get_user_permissions(user)
        
        # 권한 확인
        for perm in user_permissions:
            if (perm.resource_type == resource_type and
                (perm.resource_id is None or perm.resource_id == resource_id) and
                (perm.permission_type == permission_type or perm.permission_type == PermissionType.ADMIN)):
                return True
        
        return False
    
    def _get_user_permissions(self, user: User) -> List[Permission]:
        """사용자의 모든 권한 수집"""
        permissions = []
        
        for role_id in user.roles:
            role = self.get_role(role_id)
            if role:
                for perm_id in role.permissions:
                    perm = self.get_permission(perm_id)
                    if perm:
                        permissions.append(perm)
        
        return permissions
    
    def create_access_token(self, user_id: str, expires_hours: int = 24) -> Optional[str]:
        """접근 토큰 생성"""
        user = self.get_user(user_id)
        if not user or not user.is_active:
            return None
        
        import uuid
        token = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=expires_hours)
        
        # 사용자 권한 수집
        user_permissions = self._get_user_permissions(user)
        permission_ids = [perm.id for perm in user_permissions]
        
        access_token = AccessToken(
            token=token,
            user_id=user_id,
            expires_at=expires_at,
            permissions=permission_ids
        )
        
        self.access_tokens[token] = access_token
        self.logger.info(f"접근 토큰 생성 완료: {user_id}")
        return token
    
    def validate_access_token(self, token: str) -> Optional[User]:
        """접근 토큰 검증"""
        access_token = self.access_tokens.get(token)
        if not access_token or access_token.is_expired():
            return None
        
        return self.get_user(access_token.user_id)
    
    def revoke_access_token(self, token: str) -> bool:
        """접근 토큰 무효화"""
        if token in self.access_tokens:
            del self.access_tokens[token]
            self.logger.info(f"접근 토큰 무효화 완료: {token}")
            return True
        return False
    
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """사용자 권한 목록 조회"""
        user = self.get_user(user_id)
        if not user:
            return []
        
        return self._get_user_permissions(user)
    
    def get_user_roles(self, user_id: str) -> List[Role]:
        """사용자 역할 목록 조회"""
        user = self.get_user(user_id)
        if not user:
            return []
        
        roles = []
        for role_id in user.roles:
            role = self.get_role(role_id)
            if role:
                roles.append(role)
        
        return roles
    
    def list_users(self) -> List[User]:
        """사용자 목록 조회"""
        return list(self.users.values())
    
    def list_roles(self) -> List[Role]:
        """역할 목록 조회"""
        return list(self.roles.values())
    
    def list_permissions(self) -> List[Permission]:
        """권한 목록 조회"""
        return list(self.permissions.values())
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """사용자 정보 업데이트"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        try:
            for key, value in updates.items():
                if hasattr(user, key) and key not in ['id', 'created_at']:
                    setattr(user, key, value)
            
            user.updated_at = datetime.now()
            self.logger.info(f"사용자 정보 업데이트 완료: {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"사용자 정보 업데이트 실패: {e}")
            return False
    
    def update_role(self, role_id: str, updates: Dict[str, Any]) -> bool:
        """역할 정보 업데이트"""
        role = self.get_role(role_id)
        if not role or role.is_system_role:
            return False
        
        try:
            for key, value in updates.items():
                if hasattr(role, key) and key not in ['id', 'created_at', 'is_system_role']:
                    setattr(role, key, value)
            
            role.updated_at = datetime.now()
            self.logger.info(f"역할 정보 업데이트 완료: {role_id}")
            return True
        except Exception as e:
            self.logger.error(f"역할 정보 업데이트 실패: {e}")
            return False
    
    def delete_user(self, user_id: str) -> bool:
        """사용자 삭제"""
        if user_id in self.users:
            del self.users[user_id]
            self.logger.info(f"사용자 삭제 완료: {user_id}")
            return True
        return False
    
    def delete_role(self, role_id: str) -> bool:
        """역할 삭제"""
        role = self.get_role(role_id)
        if not role or role.is_system_role:
            return False
        
        # 해당 역할을 사용하는 사용자가 있는지 확인
        for user in self.users.values():
            if role_id in user.roles:
                self.logger.error(f"역할을 사용하는 사용자가 있습니다: {user_id}")
                return False
        
        del self.roles[role_id]
        self.logger.info(f"역할 삭제 완료: {role_id}")
        return True
    
    def export_rbac_data(self) -> Dict[str, Any]:
        """RBAC 데이터 내보내기"""
        return {
            "permissions": [asdict(p) for p in self.permissions.values()],
            "roles": [asdict(r) for r in self.roles.values()],
            "users": [asdict(u) for u in self.users.values()],
            "exported_at": datetime.now().isoformat()
        }
    
    def import_rbac_data(self, data: Dict[str, Any]) -> bool:
        """RBAC 데이터 가져오기"""
        try:
            # 권한 가져오기
            for perm_data in data.get("permissions", []):
                perm_data['resource_type'] = ResourceType(perm_data['resource_type'])
                perm_data['permission_type'] = PermissionType(perm_data['permission_type'])
                permission = Permission(**perm_data)
                self.permissions[permission.id] = permission
            
            # 역할 가져오기
            for role_data in data.get("roles", []):
                role = Role(**role_data)
                self.roles[role.id] = role
            
            # 사용자 가져오기
            for user_data in data.get("users", []):
                user = User(**user_data)
                self.users[user.id] = user
            
            self.logger.info("RBAC 데이터 가져오기 완료")
            return True
        except Exception as e:
            self.logger.error(f"RBAC 데이터 가져오기 실패: {e}")
            return False
