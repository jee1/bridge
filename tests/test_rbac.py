"""RBAC 단위 테스트."""
import pytest

from bridge.workspaces.rbac import AccessController, Role


class TestRole:
    """Role 테스트."""
    
    def test_role_creation(self):
        """역할 생성 테스트."""
        permissions = {"read", "write", "delete"}
        role = Role(name="admin", permissions=permissions)
        
        assert role.name == "admin"
        assert role.permissions == permissions
    
    def test_role_immutable(self):
        """역할 불변성 테스트."""
        permissions = frozenset({"read", "write"})
        role = Role(name="user", permissions=permissions)

        # permissions는 frozenset이므로 수정할 수 없음
        with pytest.raises(AttributeError):
            role.permissions.add("delete")


class TestAccessController:
    """AccessController 테스트."""
    
    def test_init(self):
        """컨트롤러 초기화 테스트."""
        controller = AccessController()
        assert len(controller._assignments) == 0
        assert len(controller._roles) == 0
    
    def test_register_role(self):
        """역할 등록 테스트."""
        controller = AccessController()
        role = Role(name="admin", permissions={"read", "write", "delete"})
        
        controller.register_role(role)
        assert "admin" in controller._roles
        assert controller._roles["admin"] == role
    
    def test_assign_role_to_user(self):
        """사용자에게 역할 할당 테스트."""
        controller = AccessController()
        role = Role(name="admin", permissions={"read", "write", "delete"})
        controller.register_role(role)
        
        controller.assign_role("user1", "admin")
        assert "admin" in controller._assignments["user1"]
    
    def test_assign_nonexistent_role(self):
        """존재하지 않는 역할 할당 테스트."""
        controller = AccessController()
        
        with pytest.raises(ValueError, match="Unknown role: admin"):
            controller.assign_role("user1", "admin")
    
    def test_has_permission_with_single_role(self):
        """단일 역할로 권한 확인 테스트."""
        controller = AccessController()
        role = Role(name="admin", permissions={"read", "write", "delete"})
        controller.register_role(role)
        controller.assign_role("user1", "admin")
        
        assert controller.has_permission("user1", "read") is True
        assert controller.has_permission("user1", "write") is True
        assert controller.has_permission("user1", "delete") is True
        assert controller.has_permission("user1", "execute") is False
    
    def test_has_permission_with_multiple_roles(self):
        """여러 역할로 권한 확인 테스트."""
        controller = AccessController()
        
        # 관리자 역할
        admin_role = Role(name="admin", permissions={"read", "write", "delete"})
        controller.register_role(admin_role)
        
        # 사용자 역할
        user_role = Role(name="user", permissions={"read"})
        controller.register_role(user_role)
        
        # 사용자에게 두 역할 모두 할당
        controller.assign_role("user1", "admin")
        controller.assign_role("user1", "user")
        
        # 관리자 권한 확인
        assert controller.has_permission("user1", "read") is True
        assert controller.has_permission("user1", "write") is True
        assert controller.has_permission("user1", "delete") is True
    
    def test_has_permission_nonexistent_user(self):
        """존재하지 않는 사용자 권한 확인 테스트."""
        controller = AccessController()
        
        assert controller.has_permission("nonexistent", "read") is False
    
    def test_has_permission_no_roles(self):
        """역할이 없는 사용자 권한 확인 테스트."""
        controller = AccessController()
        role = Role(name="admin", permissions={"read", "write"})
        controller.register_role(role)
        
        # 역할을 할당하지 않은 사용자
        assert controller.has_permission("user1", "read") is False
    
    def test_multiple_users_different_roles(self):
        """여러 사용자의 서로 다른 역할 테스트."""
        controller = AccessController()
        
        # 관리자 역할
        admin_role = Role(name="admin", permissions={"read", "write", "delete"})
        controller.register_role(admin_role)
        
        # 사용자 역할
        user_role = Role(name="user", permissions={"read"})
        controller.register_role(user_role)
        
        # 사용자별 역할 할당
        controller.assign_role("admin_user", "admin")
        controller.assign_role("regular_user", "user")
        
        # 관리자 사용자 권한 확인
        assert controller.has_permission("admin_user", "read") is True
        assert controller.has_permission("admin_user", "write") is True
        assert controller.has_permission("admin_user", "delete") is True
        
        # 일반 사용자 권한 확인
        assert controller.has_permission("regular_user", "read") is True
        assert controller.has_permission("regular_user", "write") is False
        assert controller.has_permission("regular_user", "delete") is False
