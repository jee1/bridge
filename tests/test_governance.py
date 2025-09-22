"""거버넌스 모듈 테스트"""

import unittest
from datetime import datetime, timedelta
from src.bridge.governance import (
    DataContract, ContractValidator, ContractRegistry,
    MetadataCatalog, SchemaRegistry, DataLineage,
    RBACManager, Role, Permission, User,
    AuditLogger, AuditEvent, AuditTrail
)
from src.bridge.governance.contracts import DataType, QualityRuleType, ColumnSchema, QualityRule
from src.bridge.governance.metadata import DataSourceType, ColumnType, DataSource, Table, Column
from src.bridge.governance.rbac import PermissionType, ResourceType
from src.bridge.governance.audit import AuditEventType, AuditSeverity


class TestDataContract(unittest.TestCase):
    """데이터 계약 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.contract = DataContract(
            id="test_contract_1",
            name="Test Contract",
            version="1.0.0",
            description="Test data contract",
            source_system="test_system"
        )
        
        # 테스트 컬럼 추가
        self.contract.add_column(ColumnSchema(
            name="id",
            data_type=DataType.INTEGER,
            nullable=False,
            description="Primary key"
        ))
        
        self.contract.add_column(ColumnSchema(
            name="name",
            data_type=DataType.STRING,
            nullable=False,
            description="Name field"
        ))
        
        # 품질 규칙 추가
        self.contract.add_quality_rule(QualityRule(
            name="not_null_name",
            rule_type=QualityRuleType.NOT_NULL,
            column="name",
            parameters={}
        ))
    
    def test_contract_creation(self):
        """계약 생성 테스트"""
        self.assertEqual(self.contract.id, "test_contract_1")
        self.assertEqual(self.contract.name, "Test Contract")
        self.assertEqual(len(self.contract.schema), 2)
        self.assertEqual(len(self.contract.quality_rules), 1)
    
    def test_contract_validation(self):
        """계약 유효성 검사 테스트"""
        validator = ContractValidator()
        result = validator.validate_contract(self.contract)
        
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
    
    def test_data_validation(self):
        """데이터 검증 테스트"""
        validator = ContractValidator()
        
        # 유효한 데이터
        valid_data = {"id": 1, "name": "Test"}
        result = validator.validate_data_against_contract(valid_data, self.contract)
        self.assertTrue(result["compliant"])
        
        # 유효하지 않은 데이터 (필수 필드 누락)
        invalid_data = {"id": 1}
        result = validator.validate_data_against_contract(invalid_data, self.contract)
        self.assertFalse(result["compliant"])
        self.assertGreater(len(result["violations"]), 0)


class TestContractRegistry(unittest.TestCase):
    """계약 레지스트리 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.registry = ContractRegistry()
        self.contract = DataContract(
            id="test_contract_1",
            name="Test Contract",
            version="1.0.0"
        )
    
    def test_register_contract(self):
        """계약 등록 테스트"""
        result = self.registry.register_contract(self.contract)
        self.assertTrue(result)
        self.assertIsNotNone(self.registry.get_contract("test_contract_1"))
    
    def test_list_contracts(self):
        """계약 목록 조회 테스트"""
        self.registry.register_contract(self.contract)
        contracts = self.registry.list_contracts()
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0].id, "test_contract_1")


class TestMetadataCatalog(unittest.TestCase):
    """메타데이터 카탈로그 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.catalog = MetadataCatalog()
        
        # 테스트 데이터 소스
        self.data_source = DataSource(
            id="test_source_1",
            name="Test Database",
            type=DataSourceType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="test_db"
        )
        
        # 테스트 테이블
        self.table = Table(
            id="test_table_1",
            name="users",
            data_source_id="test_source_1"
        )
        
        # 테스트 컬럼
        self.column = Column(
            id="test_column_1",
            name="id",
            table_id="test_table_1",
            column_type=ColumnType.INTEGER,
            primary_key=True
        )
    
    def test_register_data_source(self):
        """데이터 소스 등록 테스트"""
        result = self.catalog.register_data_source(self.data_source)
        self.assertTrue(result)
        self.assertIsNotNone(self.catalog.get_data_source("test_source_1"))
    
    def test_register_table(self):
        """테이블 등록 테스트"""
        self.catalog.register_data_source(self.data_source)
        result = self.catalog.register_table(self.table)
        self.assertTrue(result)
        self.assertIsNotNone(self.catalog.get_table("test_table_1"))
    
    def test_register_column(self):
        """컬럼 등록 테스트"""
        self.catalog.register_data_source(self.data_source)
        self.catalog.register_table(self.table)
        result = self.catalog.register_column(self.column)
        self.assertTrue(result)
        self.assertIsNotNone(self.catalog.get_column("test_column_1"))
    
    def test_get_tables_by_data_source(self):
        """데이터 소스별 테이블 조회 테스트"""
        self.catalog.register_data_source(self.data_source)
        self.catalog.register_table(self.table)
        
        tables = self.catalog.get_tables_by_data_source("test_source_1")
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].id, "test_table_1")


class TestRBACManager(unittest.TestCase):
    """RBAC 관리자 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.rbac = RBACManager()
        
        # 테스트 사용자
        self.user = User(
            id="test_user_1",
            username="testuser",
            email="test@example.com",
            full_name="Test User"
        )
    
    def test_create_user(self):
        """사용자 생성 테스트"""
        result = self.rbac.create_user(self.user)
        self.assertTrue(result)
        self.assertIsNotNone(self.rbac.get_user("test_user_1"))
    
    def test_assign_role(self):
        """역할 할당 테스트"""
        self.rbac.create_user(self.user)
        result = self.rbac.assign_role_to_user("test_user_1", "reader")
        self.assertTrue(result)
        
        user = self.rbac.get_user("test_user_1")
        self.assertIn("reader", user.roles)
    
    def test_check_permission(self):
        """권한 확인 테스트"""
        self.rbac.create_user(self.user)
        self.rbac.assign_role_to_user("test_user_1", "reader")
        
        # 읽기 권한 확인
        has_read = self.rbac.check_permission(
            "test_user_1", 
            ResourceType.SYSTEM, 
            None, 
            PermissionType.READ
        )
        self.assertTrue(has_read)
        
        # 쓰기 권한 확인 (없어야 함)
        has_write = self.rbac.check_permission(
            "test_user_1", 
            ResourceType.SYSTEM, 
            None, 
            PermissionType.WRITE
        )
        self.assertFalse(has_write)
    
    def test_create_access_token(self):
        """접근 토큰 생성 테스트"""
        self.rbac.create_user(self.user)
        token = self.rbac.create_access_token("test_user_1")
        self.assertIsNotNone(token)
        
        # 토큰 검증
        user = self.rbac.validate_access_token(token)
        self.assertIsNotNone(user)
        self.assertEqual(user.id, "test_user_1")


class TestAuditLogger(unittest.TestCase):
    """감사 로거 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.audit_logger = AuditLogger()
    
    def test_log_data_access(self):
        """데이터 접근 로깅 테스트"""
        result = self.audit_logger.log_data_access(
            user_id="test_user_1",
            username="testuser",
            resource_type="table",
            resource_id="users",
            action="read"
        )
        self.assertTrue(result)
        
        events = self.audit_logger.get_events(user_id="test_user_1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, AuditEventType.DATA_READ)
    
    def test_log_authentication(self):
        """인증 로깅 테스트"""
        result = self.audit_logger.log_authentication(
            user_id="test_user_1",
            username="testuser",
            action="login",
            success=True
        )
        self.assertTrue(result)
        
        events = self.audit_logger.get_events(user_id="test_user_1")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, AuditEventType.LOGIN)
    
    def test_get_audit_summary(self):
        """감사 로그 요약 테스트"""
        # 테스트 이벤트 생성
        self.audit_logger.log_data_access(
            user_id="test_user_1",
            username="testuser",
            resource_type="table",
            resource_id="users",
            action="read"
        )
        
        summary = self.audit_logger.get_audit_summary(days=1)
        self.assertIn("total_events", summary)
        self.assertIn("successful_events", summary)
        self.assertIn("failed_events", summary)
        self.assertEqual(summary["total_events"], 1)


class TestAuditTrail(unittest.TestCase):
    """감사 추적기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.audit_logger = AuditLogger()
        self.audit_trail = AuditTrail(self.audit_logger)
    
    def test_track_data_access(self):
        """데이터 접근 추적 테스트"""
        result = self.audit_trail.track_data_access(
            user_id="test_user_1",
            username="testuser",
            resource_type="table",
            resource_id="users",
            action="read"
        )
        self.assertTrue(result)
    
    def test_get_compliance_report(self):
        """컴플라이언스 리포트 테스트"""
        # 테스트 이벤트 생성
        self.audit_trail.track_data_access(
            user_id="test_user_1",
            username="testuser",
            resource_type="table",
            resource_id="users",
            action="read"
        )
        
        report = self.audit_trail.get_compliance_report(days=1)
        self.assertIn("total_events", report)
        self.assertIn("compliance_score", report)
        self.assertIn("recommendations", report)


class TestIntegration(unittest.TestCase):
    """통합 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.contract_registry = ContractRegistry()
        self.metadata_catalog = MetadataCatalog()
        self.rbac = RBACManager()
        self.audit_logger = AuditLogger()
        self.audit_trail = AuditTrail(self.audit_logger)
    
    def test_end_to_end_governance(self):
        """전체 거버넌스 워크플로우 테스트"""
        # 1. 사용자 생성 및 권한 설정
        user = User(
            id="test_user_1",
            username="testuser",
            email="test@example.com"
        )
        self.rbac.create_user(user)
        self.rbac.assign_role_to_user("test_user_1", "analyst")
        
        # 2. 데이터 소스 등록
        data_source = DataSource(
            id="test_source_1",
            name="Test Database",
            type=DataSourceType.POSTGRESQL
        )
        self.metadata_catalog.register_data_source(data_source)
        
        # 3. 테이블 등록
        table = Table(
            id="test_table_1",
            name="users",
            data_source_id="test_source_1"
        )
        self.metadata_catalog.register_table(table)
        
        # 4. 데이터 계약 생성
        contract = DataContract(
            id="test_contract_1",
            name="Users Contract",
            version="1.0.0"
        )
        contract.add_column(ColumnSchema(
            name="id",
            data_type=DataType.INTEGER,
            nullable=False
        ))
        self.contract_registry.register_contract(contract)
        
        # 5. 데이터 접근 추적
        self.audit_trail.track_data_access(
            user_id="test_user_1",
            username="testuser",
            resource_type="table",
            resource_id="test_table_1",
            action="read"
        )
        
        # 6. 검증
        self.assertTrue(self.rbac.check_permission(
            "test_user_1", ResourceType.ANALYTICS, None, PermissionType.READ
        ))
        
        events = self.audit_logger.get_events(user_id="test_user_1")
        self.assertEqual(len(events), 1)


if __name__ == '__main__':
    unittest.main()
