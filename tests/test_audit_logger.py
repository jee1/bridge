"""감사 로그 단위 테스트."""
from pathlib import Path

from bridge.audit.logger import AUDIT_LOG_DIR


class TestAuditLogger:
    """감사 로그 테스트."""
    
    def test_audit_log_dir_creation(self):
        """감사 로그 디렉토리 생성 테스트."""
        # AUDIT_LOG_DIR이 정의되어 있는지 확인
        assert AUDIT_LOG_DIR == Path("logs/audit")
    
    def test_write_audit_event_basic(self):
        """기본 감사 이벤트 작성 테스트."""
        # write_audit_event 함수가 존재하는지 확인
        from bridge.audit.logger import write_audit_event
        assert callable(write_audit_event)
    
    def test_write_audit_event_import(self):
        """감사 로그 모듈 임포트 테스트."""
        from bridge.audit.logger import write_audit_event, AUDIT_LOG_DIR
        assert callable(write_audit_event)
        assert isinstance(AUDIT_LOG_DIR, Path)