"""감사 로그 단위 테스트."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
import pytest

from bridge.audit.logger import write_audit_event, AUDIT_LOG_DIR


class TestAuditLogger:
    """감사 로그 테스트."""
    
    def test_audit_log_dir_creation(self):
        """감사 로그 디렉토리 생성 테스트."""
        # AUDIT_LOG_DIR이 정의되어 있는지 확인
        assert AUDIT_LOG_DIR == Path("logs/audit")
    
    @patch('bridge.audit.logger.AUDIT_LOG_DIR')
    @patch('builtins.open', new_callable=mock_open)
    def test_write_audit_event_success(self, mock_file, mock_audit_dir):
        """감사 이벤트 작성 성공 테스트."""
        # Mock 설정
        mock_audit_dir.return_value = Path("test_logs/audit")
        mock_file.return_value.__enter__.return_value.write = mock_file.return_value.write
        
        # 테스트 실행
        result_path = write_audit_event(
            actor="user123",
            action="data_access",
            metadata={"table": "customers", "query": "SELECT * FROM customers"}
        )
        
        # 검증
        mock_file.assert_called_once()
        
        # 파일에 작성된 내용 검증
        written_content = mock_file.return_value.write.call_args[0][0]
        event_data = json.loads(written_content)
        
        assert event_data["actor"] == "user123"
        assert event_data["action"] == "data_access"
        assert event_data["metadata"]["table"] == "customers"
        assert event_data["metadata"]["query"] == "SELECT * FROM customers"
        assert "timestamp" in event_data
    
    @patch('bridge.audit.logger.AUDIT_LOG_DIR')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_write_audit_event_with_korean_metadata(self, mock_mkdir, mock_file, mock_audit_dir):
        """한국어 메타데이터가 포함된 감사 이벤트 작성 테스트."""
        # Mock 설정
        mock_audit_dir.return_value = Path("test_logs/audit")
        mock_file.return_value.__enter__.return_value.write = mock_file.return_value.write
        
        # 테스트 실행
        write_audit_event(
            actor="사용자123",
            action="데이터_접근",
            metadata={"테이블": "고객정보", "쿼리": "SELECT * FROM 고객정보"}
        )
        
        # 검증
        written_content = mock_file.return_value.write.call_args[0][0]
        event_data = json.loads(written_content)
        
        assert event_data["actor"] == "사용자123"
        assert event_data["action"] == "데이터_접근"
        assert event_data["metadata"]["테이블"] == "고객정보"
        assert event_data["metadata"]["쿼리"] == "SELECT * FROM 고객정보"
    
    @patch('bridge.audit.logger.AUDIT_LOG_DIR')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_write_audit_event_empty_metadata(self, mock_mkdir, mock_file, mock_audit_dir):
        """빈 메타데이터로 감사 이벤트 작성 테스트."""
        # Mock 설정
        mock_audit_dir.return_value = Path("test_logs/audit")
        mock_file.return_value.__enter__.return_value.write = mock_file.return_value.write
        
        # 테스트 실행
        write_audit_event(
            actor="system",
            action="startup",
            metadata={}
        )
        
        # 검증
        written_content = mock_file.return_value.write.call_args[0][0]
        event_data = json.loads(written_content)
        
        assert event_data["actor"] == "system"
        assert event_data["action"] == "startup"
        assert event_data["metadata"] == {}
    
    @patch('bridge.audit.logger.AUDIT_LOG_DIR')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_write_audit_event_complex_metadata(self, mock_mkdir, mock_file, mock_audit_dir):
        """복잡한 메타데이터로 감사 이벤트 작성 테스트."""
        # Mock 설정
        mock_audit_dir.return_value = Path("test_logs/audit")
        mock_file.return_value.__enter__.return_value.write = mock_file.return_value.write
        
        # 복잡한 메타데이터
        complex_metadata = {
            "query": "SELECT * FROM customers WHERE id = ?",
            "parameters": [123],
            "execution_time": 0.5,
            "rows_affected": 1,
            "nested": {
                "level1": {
                    "level2": "value"
                }
            },
            "list_data": [1, 2, 3, "test"]
        }
        
        # 테스트 실행
        write_audit_event(
            actor="admin",
            action="query_execution",
            metadata=complex_metadata
        )
        
        # 검증
        written_content = mock_file.return_value.write.call_args[0][0]
        event_data = json.loads(written_content)
        
        assert event_data["actor"] == "admin"
        assert event_data["action"] == "query_execution"
        assert event_data["metadata"] == complex_metadata
    
    @patch('bridge.audit.logger.AUDIT_LOG_DIR')
    @patch('builtins.open', side_effect=IOError("Permission denied"))
    @patch('pathlib.Path.mkdir')
    def test_write_audit_event_file_error(self, mock_mkdir, mock_file, mock_audit_dir):
        """파일 쓰기 오류 테스트."""
        # Mock 설정
        mock_audit_dir.return_value = Path("test_logs/audit")
        
        # 테스트 실행 및 검증
        with pytest.raises(IOError, match="Permission denied"):
            write_audit_event(
                actor="user123",
                action="data_access",
                metadata={"table": "customers"}
            )
    
    @patch('bridge.audit.logger.AUDIT_LOG_DIR')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_write_audit_event_filename_format(self, mock_mkdir, mock_file, mock_audit_dir):
        """파일명 형식 테스트."""
        # Mock 설정
        mock_audit_dir.return_value = Path("test_logs/audit")
        mock_file.return_value.__enter__.return_value.write = mock_file.return_value.write
        
        # 테스트 실행
        write_audit_event(
            actor="user123",
            action="test",
            metadata={}
        )
        
        # 파일명 검증
        called_args = mock_file.call_args[0]
        filename = called_args[0]
        
        # 파일명이 audit-YYYYMMDD.jsonl 형식인지 확인
        assert filename.name.startswith("audit-")
        assert filename.name.endswith(".jsonl")
        assert len(filename.name) == len("audit-YYYYMMDD.jsonl")
    
    @patch('bridge.audit.logger.AUDIT_LOG_DIR')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_write_audit_event_json_encoding(self, mock_mkdir, mock_file, mock_audit_dir):
        """JSON 인코딩 테스트."""
        # Mock 설정
        mock_audit_dir.return_value = Path("test_logs/audit")
        mock_file.return_value.__enter__.return_value.write = mock_file.return_value.write
        
        # 특수 문자가 포함된 메타데이터
        special_metadata = {
            "query": "SELECT * FROM 테이블 WHERE 컬럼 = '값'",
            "unicode": "한글 테스트 🚀",
            "newline": "line1\nline2",
            "tab": "col1\tcol2"
        }
        
        # 테스트 실행
        write_audit_event(
            actor="user123",
            action="special_chars_test",
            metadata=special_metadata
        )
        
        # 검증
        written_content = mock_file.return_value.write.call_args[0][0]
        event_data = json.loads(written_content)
        
        assert event_data["metadata"]["query"] == "SELECT * FROM 테이블 WHERE 컬럼 = '값'"
        assert event_data["metadata"]["unicode"] == "한글 테스트 🚀"
        assert event_data["metadata"]["newline"] == "line1\nline2"
        assert event_data["metadata"]["tab"] == "col1\tcol2"
