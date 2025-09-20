"""커넥터 관련 예외 클래스."""
from __future__ import annotations


class ConnectorError(Exception):
    """커넥터 기본 예외."""
    pass


class ConnectionError(ConnectorError):
    """연결 실패 예외."""
    pass


class QueryExecutionError(ConnectorError):
    """쿼리 실행 실패 예외."""
    pass


class MetadataError(ConnectorError):
    """메타데이터 조회 실패 예외."""
    pass


class ConfigurationError(ConnectorError):
    """설정 오류 예외."""
    pass
