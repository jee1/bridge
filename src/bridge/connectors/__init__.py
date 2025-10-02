"""커넥터 패키지 초기화."""

from __future__ import annotations

from .base import BaseConnector
from .mock import MockConnector
from .registry import ConnectorNotFoundError, connector_registry

# Elasticsearch 커넥터는 의존성이 있을 때만 임포트
try:
    from .elasticsearch import ElasticsearchConnector
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    ElasticsearchConnector = None

# 기본 Mock 커넥터 등록 (테스트 및 초기 개발용)
connector_registry.register(MockConnector())

# Elasticsearch 커넥터 등록 (의존성이 있을 때만)
if ELASTICSEARCH_AVAILABLE:
    import os
    
    elasticsearch_connector = ElasticsearchConnector(
        name="elasticsearch",
        settings={
            "host": os.getenv("BRIDGE_ELASTICSEARCH_HOST", "localhost"),
            "port": int(os.getenv("BRIDGE_ELASTICSEARCH_PORT", "9200")),
            "use_ssl": os.getenv("BRIDGE_ELASTICSEARCH_USE_SSL", "false").lower() == "true",
        },
    )
    connector_registry.register(elasticsearch_connector)

__all__ = [
    "BaseConnector",
    "MockConnector",
    "ConnectorNotFoundError",
    "connector_registry",
]

if ELASTICSEARCH_AVAILABLE:
    __all__.append("ElasticsearchConnector")