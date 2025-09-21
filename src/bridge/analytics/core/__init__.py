"""Analytics 핵심 기능 모듈.

데이터 통합, 타입 정규화, 커넥터 어댑터 등의 핵심 기능을 제공합니다.
"""

from .connector_adapter import ConnectorAdapter
from .cross_source_joiner import CrossSourceJoiner
from .data_integration import UnifiedDataFrame
from .type_normalizer import TypeNormalizer
from .statistics import StatisticsAnalyzer, DescriptiveStats, CorrelationResult

__all__ = [
    "UnifiedDataFrame",
    "TypeNormalizer",
    "ConnectorAdapter",
    "CrossSourceJoiner",
    "StatisticsAnalyzer",
    "DescriptiveStats",
    "CorrelationResult",
]
