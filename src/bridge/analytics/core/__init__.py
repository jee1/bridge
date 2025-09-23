"""Analytics 핵심 기능 모듈.

데이터 통합, 타입 정규화, 커넥터 어댑터 등의 핵심 기능을 제공합니다.
CA 마일스톤 3.1: 통합 데이터 분석 레이어 추가
"""

from .connector_adapter import ConnectorAdapter
from .cross_source_joiner import CrossSourceJoiner
from .data_integration import UnifiedDataFrame
from .data_unifier import DataUnifier
from .integrated_data_layer import IntegratedDataLayer
from .quality import (
    ConsistencyStats,
    MissingValueStats,
    OutlierStats,
    QualityChecker,
    QualityReport,
)
from .schema_mapper import ColumnMapping, SchemaMapping, SchemaMapper
from .statistics import CorrelationResult, DescriptiveStats, StatisticsAnalyzer
from .streaming_processor import StreamingProcessor
from .type_converter import ConversionRule, TypeConverter
from .type_normalizer import TypeNormalizer
from .visualization import (
    ChartConfig,
    ChartGenerator,
    DashboardConfig,
    DashboardGenerator,
    ReportConfig,
    ReportGenerator,
)

__all__ = [
    # 기존 모듈
    "UnifiedDataFrame",
    "TypeNormalizer",
    "ConnectorAdapter",
    "CrossSourceJoiner",
    "StatisticsAnalyzer",
    "DescriptiveStats",
    "CorrelationResult",
    "QualityChecker",
    "MissingValueStats",
    "OutlierStats",
    "ConsistencyStats",
    "QualityReport",
    "ChartGenerator",
    "DashboardGenerator",
    "ReportGenerator",
    "ChartConfig",
    "DashboardConfig",
    "ReportConfig",
    # CA 마일스톤 3.1: 통합 데이터 분석 레이어
    "DataUnifier",
    "SchemaMapper",
    "SchemaMapping",
    "ColumnMapping",
    "TypeConverter",
    "ConversionRule",
    "StreamingProcessor",
    "IntegratedDataLayer",
]
