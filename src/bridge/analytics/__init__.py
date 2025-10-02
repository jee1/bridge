"""Bridge Analytics 모듈.

데이터 통합, 분석, 시각화 기능을 제공하는 모듈입니다.
"""

__version__ = "0.1.0"

# 메인 분석 함수들
from .analyze_data import (
    analyze_data,
    quick_analysis,
    comprehensive_analysis,
    quality_focused_analysis,
    visualization_focused_analysis,
    AnalysisConfig,
    AnalysisResult,
)

# 핵심 클래스들
from .core.data_integration import UnifiedDataFrame
from .core.integrated_data_layer import IntegratedDataLayer
from .core.statistics import StatisticsAnalyzer
from .core.visualization import ChartGenerator, DashboardGenerator, ReportGenerator
from .quality.comprehensive_metrics import ComprehensiveQualityMetrics

__all__ = [
    # 메인 분석 함수들
    "analyze_data",
    "quick_analysis", 
    "comprehensive_analysis",
    "quality_focused_analysis",
    "visualization_focused_analysis",
    "AnalysisConfig",
    "AnalysisResult",
    
    # 핵심 클래스들
    "UnifiedDataFrame",
    "IntegratedDataLayer", 
    "StatisticsAnalyzer",
    "ChartGenerator",
    "DashboardGenerator",
    "ReportGenerator",
    "ComprehensiveQualityMetrics",
]