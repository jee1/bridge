"""Bridge ML 모듈

머신러닝 모델 통합, 고급 분석 알고리즘, AI 기반 인사이트 생성,
예측 분석 및 추천 시스템을 제공합니다.

CA 마일스톤: 고급 분석 및 AI 통합
"""

from .models import (
    ModelRegistry,
    ModelVersionManager,
    InferenceEngine,
    ModelCache
)
from .algorithms import (
    TimeSeriesAnalyzer,
    AnomalyDetector,
    ClusteringAnalyzer,
    DimensionalityReducer
)

__version__ = "0.1.0"

__all__ = [
    # 모델 관리
    "ModelRegistry",
    "ModelVersionManager", 
    "InferenceEngine",
    "ModelCache",
    
    # 고급 분석 알고리즘
    "TimeSeriesAnalyzer",
    "AnomalyDetector",
    "ClusteringAnalyzer",
    "DimensionalityReducer",
]
