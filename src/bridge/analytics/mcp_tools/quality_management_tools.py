"""품질 관리 MCP 도구.

CA 마일스톤 3.3: 데이터 품질 관리 시스템
- 종합 품질 메트릭, 고급 이상치 탐지, 데이터 정제, 품질 트렌드 분석
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from bridge.analytics.core.data_integration import UnifiedDataFrame
from bridge.analytics.quality.advanced_outlier_detection import AdvancedOutlierDetector
from bridge.analytics.quality.comprehensive_metrics import ComprehensiveQualityMetrics
from bridge.analytics.quality.data_cleaning_pipeline import DataCleaningPipeline
from bridge.analytics.quality.quality_trend_analysis import QualityTrendAnalyzer

logger = logging.getLogger(__name__)


class QualityManagementTools:
    """품질 관리 MCP 도구 클래스"""

    def __init__(self):
        """품질 관리 도구 초기화"""
        self.logger = logging.getLogger(__name__)
        self.quality_calculator = ComprehensiveQualityMetrics()
        self.outlier_detector = AdvancedOutlierDetector()
        self.cleaning_pipeline = DataCleaningPipeline()
        self.trend_analyzer = QualityTrendAnalyzer()

    def calculate_comprehensive_quality(
        self,
        data: UnifiedDataFrame,
        reference_data: Optional[UnifiedDataFrame] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """종합 품질 메트릭 계산

        Args:
            data: 분석할 데이터
            reference_data: 참조 데이터 (선택사항)
            constraints: 유효성 제약조건 (선택사항)

        Returns:
            품질 메트릭 결과
        """
        try:
            metrics = self.quality_calculator.calculate_overall_quality(
                data, reference_data, constraints
            )

            return {
                "success": True,
                "quality_metrics": {
                    "completeness": metrics.completeness,
                    "accuracy": metrics.accuracy,
                    "consistency": metrics.consistency,
                    "validity": metrics.validity,
                    "overall_score": metrics.overall_score,
                    "missing_ratio": metrics.missing_ratio,
                    "duplicate_ratio": metrics.duplicate_ratio,
                    "outlier_ratio": metrics.outlier_ratio,
                    "constraint_violations": metrics.constraint_violations,
                    "data_type_consistency": metrics.data_type_consistency,
                    "range_violations": metrics.range_violations,
                    "format_violations": metrics.format_violations,
                },
                "recommendations": self._generate_quality_recommendations(metrics),
            }

        except Exception as e:
            self.logger.error(f"종합 품질 메트릭 계산 실패: {e}")
            return {"success": False, "error": str(e), "quality_metrics": None}

    def detect_advanced_outliers(
        self,
        data: UnifiedDataFrame,
        method: str = "auto",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """고급 이상치 탐지

        Args:
            data: 분석할 데이터
            method: 탐지 방법 ("isolation_forest", "lof", "one_class_svm", "ensemble", "auto")
            parameters: 탐지 매개변수 (선택사항)

        Returns:
            이상치 탐지 결과
        """
        try:
            if parameters is None:
                parameters = {}

            if method == "isolation_forest":
                result = self.outlier_detector.detect_outliers_isolation_forest(data, **parameters)
            elif method == "lof":
                result = self.outlier_detector.detect_outliers_lof(data, **parameters)
            elif method == "one_class_svm":
                result = self.outlier_detector.detect_outliers_one_class_svm(data, **parameters)
            elif method == "ensemble":
                result = self.outlier_detector.detect_outliers_ensemble(data, **parameters)
            else:  # auto
                result = self.outlier_detector.detect_outliers_auto(data)

            # 이상치 요약 생성
            summary = self.outlier_detector.get_outlier_summary(result, data)

            return {
                "success": True,
                "outlier_detection": {
                    "method": result.method,
                    "outlier_count": len(result.outlier_indices),
                    "outlier_ratio": result.outlier_ratio,
                    "confidence": result.confidence,
                    "outlier_indices": result.outlier_indices,
                    "outlier_scores": result.outlier_scores,
                    "details": result.details,
                },
                "summary": summary,
            }

        except Exception as e:
            self.logger.error(f"고급 이상치 탐지 실패: {e}")
            return {"success": False, "error": str(e), "outlier_detection": None}

    def clean_data_pipeline(
        self, data: UnifiedDataFrame, pipeline_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """데이터 정제 파이프라인 실행

        Args:
            data: 정제할 데이터
            pipeline_config: 파이프라인 설정 (선택사항)

        Returns:
            정제 결과
        """
        try:
            # 파이프라인 설정 적용
            if pipeline_config:
                self._configure_pipeline(pipeline_config)
            else:
                # 기본 파이프라인 생성
                self.cleaning_pipeline.create_default_pipeline()

            # 데이터 정제 실행
            result = self.cleaning_pipeline.clean_data(data)

            return {
                "success": True,
                "cleaning_result": {
                    "cleaned_data_shape": result.cleaned_data.to_pandas().shape,
                    "original_data_shape": data.to_pandas().shape,
                    "removed_rows": result.removed_rows,
                    "removed_columns": result.removed_columns,
                    "transformed_columns": result.transformed_columns,
                    "quality_improvement": result.quality_improvement,
                    "cleaning_steps": result.cleaning_steps,
                },
                "pipeline_summary": self.cleaning_pipeline.get_pipeline_summary(),
            }

        except Exception as e:
            self.logger.error(f"데이터 정제 파이프라인 실행 실패: {e}")
            return {"success": False, "error": str(e), "cleaning_result": None}

    def analyze_quality_trends(
        self, data: UnifiedDataFrame, days: int = 30, add_snapshot: bool = True
    ) -> Dict[str, Any]:
        """품질 트렌드 분석

        Args:
            data: 분석할 데이터
            days: 분석할 기간 (일)
            add_snapshot: 현재 데이터를 스냅샷으로 추가할지 여부

        Returns:
            트렌드 분석 결과
        """
        try:
            # 현재 데이터를 스냅샷으로 추가
            if add_snapshot:
                self.trend_analyzer.add_quality_snapshot(data)

            # 트렌드 분석 실행
            analysis_result = self.trend_analyzer.analyze_trends(days)

            # 대시보드 데이터 생성
            dashboard_data = self.trend_analyzer.get_quality_dashboard_data()

            return {
                "success": True,
                "trend_analysis": {
                    "overall_trend": analysis_result.overall_trend,
                    "trend_strength": analysis_result.trend_strength,
                    "predictions": analysis_result.predictions,
                    "alerts": analysis_result.alerts,
                    "recommendations": analysis_result.recommendations,
                    "total_snapshots": len(analysis_result.trends),
                },
                "dashboard_data": dashboard_data,
                "trend_summary": self.trend_analyzer.get_trend_summary(),
            }

        except Exception as e:
            self.logger.error(f"품질 트렌드 분석 실패: {e}")
            return {"success": False, "error": str(e), "trend_analysis": None}

    def set_quality_thresholds(self, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """품질 임계값 설정

        Args:
            thresholds: 임계값 딕셔너리

        Returns:
            설정 결과
        """
        try:
            self.trend_analyzer.set_thresholds(thresholds)

            return {
                "success": True,
                "message": "품질 임계값이 설정되었습니다.",
                "thresholds": thresholds,
            }

        except Exception as e:
            self.logger.error(f"품질 임계값 설정 실패: {e}")
            return {"success": False, "error": str(e)}

    def _configure_pipeline(self, config: Dict[str, Any]):
        """파이프라인 설정 적용

        Args:
            config: 파이프라인 설정
        """
        try:
            # 기존 단계 제거
            self.cleaning_pipeline.cleaning_steps = []

            # 설정된 단계 추가
            if "steps" in config:
                for step_config in config["steps"]:
                    self.cleaning_pipeline.add_step(
                        name=step_config["name"],
                        function=step_config["function"],
                        parameters=step_config.get("parameters", {}),
                        enabled=step_config.get("enabled", True),
                    )
            else:
                # 기본 파이프라인 생성
                self.cleaning_pipeline.create_default_pipeline()

        except Exception as e:
            self.logger.error(f"파이프라인 설정 실패: {e}")
            # 기본 파이프라인으로 폴백
            self.cleaning_pipeline.create_default_pipeline()

    def _generate_quality_recommendations(self, metrics) -> List[str]:
        """품질 개선 권장사항 생성

        Args:
            metrics: 품질 메트릭

        Returns:
            권장사항 목록
        """
        recommendations = []

        try:
            if metrics.completeness < 0.8:
                recommendations.append(
                    "완전성을 개선하세요. 결측값 처리를 강화하고 데이터 수집 프로세스를 점검하세요."
                )

            if metrics.accuracy < 0.8:
                recommendations.append(
                    "정확성을 개선하세요. 중복 데이터를 제거하고 데이터 검증 규칙을 강화하세요."
                )

            if metrics.consistency < 0.8:
                recommendations.append(
                    "일관성을 개선하세요. 데이터 표준화 규칙을 수립하고 변환 프로세스를 점검하세요."
                )

            if metrics.validity < 0.8:
                recommendations.append(
                    "유효성을 개선하세요. 데이터 검증 규칙을 강화하고 제약조건을 점검하세요."
                )

            if metrics.missing_ratio > 0.2:
                recommendations.append(
                    f"결측값 비율이 높습니다 ({metrics.missing_ratio:.2%}). 데이터 수집 프로세스를 점검하세요."
                )

            if metrics.duplicate_ratio > 0.1:
                recommendations.append(
                    f"중복 데이터 비율이 높습니다 ({metrics.duplicate_ratio:.2%}). 중복 제거 프로세스를 강화하세요."
                )

            if metrics.outlier_ratio > 0.1:
                recommendations.append(
                    f"이상치 비율이 높습니다 ({metrics.outlier_ratio:.2%}). 데이터 품질을 점검하세요."
                )

            if not recommendations:
                recommendations.append("데이터 품질이 양호합니다. 현재 프로세스를 유지하세요.")

            return recommendations

        except Exception as e:
            self.logger.error(f"권장사항 생성 실패: {e}")
            return ["권장사항 생성 중 오류가 발생했습니다."]
