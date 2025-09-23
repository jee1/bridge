"""품질 트렌드 분석 시스템.

CA 마일스톤 3.3: 데이터 품질 관리 시스템
- 시간에 따른 품질 변화 추적 및 예측
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

from bridge.analytics.core.data_integration import UnifiedDataFrame

from .comprehensive_metrics import ComprehensiveQualityMetrics, QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class QualityTrend:
    """품질 트렌드 분석 결과를 담는 데이터 클래스"""

    timestamp: str
    metrics: QualityMetrics
    trend_direction: str  # 'improving', 'stable', 'declining'
    trend_score: float  # -1 to 1
    alerts: List[str]  # 품질 경고 목록
    prediction: Optional[Dict[str, float]] = None  # 미래 예측


@dataclass
class TrendAnalysisResult:
    """트렌드 분석 결과를 담는 데이터 클래스"""

    trends: List[QualityTrend]
    overall_trend: str
    trend_strength: float
    predictions: Dict[str, List[float]]
    alerts: List[str]
    recommendations: List[str]


class QualityTrendAnalyzer:
    """품질 트렌드 분석 클래스"""

    def __init__(self):
        """품질 트렌드 분석기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.quality_calculator = ComprehensiveQualityMetrics()
        self.trend_history: List[QualityTrend] = []

        # 품질 임계값 설정
        self.thresholds = {
            "completeness": 0.8,
            "accuracy": 0.8,
            "consistency": 0.8,
            "validity": 0.8,
            "overall": 0.8,
        }

    def add_quality_snapshot(
        self, data: UnifiedDataFrame, timestamp: Optional[str] = None
    ) -> QualityTrend:
        """품질 스냅샷 추가

        Args:
            data: 분석할 데이터
            timestamp: 타임스탬프 (선택사항)

        Returns:
            QualityTrend: 품질 트렌드
        """
        try:
            if timestamp is None:
                timestamp = datetime.now().isoformat()

            # 품질 메트릭 계산
            metrics = self.quality_calculator.calculate_overall_quality(data)

            # 트렌드 방향 계산
            trend_direction, trend_score = self._calculate_trend_direction(metrics)

            # 경고 생성
            alerts = self._generate_alerts(metrics)

            # 품질 트렌드 생성
            trend = QualityTrend(
                timestamp=timestamp,
                metrics=metrics,
                trend_direction=trend_direction,
                trend_score=trend_score,
                alerts=alerts,
            )

            # 히스토리에 추가
            self.trend_history.append(trend)

            return trend

        except Exception as e:
            self.logger.error(f"품질 스냅샷 추가 실패: {e}")
            return QualityTrend(
                timestamp=timestamp or datetime.now().isoformat(),
                metrics=QualityMetrics(
                    completeness=0.0,
                    accuracy=0.0,
                    consistency=0.0,
                    validity=0.0,
                    overall_score=0.0,
                    missing_ratio=1.0,
                    duplicate_ratio=1.0,
                    outlier_ratio=1.0,
                    constraint_violations=0,
                    data_type_consistency=0.0,
                    range_violations=0,
                    format_violations=0,
                ),
                trend_direction="stable",
                trend_score=0.0,
                alerts=[f"오류: {str(e)}"],
            )

    def analyze_trends(self, days: int = 30) -> TrendAnalysisResult:
        """트렌드 분석 수행

        Args:
            days: 분석할 기간 (일)

        Returns:
            TrendAnalysisResult: 트렌드 분석 결과
        """
        try:
            # 최근 N일 데이터 필터링
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_trends = [
                trend
                for trend in self.trend_history
                if datetime.fromisoformat(trend.timestamp) >= cutoff_date
            ]

            if len(recent_trends) < 2:
                return TrendAnalysisResult(
                    trends=recent_trends,
                    overall_trend="insufficient_data",
                    trend_strength=0.0,
                    predictions={},
                    alerts=["데이터가 부족합니다."],
                    recommendations=["더 많은 데이터를 수집하세요."],
                )

            # 전체 트렌드 계산
            overall_trend, trend_strength = self._calculate_overall_trend(recent_trends)

            # 미래 예측
            predictions = self._predict_future_quality(recent_trends)

            # 경고 및 권장사항 생성
            alerts = self._generate_trend_alerts(recent_trends)
            recommendations = self._generate_recommendations(recent_trends, overall_trend)

            return TrendAnalysisResult(
                trends=recent_trends,
                overall_trend=overall_trend,
                trend_strength=trend_strength,
                predictions=predictions,
                alerts=alerts,
                recommendations=recommendations,
            )

        except Exception as e:
            self.logger.error(f"트렌드 분석 실패: {e}")
            return TrendAnalysisResult(
                trends=[],
                overall_trend="error",
                trend_strength=0.0,
                predictions={},
                alerts=[f"분석 오류: {str(e)}"],
                recommendations=["시스템 관리자에게 문의하세요."],
            )

    def _calculate_trend_direction(self, metrics: QualityMetrics) -> Tuple[str, float]:
        """트렌드 방향 계산

        Args:
            metrics: 현재 품질 메트릭

        Returns:
            (트렌드 방향, 트렌드 점수)
        """
        try:
            if len(self.trend_history) < 2:
                return "stable", 0.0

            # 최근 3개 데이터 포인트 사용
            recent_trends = self.trend_history[-3:]

            # 전체 품질 점수 추세 계산
            scores = [trend.metrics.overall_score for trend in recent_trends]

            if len(scores) < 2:
                return "stable", 0.0

            # 선형 회귀로 트렌드 계산
            X = np.array(range(len(scores))).reshape(-1, 1)
            y = np.array(scores)

            model = LinearRegression()
            model.fit(X, y)

            slope = model.coef_[0]

            if slope > 0.01:
                return "improving", min(1.0, slope * 10)
            elif slope < -0.01:
                return "declining", max(-1.0, slope * 10)
            else:
                return "stable", 0.0

        except Exception as e:
            self.logger.error(f"트렌드 방향 계산 실패: {e}")
            return "stable", 0.0

    def _calculate_overall_trend(self, trends: List[QualityTrend]) -> Tuple[str, float]:
        """전체 트렌드 계산

        Args:
            trends: 트렌드 목록

        Returns:
            (전체 트렌드, 트렌드 강도)
        """
        try:
            if len(trends) < 2:
                return "insufficient_data", 0.0

            # 각 메트릭별 트렌드 계산
            metrics = ["completeness", "accuracy", "consistency", "validity", "overall_score"]
            trend_scores = []

            for metric in metrics:
                values = [getattr(trend.metrics, metric) for trend in trends]
                if len(values) >= 2:
                    # 선형 회귀로 트렌드 계산
                    X = np.array(range(len(values))).reshape(-1, 1)
                    y = np.array(values)

                    model = LinearRegression()
                    model.fit(X, y)
                    slope = model.coef_[0]
                    trend_scores.append(slope)

            if not trend_scores:
                return "stable", 0.0

            # 평균 트렌드 점수
            avg_trend_score = np.mean(trend_scores)

            if avg_trend_score > 0.01:
                return "improving", min(1.0, avg_trend_score * 10)
            elif avg_trend_score < -0.01:
                return "declining", max(-1.0, avg_trend_score * 10)
            else:
                return "stable", 0.0

        except Exception as e:
            self.logger.error(f"전체 트렌드 계산 실패: {e}")
            return "stable", 0.0

    def _predict_future_quality(
        self, trends: List[QualityTrend], days_ahead: int = 7
    ) -> Dict[str, List[float]]:
        """미래 품질 예측

        Args:
            trends: 트렌드 목록
            days_ahead: 예측할 일수

        Returns:
            예측 결과 딕셔너리
        """
        try:
            if len(trends) < 3:
                return {}

            # 각 메트릭별 예측
            metrics = ["completeness", "accuracy", "consistency", "validity", "overall_score"]
            predictions = {}

            for metric in metrics:
                values = [getattr(trend.metrics, metric) for trend in trends]

                if len(values) >= 3:
                    # 다항식 회귀로 예측
                    X = np.array(range(len(values))).reshape(-1, 1)
                    y = np.array(values)

                    # 2차 다항식 피처 생성
                    poly_features = PolynomialFeatures(degree=2)
                    X_poly = poly_features.fit_transform(X)

                    model = LinearRegression()
                    model.fit(X_poly, y)

                    # 미래 예측
                    future_X = np.array(range(len(values), len(values) + days_ahead)).reshape(-1, 1)
                    future_X_poly = poly_features.transform(future_X)
                    future_predictions = model.predict(future_X_poly)

                    # 0-1 범위로 제한
                    future_predictions = np.clip(future_predictions, 0, 1)
                    predictions[metric] = future_predictions.tolist()

            return predictions

        except Exception as e:
            self.logger.error(f"미래 품질 예측 실패: {e}")
            return {}

    def _generate_alerts(self, metrics: QualityMetrics) -> List[str]:
        """품질 경고 생성

        Args:
            metrics: 품질 메트릭

        Returns:
            경고 목록
        """
        alerts = []

        # 각 메트릭별 임계값 검사
        if metrics.completeness < self.thresholds["completeness"]:
            alerts.append(f"완전성 점수가 낮습니다: {metrics.completeness:.2f}")

        if metrics.accuracy < self.thresholds["accuracy"]:
            alerts.append(f"정확성 점수가 낮습니다: {metrics.accuracy:.2f}")

        if metrics.consistency < self.thresholds["consistency"]:
            alerts.append(f"일관성 점수가 낮습니다: {metrics.consistency:.2f}")

        if metrics.validity < self.thresholds["validity"]:
            alerts.append(f"유효성 점수가 낮습니다: {metrics.validity:.2f}")

        if metrics.overall_score < self.thresholds["overall"]:
            alerts.append(f"전체 품질 점수가 낮습니다: {metrics.overall_score:.2f}")

        # 특별한 경고
        if metrics.missing_ratio > 0.5:
            alerts.append(f"결측값 비율이 높습니다: {metrics.missing_ratio:.2f}")

        if metrics.duplicate_ratio > 0.1:
            alerts.append(f"중복 데이터 비율이 높습니다: {metrics.duplicate_ratio:.2f}")

        if metrics.outlier_ratio > 0.1:
            alerts.append(f"이상치 비율이 높습니다: {metrics.outlier_ratio:.2f}")

        return alerts

    def _generate_trend_alerts(self, trends: List[QualityTrend]) -> List[str]:
        """트렌드 경고 생성

        Args:
            trends: 트렌드 목록

        Returns:
            경고 목록
        """
        alerts = []

        if len(trends) < 2:
            return alerts

        # 최근 트렌드 분석
        recent_trends = trends[-5:]  # 최근 5개 데이터 포인트

        # 급격한 품질 하락 감지
        if len(recent_trends) >= 2:
            current_score = recent_trends[-1].metrics.overall_score
            previous_score = recent_trends[-2].metrics.overall_score

            if current_score - previous_score < -0.1:
                alerts.append(
                    f"품질이 급격히 하락했습니다: {previous_score:.2f} → {current_score:.2f}"
                )

        # 지속적인 품질 하락 감지
        if len(recent_trends) >= 3:
            scores = [trend.metrics.overall_score for trend in recent_trends]
            if all(scores[i] > scores[i + 1] for i in range(len(scores) - 1)):
                alerts.append("품질이 지속적으로 하락하고 있습니다.")

        # 임계값 이하 지속 감지
        low_quality_days = sum(
            1 for trend in recent_trends if trend.metrics.overall_score < self.thresholds["overall"]
        )

        if low_quality_days >= 3:
            alerts.append(f"품질이 {low_quality_days}일 연속으로 임계값 이하입니다.")

        return alerts

    def _generate_recommendations(
        self, trends: List[QualityTrend], overall_trend: str
    ) -> List[str]:
        """권장사항 생성

        Args:
            trends: 트렌드 목록
            overall_trend: 전체 트렌드

        Returns:
            권장사항 목록
        """
        recommendations = []

        if not trends:
            return recommendations

        latest_metrics = trends[-1].metrics

        # 완전성 개선 권장사항
        if latest_metrics.completeness < self.thresholds["completeness"]:
            recommendations.append(
                "결측값 처리를 개선하세요. 데이터 수집 프로세스를 점검하고 결측값 대체 전략을 수립하세요."
            )

        # 정확성 개선 권장사항
        if latest_metrics.accuracy < self.thresholds["accuracy"]:
            recommendations.append(
                "데이터 정확성을 개선하세요. 중복 데이터를 제거하고 데이터 검증 규칙을 강화하세요."
            )

        # 일관성 개선 권장사항
        if latest_metrics.consistency < self.thresholds["consistency"]:
            recommendations.append(
                "데이터 일관성을 개선하세요. 데이터 표준화 규칙을 수립하고 데이터 변환 프로세스를 점검하세요."
            )

        # 유효성 개선 권장사항
        if latest_metrics.validity < self.thresholds["validity"]:
            recommendations.append(
                "데이터 유효성을 개선하세요. 데이터 검증 규칙을 강화하고 제약조건을 점검하세요."
            )

        # 트렌드 기반 권장사항
        if overall_trend == "declining":
            recommendations.append(
                "품질이 하락하고 있습니다. 데이터 품질 관리 프로세스를 전면 점검하세요."
            )
        elif overall_trend == "improving":
            recommendations.append("품질이 개선되고 있습니다. 현재 프로세스를 유지하세요.")
        else:
            recommendations.append("품질이 안정적입니다. 지속적인 모니터링을 유지하세요.")

        return recommendations

    def get_quality_dashboard_data(self) -> Dict[str, Any]:
        """품질 대시보드 데이터 생성

        Returns:
            대시보드 데이터 딕셔너리
        """
        try:
            if not self.trend_history:
                return {
                    "trends": [],
                    "overall_trend": "no_data",
                    "trend_strength": 0.0,
                    "alerts": ["데이터가 없습니다."],
                    "recommendations": ["품질 모니터링을 시작하세요."],
                }

            # 최근 30일 트렌드 분석
            analysis_result = self.analyze_trends(days=30)

            # 대시보드 데이터 구성
            dashboard_data = {
                "trends": [
                    {
                        "timestamp": trend.timestamp,
                        "completeness": trend.metrics.completeness,
                        "accuracy": trend.metrics.accuracy,
                        "consistency": trend.metrics.consistency,
                        "validity": trend.metrics.validity,
                        "overall_score": trend.metrics.overall_score,
                        "trend_direction": trend.trend_direction,
                        "trend_score": trend.trend_score,
                        "alerts": trend.alerts,
                    }
                    for trend in analysis_result.trends
                ],
                "overall_trend": analysis_result.overall_trend,
                "trend_strength": analysis_result.trend_strength,
                "predictions": analysis_result.predictions,
                "alerts": analysis_result.alerts,
                "recommendations": analysis_result.recommendations,
                "thresholds": self.thresholds,
            }

            return dashboard_data

        except Exception as e:
            self.logger.error(f"대시보드 데이터 생성 실패: {e}")
            return {
                "trends": [],
                "overall_trend": "error",
                "trend_strength": 0.0,
                "alerts": [f"오류: {str(e)}"],
                "recommendations": ["시스템 관리자에게 문의하세요."],
            }

    def set_thresholds(self, thresholds: Dict[str, float]):
        """품질 임계값 설정

        Args:
            thresholds: 임계값 딕셔너리
        """
        self.thresholds.update(thresholds)

    def get_trend_summary(self) -> Dict[str, Any]:
        """트렌드 요약 정보 반환

        Returns:
            트렌드 요약 딕셔너리
        """
        return {
            "total_snapshots": len(self.trend_history),
            "latest_timestamp": self.trend_history[-1].timestamp if self.trend_history else None,
            "latest_quality_score": (
                self.trend_history[-1].metrics.overall_score if self.trend_history else 0.0
            ),
            "thresholds": self.thresholds,
        }
