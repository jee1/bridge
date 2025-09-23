"""고급 이상치 탐지 시스템.

CA 마일스톤 3.3: 데이터 품질 관리 시스템
- Isolation Forest, LOF 등 머신러닝 기반 이상치 탐지
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from bridge.analytics.core.data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)


@dataclass
class OutlierDetectionResult:
    """이상치 탐지 결과를 담는 데이터 클래스"""

    outlier_indices: List[int]  # 이상치 인덱스
    outlier_scores: List[float]  # 이상치 점수
    outlier_ratio: float  # 이상치 비율
    method: str  # 사용된 방법
    confidence: float  # 신뢰도
    details: Dict[str, Any]  # 세부 정보


class AdvancedOutlierDetector:
    """고급 이상치 탐지 클래스"""

    def __init__(self):
        """이상치 탐지기 초기화"""
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()

    def detect_outliers_isolation_forest(
        self, data: UnifiedDataFrame, contamination: float = 0.1, random_state: int = 42
    ) -> OutlierDetectionResult:
        """Isolation Forest를 사용한 이상치 탐지

        Args:
            data: 분석할 데이터
            contamination: 이상치 비율 (0-0.5)
            random_state: 랜덤 시드

        Returns:
            OutlierDetectionResult: 이상치 탐지 결과
        """
        try:
            df = data.to_pandas()

            # 숫자형 컬럼만 선택
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return OutlierDetectionResult(
                    outlier_indices=[],
                    outlier_scores=[],
                    outlier_ratio=0.0,
                    method="isolation_forest",
                    confidence=0.0,
                    details={"error": "숫자형 컬럼이 없습니다."},
                )

            # 데이터 전처리
            X = df[numeric_columns].fillna(df[numeric_columns].mean())
            X_scaled = self.scaler.fit_transform(X)

            # Isolation Forest 모델 훈련
            model = IsolationForest(
                contamination=contamination, random_state=random_state, n_estimators=100
            )

            # 이상치 예측
            outlier_predictions = model.fit_predict(X_scaled)
            outlier_scores = model.decision_function(X_scaled)

            # 이상치 인덱스 추출
            outlier_indices = np.where(outlier_predictions == -1)[0].tolist()
            outlier_ratio = len(outlier_indices) / len(df)

            # 신뢰도 계산 (점수 기반)
            confidence = (
                np.mean(np.abs(outlier_scores[outlier_indices])) if outlier_indices else 0.0
            )

            return OutlierDetectionResult(
                outlier_indices=outlier_indices,
                outlier_scores=outlier_scores.tolist(),
                outlier_ratio=outlier_ratio,
                method="isolation_forest",
                confidence=confidence,
                details={
                    "contamination": contamination,
                    "n_estimators": 100,
                    "numeric_columns": numeric_columns.tolist(),
                    "total_samples": len(df),
                },
            )

        except Exception as e:
            self.logger.error(f"Isolation Forest 이상치 탐지 실패: {e}")
            return OutlierDetectionResult(
                outlier_indices=[],
                outlier_scores=[],
                outlier_ratio=0.0,
                method="isolation_forest",
                confidence=0.0,
                details={"error": str(e)},
            )

    def detect_outliers_lof(
        self, data: UnifiedDataFrame, n_neighbors: int = 20, contamination: float = 0.1
    ) -> OutlierDetectionResult:
        """Local Outlier Factor를 사용한 이상치 탐지

        Args:
            data: 분석할 데이터
            n_neighbors: 이웃 수
            contamination: 이상치 비율 (0-0.5)

        Returns:
            OutlierDetectionResult: 이상치 탐지 결과
        """
        try:
            df = data.to_pandas()

            # 숫자형 컬럼만 선택
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return OutlierDetectionResult(
                    outlier_indices=[],
                    outlier_scores=[],
                    outlier_ratio=0.0,
                    method="lof",
                    confidence=0.0,
                    details={"error": "숫자형 컬럼이 없습니다."},
                )

            # 데이터 전처리
            X = df[numeric_columns].fillna(df[numeric_columns].mean())
            X_scaled = self.scaler.fit_transform(X)

            # LOF 모델 훈련
            model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)

            # 이상치 예측
            outlier_predictions = model.fit_predict(X_scaled)
            outlier_scores = model.negative_outlier_factor_

            # 이상치 인덱스 추출
            outlier_indices = np.where(outlier_predictions == -1)[0].tolist()
            outlier_ratio = len(outlier_indices) / len(df)

            # 신뢰도 계산 (점수 기반)
            confidence = (
                np.mean(np.abs(outlier_scores[outlier_indices])) if outlier_indices else 0.0
            )

            return OutlierDetectionResult(
                outlier_indices=outlier_indices,
                outlier_scores=outlier_scores.tolist(),
                outlier_ratio=outlier_ratio,
                method="lof",
                confidence=confidence,
                details={
                    "n_neighbors": n_neighbors,
                    "contamination": contamination,
                    "numeric_columns": numeric_columns.tolist(),
                    "total_samples": len(df),
                },
            )

        except Exception as e:
            self.logger.error(f"LOF 이상치 탐지 실패: {e}")
            return OutlierDetectionResult(
                outlier_indices=[],
                outlier_scores=[],
                outlier_ratio=0.0,
                method="lof",
                confidence=0.0,
                details={"error": str(e)},
            )

    def detect_outliers_one_class_svm(
        self, data: UnifiedDataFrame, nu: float = 0.1, kernel: str = "rbf"
    ) -> OutlierDetectionResult:
        """One-Class SVM을 사용한 이상치 탐지

        Args:
            data: 분석할 데이터
            nu: 이상치 비율 (0-1)
            kernel: 커널 함수

        Returns:
            OutlierDetectionResult: 이상치 탐지 결과
        """
        try:
            df = data.to_pandas()

            # 숫자형 컬럼만 선택
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == 0:
                return OutlierDetectionResult(
                    outlier_indices=[],
                    outlier_scores=[],
                    outlier_ratio=0.0,
                    method="one_class_svm",
                    confidence=0.0,
                    details={"error": "숫자형 컬럼이 없습니다."},
                )

            # 데이터 전처리
            X = df[numeric_columns].fillna(df[numeric_columns].mean())
            X_scaled = self.scaler.fit_transform(X)

            # One-Class SVM 모델 훈련
            model = OneClassSVM(nu=nu, kernel=kernel)

            # 이상치 예측
            outlier_predictions = model.fit_predict(X_scaled)
            outlier_scores = model.decision_function(X_scaled)

            # 이상치 인덱스 추출
            outlier_indices = np.where(outlier_predictions == -1)[0].tolist()
            outlier_ratio = len(outlier_indices) / len(df)

            # 신뢰도 계산 (점수 기반)
            confidence = (
                np.mean(np.abs(outlier_scores[outlier_indices])) if outlier_indices else 0.0
            )

            return OutlierDetectionResult(
                outlier_indices=outlier_indices,
                outlier_scores=outlier_scores.tolist(),
                outlier_ratio=outlier_ratio,
                method="one_class_svm",
                confidence=confidence,
                details={
                    "nu": nu,
                    "kernel": kernel,
                    "numeric_columns": numeric_columns.tolist(),
                    "total_samples": len(df),
                },
            )

        except Exception as e:
            self.logger.error(f"One-Class SVM 이상치 탐지 실패: {e}")
            return OutlierDetectionResult(
                outlier_indices=[],
                outlier_scores=[],
                outlier_ratio=0.0,
                method="one_class_svm",
                confidence=0.0,
                details={"error": str(e)},
            )

    def detect_outliers_ensemble(
        self, data: UnifiedDataFrame, methods: List[str] = None, voting_threshold: float = 0.5
    ) -> OutlierDetectionResult:
        """앙상블 방법을 사용한 이상치 탐지

        Args:
            data: 분석할 데이터
            methods: 사용할 방법들
            voting_threshold: 투표 임계값

        Returns:
            OutlierDetectionResult: 이상치 탐지 결과
        """
        try:
            if methods is None:
                methods = ["isolation_forest", "lof", "one_class_svm"]

            # 각 방법으로 이상치 탐지
            results = []
            for method in methods:
                if method == "isolation_forest":
                    result = self.detect_outliers_isolation_forest(data)
                elif method == "lof":
                    result = self.detect_outliers_lof(data)
                elif method == "one_class_svm":
                    result = self.detect_outliers_one_class_svm(data)
                else:
                    continue

                results.append(result)

            if not results:
                return OutlierDetectionResult(
                    outlier_indices=[],
                    outlier_scores=[],
                    outlier_ratio=0.0,
                    method="ensemble",
                    confidence=0.0,
                    details={"error": "유효한 방법이 없습니다."},
                )

            # 앙상블 결과 계산
            df = data.to_pandas()
            outlier_votes = np.zeros(len(df))

            for result in results:
                for idx in result.outlier_indices:
                    outlier_votes[idx] += 1

            # 투표 임계값 이상인 경우 이상치로 판정
            outlier_indices = np.where(outlier_votes >= voting_threshold * len(results))[0].tolist()
            outlier_ratio = len(outlier_indices) / len(df)

            # 신뢰도 계산 (투표 비율)
            confidence = (
                np.mean(outlier_votes[outlier_indices]) / len(results) if outlier_indices else 0.0
            )

            return OutlierDetectionResult(
                outlier_indices=outlier_indices,
                outlier_scores=outlier_votes.tolist(),
                outlier_ratio=outlier_ratio,
                method="ensemble",
                confidence=confidence,
                details={
                    "methods": methods,
                    "voting_threshold": voting_threshold,
                    "individual_results": [r.details for r in results],
                    "total_samples": len(df),
                },
            )

        except Exception as e:
            self.logger.error(f"앙상블 이상치 탐지 실패: {e}")
            return OutlierDetectionResult(
                outlier_indices=[],
                outlier_scores=[],
                outlier_ratio=0.0,
                method="ensemble",
                confidence=0.0,
                details={"error": str(e)},
            )

    def detect_outliers_auto(self, data: UnifiedDataFrame) -> OutlierDetectionResult:
        """자동 이상치 탐지 (최적 방법 선택)

        Args:
            data: 분석할 데이터

        Returns:
            OutlierDetectionResult: 이상치 탐지 결과
        """
        try:
            df = data.to_pandas()

            # 데이터 크기에 따른 방법 선택
            if len(df) < 100:
                # 작은 데이터셋: LOF 사용
                return self.detect_outliers_lof(data, n_neighbors=min(10, len(df) // 2))
            elif len(df) < 1000:
                # 중간 데이터셋: Isolation Forest 사용
                return self.detect_outliers_isolation_forest(data)
            else:
                # 큰 데이터셋: 앙상블 사용
                return self.detect_outliers_ensemble(data)

        except Exception as e:
            self.logger.error(f"자동 이상치 탐지 실패: {e}")
            return OutlierDetectionResult(
                outlier_indices=[],
                outlier_scores=[],
                outlier_ratio=0.0,
                method="auto",
                confidence=0.0,
                details={"error": str(e)},
            )

    def get_outlier_summary(
        self, result: OutlierDetectionResult, data: UnifiedDataFrame
    ) -> Dict[str, Any]:
        """이상치 요약 정보 생성

        Args:
            result: 이상치 탐지 결과
            data: 원본 데이터

        Returns:
            이상치 요약 딕셔너리
        """
        try:
            df = data.to_pandas()

            if not result.outlier_indices:
                return {
                    "outlier_count": 0,
                    "outlier_ratio": 0.0,
                    "method": result.method,
                    "confidence": result.confidence,
                    "outlier_data": None,
                }

            # 이상치 데이터 추출
            outlier_data = df.iloc[result.outlier_indices]

            # 이상치 통계
            outlier_stats = {}
            for col in outlier_data.columns:
                if outlier_data[col].dtype in ["int64", "float64"]:
                    outlier_stats[col] = {
                        "mean": float(outlier_data[col].mean()),
                        "std": float(outlier_data[col].std()),
                        "min": float(outlier_data[col].min()),
                        "max": float(outlier_data[col].max()),
                    }

            return {
                "outlier_count": len(result.outlier_indices),
                "outlier_ratio": result.outlier_ratio,
                "method": result.method,
                "confidence": result.confidence,
                "outlier_data": outlier_data.to_dict("records"),
                "outlier_stats": outlier_stats,
                "details": result.details,
            }

        except Exception as e:
            self.logger.error(f"이상치 요약 생성 실패: {e}")
            return {
                "outlier_count": 0,
                "outlier_ratio": 0.0,
                "method": result.method,
                "confidence": 0.0,
                "outlier_data": None,
                "error": str(e),
            }
