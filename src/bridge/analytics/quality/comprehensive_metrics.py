"""종합 품질 메트릭 시스템.

CA 마일스톤 3.3: 데이터 품질 관리 시스템
- 완전성, 정확성, 일관성, 유효성 메트릭을 종합적으로 평가
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from bridge.analytics.core.data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """품질 메트릭 결과를 담는 데이터 클래스"""

    completeness: float  # 완전성 (0-1)
    accuracy: float  # 정확성 (0-1)
    consistency: float  # 일관성 (0-1)
    validity: float  # 유효성 (0-1)
    overall_score: float  # 전체 품질 점수 (0-1)

    # 세부 메트릭
    missing_ratio: float
    duplicate_ratio: float
    outlier_ratio: float
    constraint_violations: int
    data_type_consistency: float
    range_violations: int
    format_violations: int


@dataclass
class QualityTrend:
    """품질 트렌드 분석 결과를 담는 데이터 클래스"""

    timestamp: str
    metrics: QualityMetrics
    trend_direction: str  # 'improving', 'stable', 'declining'
    trend_score: float  # -1 to 1
    alerts: List[str]  # 품질 경고 목록


class ComprehensiveQualityMetrics:
    """종합 품질 메트릭 계산 클래스"""

    def __init__(self):
        """품질 메트릭 계산기 초기화"""
        self.logger = logging.getLogger(__name__)

    def calculate_completeness(self, data: UnifiedDataFrame) -> Dict[str, Any]:
        """완전성 메트릭 계산

        Args:
            data: 분석할 데이터

        Returns:
            완전성 메트릭 딕셔너리
        """
        try:
            df = data.to_pandas()

            # 전체 결측값 비율
            total_missing = df.isnull().sum().sum()
            total_cells = df.size
            missing_ratio = total_missing / total_cells if total_cells > 0 else 0

            # 컬럼별 결측값 비율
            column_missing = df.isnull().sum() / len(df)

            # 완전성 점수 (1 - 결측값 비율)
            completeness_score = 1 - missing_ratio

            return {
                "completeness_score": completeness_score,
                "missing_ratio": missing_ratio,
                "column_missing_ratios": column_missing.to_dict(),
                "total_missing_cells": int(total_missing),
                "total_cells": total_cells,
            }

        except Exception as e:
            self.logger.error(f"완전성 메트릭 계산 실패: {e}")
            return {
                "completeness_score": 0.0,
                "missing_ratio": 1.0,
                "column_missing_ratios": {},
                "total_missing_cells": 0,
                "total_cells": 0,
            }

    def calculate_accuracy(
        self, data: UnifiedDataFrame, reference_data: Optional[UnifiedDataFrame] = None
    ) -> Dict[str, Any]:
        """정확성 메트릭 계산

        Args:
            data: 분석할 데이터
            reference_data: 참조 데이터 (선택사항)

        Returns:
            정확성 메트릭 딕셔너리
        """
        try:
            df = data.to_pandas()

            # 중복 데이터 비율
            duplicate_ratio = df.duplicated().sum() / len(df) if len(df) > 0 else 0

            # 데이터 타입 일관성
            type_consistency = self._calculate_type_consistency(df)

            # 범위 위반 검사
            range_violations = self._check_range_violations(df)

            # 정확성 점수 계산
            accuracy_score = 1 - (duplicate_ratio + (1 - type_consistency) + range_violations)
            accuracy_score = max(0, min(1, accuracy_score))  # 0-1 범위로 제한

            return {
                "accuracy_score": accuracy_score,
                "duplicate_ratio": duplicate_ratio,
                "type_consistency": type_consistency,
                "range_violations": range_violations,
                "duplicate_count": int(df.duplicated().sum()),
            }

        except Exception as e:
            self.logger.error(f"정확성 메트릭 계산 실패: {e}")
            return {
                "accuracy_score": 0.0,
                "duplicate_ratio": 1.0,
                "type_consistency": 0.0,
                "range_violations": 1.0,
                "duplicate_count": 0,
            }

    def calculate_consistency(self, data: UnifiedDataFrame) -> Dict[str, Any]:
        """일관성 메트릭 계산

        Args:
            data: 분석할 데이터

        Returns:
            일관성 메트릭 딕셔너리
        """
        try:
            df = data.to_pandas()

            # 컬럼별 데이터 타입 일관성
            type_consistency = self._calculate_type_consistency(df)

            # 값의 일관성 (예: 같은 ID에 다른 이름)
            value_consistency = self._calculate_value_consistency(df)

            # 포맷 일관성
            format_consistency = self._calculate_format_consistency(df)

            # 일관성 점수
            consistency_score = (type_consistency + value_consistency + format_consistency) / 3

            return {
                "consistency_score": consistency_score,
                "type_consistency": type_consistency,
                "value_consistency": value_consistency,
                "format_consistency": format_consistency,
            }

        except Exception as e:
            self.logger.error(f"일관성 메트릭 계산 실패: {e}")
            return {
                "consistency_score": 0.0,
                "type_consistency": 0.0,
                "value_consistency": 0.0,
                "format_consistency": 0.0,
            }

    def calculate_validity(
        self, data: UnifiedDataFrame, constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """유효성 메트릭 계산

        Args:
            data: 분석할 데이터
            constraints: 유효성 제약조건 (선택사항)

        Returns:
            유효성 메트릭 딕셔너리
        """
        try:
            df = data.to_pandas()

            # 기본 유효성 검사
            validity_checks = self._perform_validity_checks(df, constraints)

            # 유효성 점수
            validity_score = validity_checks["validity_score"]

            return {
                "validity_score": validity_score,
                "constraint_violations": validity_checks["constraint_violations"],
                "format_violations": validity_checks["format_violations"],
                "range_violations": validity_checks["range_violations"],
                "total_violations": validity_checks["total_violations"],
            }

        except Exception as e:
            self.logger.error(f"유효성 메트릭 계산 실패: {e}")
            return {
                "validity_score": 0.0,
                "constraint_violations": 0,
                "format_violations": 0,
                "range_violations": 0,
                "total_violations": 0,
            }

    def calculate_overall_quality(
        self,
        data: UnifiedDataFrame,
        reference_data: Optional[UnifiedDataFrame] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> QualityMetrics:
        """전체 품질 메트릭 계산

        Args:
            data: 분석할 데이터
            reference_data: 참조 데이터 (선택사항)
            constraints: 유효성 제약조건 (선택사항)

        Returns:
            QualityMetrics: 전체 품질 메트릭
        """
        try:
            # 각 메트릭 계산
            completeness = self.calculate_completeness(data)
            accuracy = self.calculate_accuracy(data, reference_data)
            consistency = self.calculate_consistency(data)
            validity = self.calculate_validity(data, constraints)

            # 전체 품질 점수 계산 (가중 평균)
            overall_score = (
                completeness["completeness_score"] * 0.3
                + accuracy["accuracy_score"] * 0.3
                + consistency["consistency_score"] * 0.2
                + validity["validity_score"] * 0.2
            )

            return QualityMetrics(
                completeness=completeness["completeness_score"],
                accuracy=accuracy["accuracy_score"],
                consistency=consistency["consistency_score"],
                validity=validity["validity_score"],
                overall_score=overall_score,
                missing_ratio=completeness["missing_ratio"],
                duplicate_ratio=accuracy["duplicate_ratio"],
                outlier_ratio=0.0,  # 이상치 비율은 별도 계산
                constraint_violations=validity["constraint_violations"],
                data_type_consistency=consistency["type_consistency"],
                range_violations=validity["range_violations"],
                format_violations=validity["format_violations"],
            )

        except Exception as e:
            self.logger.error(f"전체 품질 메트릭 계산 실패: {e}")
            return QualityMetrics(
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
            )

    def _calculate_type_consistency(self, df: pd.DataFrame) -> float:
        """데이터 타입 일관성 계산"""
        try:
            consistency_scores = []
            for col in df.columns:
                if df[col].dtype == "object":
                    # 문자열 컬럼의 일관성 검사
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # 숫자로 변환 가능한 비율
                        numeric_ratio = non_null_values.str.match(r"^-?\d+\.?\d*$").mean()
                        consistency_scores.append(1 - numeric_ratio)
                    else:
                        consistency_scores.append(1.0)
                else:
                    # 숫자 컬럼의 일관성 검사
                    consistency_scores.append(1.0)

            return np.mean(consistency_scores) if consistency_scores else 1.0

        except Exception:
            return 0.0

    def _calculate_value_consistency(self, df: pd.DataFrame) -> float:
        """값의 일관성 계산"""
        try:
            # ID 컬럼이 있는 경우 일관성 검사
            id_columns = [col for col in df.columns if "id" in col.lower()]
            if not id_columns:
                return 1.0

            consistency_scores = []
            for id_col in id_columns:
                # 같은 ID에 대해 다른 값이 있는지 검사
                grouped = df.groupby(id_col).nunique()
                inconsistent_groups = (grouped > 1).sum()
                total_groups = len(grouped)

                if total_groups > 0:
                    consistency_score = 1 - (inconsistent_groups / total_groups)
                    consistency_scores.append(consistency_score)

            return np.mean(consistency_scores) if consistency_scores else 1.0

        except Exception:
            return 1.0

    def _calculate_format_consistency(self, df: pd.DataFrame) -> float:
        """포맷 일관성 계산"""
        try:
            format_scores = []
            for col in df.columns:
                if df[col].dtype == "object":
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        # 이메일, 전화번호 등 포맷 일관성 검사
                        if "email" in col.lower():
                            email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                            valid_emails = non_null_values.str.match(email_pattern).mean()
                            format_scores.append(valid_emails)
                        elif "phone" in col.lower():
                            phone_pattern = r"^[\d\s\-\+\(\)]+$"
                            valid_phones = non_null_values.str.match(phone_pattern).mean()
                            format_scores.append(valid_phones)
                        else:
                            format_scores.append(1.0)
                    else:
                        format_scores.append(1.0)
                else:
                    format_scores.append(1.0)

            return np.mean(format_scores) if format_scores else 1.0

        except Exception:
            return 1.0

    def _check_range_violations(self, df: pd.DataFrame) -> float:
        """범위 위반 검사"""
        try:
            violations = 0
            total_checks = 0

            for col in df.columns:
                if df[col].dtype in ["int64", "float64"]:
                    # 음수 값 검사
                    if "age" in col.lower() or "count" in col.lower():
                        negative_count = (df[col] < 0).sum()
                        violations += negative_count
                        total_checks += len(df[col])

                    # 비정상적으로 큰 값 검사
                    if len(df[col]) > 0:
                        q99 = df[col].quantile(0.99)
                        q01 = df[col].quantile(0.01)
                        iqr = q99 - q01
                        upper_bound = q99 + 3 * iqr
                        lower_bound = q01 - 3 * iqr

                        outlier_count = ((df[col] > upper_bound) | (df[col] < lower_bound)).sum()
                        violations += outlier_count
                        total_checks += len(df[col])

            return violations / total_checks if total_checks > 0 else 0.0

        except Exception:
            return 0.0

    def _perform_validity_checks(
        self, df: pd.DataFrame, constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """유효성 검사 수행"""
        try:
            constraint_violations = 0
            format_violations = 0
            range_violations = 0

            # 기본 제약조건 검사
            if constraints:
                for col, rules in constraints.items():
                    if col in df.columns:
                        if "min" in rules:
                            constraint_violations += (df[col] < rules["min"]).sum()
                        if "max" in rules:
                            constraint_violations += (df[col] > rules["max"]).sum()
                        if "pattern" in rules:
                            pattern = rules["pattern"]
                            format_violations += (~df[col].astype(str).str.match(pattern)).sum()

            # 범위 위반 검사
            range_violations = int(self._check_range_violations(df) * len(df))

            total_violations = constraint_violations + format_violations + range_violations
            total_checks = len(df) * len(df.columns)

            validity_score = 1 - (total_violations / total_checks) if total_checks > 0 else 1.0
            validity_score = max(0, min(1, validity_score))

            return {
                "validity_score": validity_score,
                "constraint_violations": int(constraint_violations),
                "format_violations": int(format_violations),
                "range_violations": int(range_violations),
                "total_violations": int(total_violations),
            }

        except Exception as e:
            self.logger.error(f"유효성 검사 실패: {e}")
            return {
                "validity_score": 0.0,
                "constraint_violations": 0,
                "format_violations": 0,
                "range_violations": 0,
                "total_violations": 0,
            }
