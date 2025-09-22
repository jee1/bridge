"""데이터 품질 검사 모듈

C1 마일스톤 1.3: 데이터 품질 검사
- 결측값 분석: 결측값 비율, 패턴 분석
- 이상치 탐지: IQR 기반 이상치 식별
- 데이터 일관성: 기본적인 데이터 무결성 검사
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from bridge.analytics.core.data_integration import UnifiedDataFrame


@dataclass
class MissingValueStats:
    """결측값 통계를 담는 데이터 클래스"""

    total_values: int
    missing_count: int
    missing_ratio: float
    complete_count: int
    complete_ratio: float


@dataclass
class OutlierStats:
    """이상치 통계를 담는 데이터 클래스"""

    total_values: int
    outlier_count: int
    outlier_ratio: float
    outlier_indices: List[int]
    outlier_values: List[float]
    lower_bound: float
    upper_bound: float


@dataclass
class ConsistencyStats:
    """데이터 일관성 통계를 담는 데이터 클래스"""

    duplicate_rows: int
    duplicate_ratio: float
    unique_rows: int
    unique_ratio: float
    data_types_consistent: bool
    schema_issues: List[str]


@dataclass
class QualityReport:
    """종합 품질 리포트를 담는 데이터 클래스"""

    overall_score: float  # 0-100 점수
    missing_value_score: float
    outlier_score: float
    consistency_score: float
    recommendations: List[str]
    critical_issues: List[str]


class QualityChecker:
    """데이터 품질 검사를 수행하는 클래스"""

    def __init__(self):
        """품질 검사기 초기화"""
        pass

    def analyze_missing_values(
        self, df: UnifiedDataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, MissingValueStats]:
        """결측값 분석을 수행합니다.

        Args:
            df: 분석할 데이터프레임
            columns: 분석할 컬럼 목록. None이면 모든 컬럼 분석

        Returns:
            컬럼별 결측값 통계 결과 딕셔너리
        """
        # Arrow Table을 pandas DataFrame으로 변환
        pandas_df = df.to_pandas()

        # 분석할 컬럼 선택
        if columns is None:
            target_columns = pandas_df.columns.tolist()
        else:
            target_columns = [col for col in columns if col in pandas_df.columns]

        results = {}

        for col in target_columns:
            series = pandas_df[col]
            total_values = len(series)
            missing_count = series.isnull().sum()
            missing_ratio = missing_count / total_values if total_values > 0 else 0.0
            complete_count = total_values - missing_count
            complete_ratio = complete_count / total_values if total_values > 0 else 0.0

            results[col] = MissingValueStats(
                total_values=total_values,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                complete_count=complete_count,
                complete_ratio=complete_ratio,
            )

        return results

    def detect_outliers(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> Dict[str, OutlierStats]:
        """이상치를 탐지합니다.

        Args:
            df: 분석할 데이터프레임
            columns: 분석할 컬럼 목록. None이면 모든 숫자형 컬럼 분석
            method: 이상치 탐지 방법 ('iqr', 'zscore')
            threshold: 이상치 판정 임계값

        Returns:
            컬럼별 이상치 통계 결과 딕셔너리
        """
        # Arrow Table을 pandas DataFrame으로 변환
        pandas_df = df.to_pandas()

        # 숫자형 컬럼만 선택
        if columns is None:
            numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [
                col
                for col in columns
                if col in pandas_df.columns
                and pandas_df[col].dtype in [np.number, "int64", "float64"]
            ]

        results = {}

        for col in numeric_columns:
            series = pandas_df[col].dropna()

            if len(series) == 0:
                results[col] = OutlierStats(
                    total_values=0,
                    outlier_count=0,
                    outlier_ratio=0.0,
                    outlier_indices=[],
                    outlier_values=[],
                    lower_bound=0.0,
                    upper_bound=0.0,
                )
                continue

            if method == "iqr":
                outlier_indices, outlier_values, lower_bound, upper_bound = (
                    self._detect_outliers_iqr(series, threshold)
                )
            elif method == "zscore":
                outlier_indices, outlier_values, lower_bound, upper_bound = (
                    self._detect_outliers_zscore(series, threshold)
                )
            else:
                raise ValueError(f"지원하지 않는 이상치 탐지 방법: {method}")

            total_values = len(series)
            outlier_count = len(outlier_indices)
            outlier_ratio = outlier_count / total_values if total_values > 0 else 0.0

            results[col] = OutlierStats(
                total_values=total_values,
                outlier_count=outlier_count,
                outlier_ratio=outlier_ratio,
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )

        return results

    def _detect_outliers_iqr(
        self, series: pd.Series, threshold: float
    ) -> Tuple[List[int], List[float], float, float]:
        """IQR 방법으로 이상치를 탐지합니다."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_indices = series[outlier_mask].index.tolist()
        outlier_values = series[outlier_mask].tolist()

        return outlier_indices, outlier_values, lower_bound, upper_bound

    def _detect_outliers_zscore(
        self, series: pd.Series, threshold: float
    ) -> Tuple[List[int], List[float], float, float]:
        """Z-score 방법으로 이상치를 탐지합니다."""
        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_mask = z_scores > threshold

        outlier_indices = series[outlier_mask].index.tolist()
        outlier_values = series[outlier_mask].tolist()

        # Z-score 방법에서는 경계값을 계산하지 않음
        lower_bound = float("nan")
        upper_bound = float("nan")

        return outlier_indices, outlier_values, lower_bound, upper_bound

    def check_consistency(self, df: UnifiedDataFrame) -> ConsistencyStats:
        """데이터 일관성을 검사합니다.

        Args:
            df: 분석할 데이터프레임

        Returns:
            데이터 일관성 통계 결과
        """
        # Arrow Table을 pandas DataFrame으로 변환
        pandas_df = df.to_pandas()

        # 중복 행 검사
        total_rows = len(pandas_df)
        duplicate_rows = pandas_df.duplicated().sum()
        duplicate_ratio = duplicate_rows / total_rows if total_rows > 0 else 0.0
        unique_rows = total_rows - duplicate_rows
        unique_ratio = unique_rows / total_rows if total_rows > 0 else 0.0

        # 데이터 타입 일관성 검사
        data_types_consistent = True
        schema_issues = []

        for col in pandas_df.columns:
            series = pandas_df[col]

            # 숫자형 컬럼에서 문자열이 섞여있는지 확인
            if series.dtype in [np.number, "int64", "float64"]:
                # 숫자로 변환할 수 없는 값이 있는지 확인
                try:
                    pd.to_numeric(series, errors="raise")
                except (ValueError, TypeError):
                    data_types_consistent = False
                    schema_issues.append(f"컬럼 '{col}': 숫자형이지만 숫자가 아닌 값 포함")

            # 문자열 컬럼에서 예상치 못한 패턴 확인
            elif series.dtype == "object":
                # 모든 값이 동일한 타입인지 확인
                value_types = set(type(val).__name__ for val in series.dropna())
                if len(value_types) > 1:
                    schema_issues.append(
                        f"컬럼 '{col}': 혼합된 데이터 타입 ({', '.join(value_types)})"
                    )

        return ConsistencyStats(
            duplicate_rows=duplicate_rows,
            duplicate_ratio=duplicate_ratio,
            unique_rows=unique_rows,
            unique_ratio=unique_ratio,
            data_types_consistent=data_types_consistent,
            schema_issues=schema_issues,
        )

    def generate_quality_report(
        self, df: UnifiedDataFrame, columns: Optional[List[str]] = None
    ) -> QualityReport:
        """종합 품질 리포트를 생성합니다.

        Args:
            df: 분석할 데이터프레임
            columns: 분석할 컬럼 목록. None이면 모든 컬럼 분석

        Returns:
            종합 품질 리포트
        """
        # 결측값 분석
        missing_stats = self.analyze_missing_values(df, columns)

        # 이상치 탐지
        outlier_stats = self.detect_outliers(df, columns)

        # 일관성 검사
        consistency_stats = self.check_consistency(df)

        # 점수 계산
        missing_score = self._calculate_missing_score(missing_stats)
        outlier_score = self._calculate_outlier_score(outlier_stats)
        consistency_score = self._calculate_consistency_score(consistency_stats)

        # 전체 점수 계산 (가중평균)
        overall_score = missing_score * 0.4 + outlier_score * 0.3 + consistency_score * 0.3

        # 권장사항 및 중요 이슈 생성
        recommendations = self._generate_recommendations(
            missing_stats, outlier_stats, consistency_stats
        )
        critical_issues = self._identify_critical_issues(
            missing_stats, outlier_stats, consistency_stats
        )

        return QualityReport(
            overall_score=overall_score,
            missing_value_score=missing_score,
            outlier_score=outlier_score,
            consistency_score=consistency_score,
            recommendations=recommendations,
            critical_issues=critical_issues,
        )

    def _calculate_missing_score(self, missing_stats: Dict[str, MissingValueStats]) -> float:
        """결측값 점수를 계산합니다."""
        if not missing_stats:
            return 100.0

        # 평균 완전성 비율을 점수로 변환
        avg_completeness = float(
            np.mean([stats.complete_ratio for stats in missing_stats.values()])
        )
        return avg_completeness * 100.0

    def _calculate_outlier_score(self, outlier_stats: Dict[str, OutlierStats]) -> float:
        """이상치 점수를 계산합니다."""
        if not outlier_stats:
            return 100.0

        # 평균 이상치 비율을 점수로 변환 (이상치가 적을수록 높은 점수)
        avg_outlier_ratio = float(
            np.mean([stats.outlier_ratio for stats in outlier_stats.values()])
        )
        return max(0.0, 100.0 - (avg_outlier_ratio * 100.0))

    def _calculate_consistency_score(self, consistency_stats: ConsistencyStats) -> float:
        """일관성 점수를 계산합니다."""
        score = 100.0

        # 중복 행 비율에 따른 감점
        score -= consistency_stats.duplicate_ratio * 20

        # 데이터 타입 일관성에 따른 감점
        if not consistency_stats.data_types_consistent:
            score -= 30

        # 스키마 이슈에 따른 감점
        score -= len(consistency_stats.schema_issues) * 10

        return max(0.0, score)

    def _generate_recommendations(
        self,
        missing_stats: Dict[str, MissingValueStats],
        outlier_stats: Dict[str, OutlierStats],
        consistency_stats: ConsistencyStats,
    ) -> List[str]:
        """권장사항을 생성합니다."""
        recommendations = []

        # 결측값 관련 권장사항
        high_missing_cols = [
            col for col, stats in missing_stats.items() if stats.missing_ratio > 0.1
        ]
        if high_missing_cols:
            recommendations.append(
                f"높은 결측값 비율 컬럼: {', '.join(high_missing_cols)} - 결측값 처리 방안 검토 필요"
            )

        # 이상치 관련 권장사항
        high_outlier_cols = [
            col for col, stats in outlier_stats.items() if stats.outlier_ratio > 0.05
        ]
        if high_outlier_cols:
            recommendations.append(
                f"높은 이상치 비율 컬럼: {', '.join(high_outlier_cols)} - 이상치 처리 방안 검토 필요"
            )

        # 일관성 관련 권장사항
        if consistency_stats.duplicate_ratio > 0.1:
            recommendations.append("중복 행 비율이 높음 - 중복 제거 또는 데이터 소스 검토 필요")

        if consistency_stats.schema_issues:
            recommendations.append("데이터 타입 일관성 문제 발견 - 데이터 정제 필요")

        return recommendations

    def _identify_critical_issues(
        self,
        missing_stats: Dict[str, MissingValueStats],
        outlier_stats: Dict[str, OutlierStats],
        consistency_stats: ConsistencyStats,
    ) -> List[str]:
        """중요 이슈를 식별합니다."""
        critical_issues = []

        # 심각한 결측값 문제
        critical_missing_cols = [
            col for col, stats in missing_stats.items() if stats.missing_ratio > 0.5
        ]
        if critical_missing_cols:
            critical_issues.append(
                f"심각한 결측값 문제: {', '.join(critical_missing_cols)} (50% 이상 결측)"
            )

        # 심각한 이상치 문제
        critical_outlier_cols = [
            col for col, stats in outlier_stats.items() if stats.outlier_ratio > 0.2
        ]
        if critical_outlier_cols:
            critical_issues.append(
                f"심각한 이상치 문제: {', '.join(critical_outlier_cols)} (20% 이상 이상치)"
            )

        # 심각한 일관성 문제
        if consistency_stats.duplicate_ratio > 0.5:
            critical_issues.append("심각한 중복 문제: 50% 이상 중복 행")

        if not consistency_stats.data_types_consistent:
            critical_issues.append("심각한 데이터 타입 불일치: 데이터 정제 필수")

        return critical_issues
