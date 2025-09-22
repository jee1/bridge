"""기본 통계 분석 도구 모듈

C1 마일스톤 1.2: 기본 통계 분석 도구
- 기술 통계: 평균, 중앙값, 표준편차, 분위수, 최댓값, 최솟값
- 분포 분석: 히스토그램, 기본 분포 통계
- 상관관계 분석: 피어슨 상관계수, 상관관계 매트릭스
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa

from bridge.analytics.core.data_integration import UnifiedDataFrame


@dataclass
class DescriptiveStats:
    """기술 통계 결과를 담는 데이터 클래스"""

    count: int
    mean: float
    std: float
    min: float
    max: float
    q25: float  # 25% 분위수
    q50: float  # 50% 분위수 (중앙값)
    q75: float  # 75% 분위수
    missing_count: int
    missing_ratio: float


@dataclass
class CorrelationResult:
    """상관관계 분석 결과를 담는 데이터 클래스"""

    correlation_matrix: pd.DataFrame
    strong_correlations: List[Dict[str, Any]]  # 강한 상관관계 (|r| > 0.7)
    moderate_correlations: List[Dict[str, Any]]  # 중간 상관관계 (0.3 < |r| <= 0.7)


class StatisticsAnalyzer:
    """기본 통계 분석을 수행하는 클래스"""

    def __init__(self):
        """통계 분석기 초기화"""
        pass

    def calculate_descriptive_stats(
        self, df: UnifiedDataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, DescriptiveStats]:
        """기술 통계를 계산합니다.

        Args:
            df: 분석할 데이터프레임
            columns: 분석할 컬럼 목록. None이면 모든 숫자형 컬럼 분석

        Returns:
            컬럼별 기술 통계 결과 딕셔너리
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
            series = pandas_df[col]

            # 결측값 통계
            missing_count = series.isnull().sum()
            missing_ratio = missing_count / len(series)

            # 유효한 값들로만 통계 계산
            valid_series = series.dropna()

            if len(valid_series) == 0:
                # 모든 값이 결측값인 경우
                results[col] = DescriptiveStats(
                    count=0,
                    mean=0.0,
                    std=0.0,
                    min=0.0,
                    max=0.0,
                    q25=0.0,
                    q50=0.0,
                    q75=0.0,
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                )
            else:
                results[col] = DescriptiveStats(
                    count=len(valid_series),
                    mean=float(valid_series.mean()),
                    std=float(valid_series.std()),
                    min=float(valid_series.min()),
                    max=float(valid_series.max()),
                    q25=float(valid_series.quantile(0.25)),
                    q50=float(valid_series.quantile(0.50)),
                    q75=float(valid_series.quantile(0.75)),
                    missing_count=missing_count,
                    missing_ratio=missing_ratio,
                )

        return results

    def calculate_correlation(
        self, df: UnifiedDataFrame, columns: Optional[List[str]] = None, method: str = "pearson"
    ) -> CorrelationResult:
        """상관관계 분석을 수행합니다.

        Args:
            df: 분석할 데이터프레임
            columns: 분석할 컬럼 목록. None이면 모든 숫자형 컬럼 분석
            method: 상관계수 계산 방법 ('pearson', 'spearman', 'kendall')

        Returns:
            상관관계 분석 결과
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

        if len(numeric_columns) < 2:
            return CorrelationResult(
                correlation_matrix=pd.DataFrame(), strong_correlations=[], moderate_correlations=[]
            )

        # 상관계수 계산
        corr_matrix = pandas_df[numeric_columns].corr(method=method)

        # 강한 상관관계와 중간 상관관계 식별
        strong_correlations = []
        moderate_correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if not np.isnan(corr_value):
                    correlation_info = {
                        "column1": col1,
                        "column2": col2,
                        "correlation": corr_value,
                        "abs_correlation": abs(corr_value),
                    }

                    if abs(corr_value) > 0.7:
                        strong_correlations.append(correlation_info)
                    elif abs(corr_value) > 0.3:
                        moderate_correlations.append(correlation_info)

        # 절댓값 기준으로 정렬
        strong_correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)
        moderate_correlations.sort(key=lambda x: x["abs_correlation"], reverse=True)

        return CorrelationResult(
            correlation_matrix=corr_matrix,
            strong_correlations=strong_correlations,
            moderate_correlations=moderate_correlations,
        )

    def calculate_distribution_stats(
        self, df: UnifiedDataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """분포 통계를 계산합니다.

        Args:
            df: 분석할 데이터프레임
            columns: 분석할 컬럼 목록. None이면 모든 숫자형 컬럼 분석

        Returns:
            컬럼별 분포 통계 결과 딕셔너리
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
                results[col] = {
                    "skewness": 0.0,
                    "kurtosis": 0.0,
                    "variance": 0.0,
                    "range": 0.0,
                    "iqr": 0.0,
                }
                continue

            # 왜도 (skewness)
            skewness = float(series.skew())

            # 첨도 (kurtosis)
            kurtosis = float(series.kurtosis())

            # 분산
            variance = float(series.var())

            # 범위
            range_val = float(series.max() - series.min())

            # 사분위수 범위 (IQR)
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            iqr = q75 - q25

            results[col] = {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "variance": variance,
                "range": range_val,
                "iqr": iqr,
            }

        return results

    def generate_summary_report(
        self, df: UnifiedDataFrame, columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """종합 통계 요약 리포트를 생성합니다.

        Args:
            df: 분석할 데이터프레임
            columns: 분석할 컬럼 목록. None이면 모든 숫자형 컬럼 분석

        Returns:
            종합 통계 요약 리포트
        """
        # 기술 통계
        descriptive_stats = self.calculate_descriptive_stats(df, columns)

        # 상관관계 분석
        correlation_result = self.calculate_correlation(df, columns)

        # 분포 통계
        distribution_stats = self.calculate_distribution_stats(df, columns)

        # 전체 데이터 요약
        pandas_df = df.to_pandas()
        total_rows = len(pandas_df)
        total_columns = len(pandas_df.columns)
        numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()

        return {
            "data_overview": {
                "total_rows": total_rows,
                "total_columns": total_columns,
                "numeric_columns": len(numeric_columns),
                "analyzed_columns": list(descriptive_stats.keys()),
            },
            "descriptive_stats": {
                col: {
                    "count": stats.count,
                    "mean": stats.mean,
                    "std": stats.std,
                    "min": stats.min,
                    "max": stats.max,
                    "q25": stats.q25,
                    "q50": stats.q50,
                    "q75": stats.q75,
                    "missing_count": stats.missing_count,
                    "missing_ratio": stats.missing_ratio,
                }
                for col, stats in descriptive_stats.items()
            },
            "correlation_analysis": {
                "strong_correlations": correlation_result.strong_correlations,
                "moderate_correlations": correlation_result.moderate_correlations,
                "correlation_matrix": correlation_result.correlation_matrix.to_dict(),
            },
            "distribution_stats": distribution_stats,
        }
