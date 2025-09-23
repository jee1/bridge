"""고급 통계 분석 모듈.

CA 마일스톤 3.2: 고급 통계 분석 및 시각화
고급 통계 분석 기능을 제공하는 모듈입니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu, pearsonr, spearmanr, ttest_ind, ttest_rel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from bridge.analytics.core.data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)


class AdvancedStatistics:
    """고급 통계 분석 클래스.
    
    고급 통계 분석 기능을 제공합니다.
    """

    def __init__(self):
        """AdvancedStatistics를 초기화합니다."""
        self.scaler = StandardScaler()

    def descriptive_statistics(
        self, 
        data: UnifiedDataFrame, 
        columns: Optional[List[str]] = None,
        include_percentiles: bool = True,
        include_skewness: bool = True,
        include_kurtosis: bool = True
    ) -> Dict[str, Any]:
        """고급 기술 통계를 계산합니다.
        
        Args:
            data: 분석할 데이터
            columns: 분석할 컬럼 목록 (None이면 모든 숫자 컬럼)
            include_percentiles: 백분위수 포함 여부
            include_skewness: 왜도 포함 여부
            include_kurtosis: 첨도 포함 여부
            
        Returns:
            Dict[str, Any]: 고급 기술 통계 결과
        """
        df = data.to_pandas()
        
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [col for col in columns if col in df.columns and df[col].dtype in [np.number, 'int64', 'float64']]
        
        if not numeric_columns:
            return {"error": "분석할 숫자 컬럼이 없습니다."}
        
        results = {}
        
        for column in numeric_columns:
            series = df[column].dropna()
            
            if len(series) == 0:
                continue
                
            stats_dict = {
                "count": len(series),
                "mean": series.mean(),
                "median": series.median(),
                "mode": series.mode().iloc[0] if not series.mode().empty else None,
                "std": series.std(),
                "var": series.var(),
                "min": series.min(),
                "max": series.max(),
                "range": series.max() - series.min(),
                "iqr": series.quantile(0.75) - series.quantile(0.25),
                "missing_count": df[column].isna().sum(),
                "missing_ratio": df[column].isna().sum() / len(df)
            }
            
            if include_percentiles:
                percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                for p in percentiles:
                    stats_dict[f"p{p}"] = series.quantile(p / 100)
            
            if include_skewness:
                stats_dict["skewness"] = series.skew()
                stats_dict["skewness_interpretation"] = self._interpret_skewness(stats_dict["skewness"])
            
            if include_kurtosis:
                stats_dict["kurtosis"] = series.kurtosis()
                stats_dict["kurtosis_interpretation"] = self._interpret_kurtosis(stats_dict["kurtosis"])
            
            # 정규성 검정
            if len(series) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(series)
                stats_dict["normality_test"] = {
                    "shapiro_statistic": shapiro_stat,
                    "shapiro_p_value": shapiro_p,
                    "is_normal": shapiro_p > 0.05
                }
            
            results[column] = stats_dict
        
        return {
            "descriptive_statistics": results,
            "summary": {
                "total_columns": len(numeric_columns),
                "analyzed_columns": list(results.keys()),
                "total_rows": len(df)
            }
        }

    def correlation_analysis(
        self, 
        data: UnifiedDataFrame, 
        columns: Optional[List[str]] = None,
        methods: List[str] = ["pearson", "spearman", "kendall"]
    ) -> Dict[str, Any]:
        """고급 상관관계 분석을 수행합니다.
        
        Args:
            data: 분석할 데이터
            columns: 분석할 컬럼 목록
            methods: 상관관계 계산 방법들
            
        Returns:
            Dict[str, Any]: 상관관계 분석 결과
        """
        df = data.to_pandas()
        
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [col for col in columns if col in df.columns and df[col].dtype in [np.number, 'int64', 'float64']]
        
        if len(numeric_columns) < 2:
            return {"error": "상관관계 분석을 위해서는 최소 2개의 숫자 컬럼이 필요합니다."}
        
        results = {}
        
        for method in methods:
            if method == "pearson":
                corr_matrix = df[numeric_columns].corr(method='pearson')
            elif method == "spearman":
                corr_matrix = df[numeric_columns].corr(method='spearman')
            elif method == "kendall":
                corr_matrix = df[numeric_columns].corr(method='kendall')
            else:
                continue
            
            # 상관관계 강도 해석
            corr_strength = self._interpret_correlation_strength(corr_matrix)
            
            results[method] = {
                "correlation_matrix": corr_matrix.to_dict(),
                "correlation_strength": corr_strength,
                "high_correlations": self._find_high_correlations(corr_matrix, threshold=0.7)
            }
        
        return {
            "correlation_analysis": results,
            "summary": {
                "methods_used": list(results.keys()),
                "columns_analyzed": numeric_columns,
                "total_columns": len(numeric_columns)
            }
        }

    def distribution_analysis(
        self, 
        data: UnifiedDataFrame, 
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """분포 분석을 수행합니다.
        
        Args:
            data: 분석할 데이터
            columns: 분석할 컬럼 목록
            
        Returns:
            Dict[str, Any]: 분포 분석 결과
        """
        df = data.to_pandas()
        
        if columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [col for col in columns if col in df.columns and df[col].dtype in [np.number, 'int64', 'float64']]
        
        if not numeric_columns:
            return {"error": "분석할 숫자 컬럼이 없습니다."}
        
        results = {}
        
        for column in numeric_columns:
            series = df[column].dropna()
            
            if len(series) == 0:
                continue
            
            # 기본 분포 통계
            distribution_stats = {
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
                "jarque_bera_stat": stats.jarque_bera(series)[0],
                "jarque_bera_p_value": stats.jarque_bera(series)[1],
                "is_normal": stats.jarque_bera(series)[1] > 0.05
            }
            
            # 분포 유형 분류
            distribution_type = self._classify_distribution(series)
            distribution_stats["distribution_type"] = distribution_type
            
            # 히스토그램 데이터
            hist, bin_edges = np.histogram(series, bins=30)
            distribution_stats["histogram"] = {
                "counts": hist.tolist(),
                "bin_edges": bin_edges.tolist(),
                "bin_centers": [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
            }
            
            results[column] = distribution_stats
        
        return {
            "distribution_analysis": results,
            "summary": {
                "columns_analyzed": list(results.keys()),
                "total_columns": len(numeric_columns)
            }
        }

    def hypothesis_testing(
        self, 
        data: UnifiedDataFrame, 
        test_type: str,
        column1: str,
        column2: Optional[str] = None,
        group_column: Optional[str] = None,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """가설 검정을 수행합니다.
        
        Args:
            data: 분석할 데이터
            test_type: 검정 유형 ("t_test", "mann_whitney", "anova", "kruskal_wallis", "chi_square")
            column1: 첫 번째 컬럼
            column2: 두 번째 컬럼 (필요한 경우)
            group_column: 그룹 컬럼 (필요한 경우)
            alpha: 유의수준
            
        Returns:
            Dict[str, Any]: 가설 검정 결과
        """
        df = data.to_pandas()
        
        if test_type == "t_test":
            return self._t_test(df, column1, column2, alpha)
        elif test_type == "mann_whitney":
            return self._mann_whitney_test(df, column1, column2, alpha)
        elif test_type == "anova":
            return self._anova_test(df, column1, group_column, alpha)
        elif test_type == "kruskal_wallis":
            return self._kruskal_wallis_test(df, column1, group_column, alpha)
        elif test_type == "chi_square":
            return self._chi_square_test(df, column1, column2, alpha)
        else:
            return {"error": f"지원하지 않는 검정 유형: {test_type}"}

    def regression_analysis(
        self, 
        data: UnifiedDataFrame, 
        target_column: str,
        feature_columns: List[str],
        include_interaction: bool = False
    ) -> Dict[str, Any]:
        """회귀 분석을 수행합니다.
        
        Args:
            data: 분석할 데이터
            target_column: 종속 변수 컬럼
            feature_columns: 독립 변수 컬럼 목록
            include_interaction: 상호작용 항 포함 여부
            
        Returns:
            Dict[str, Any]: 회귀 분석 결과
        """
        df = data.to_pandas()
        
        # 결측값 제거
        analysis_df = df[[target_column] + feature_columns].dropna()
        
        if len(analysis_df) < 2:
            return {"error": "회귀 분석을 위한 충분한 데이터가 없습니다."}
        
        X = analysis_df[feature_columns]
        y = analysis_df[target_column]
        
        # 상호작용 항 추가
        if include_interaction and len(feature_columns) > 1:
            from sklearn.preprocessing import PolynomialFeatures
            poly = PolynomialFeatures(degree=2, include_bias=False)
            X = poly.fit_transform(X)
            feature_names = poly.get_feature_names_out(feature_columns)
        else:
            feature_names = feature_columns
        
        # 회귀 모델 훈련
        model = LinearRegression()
        model.fit(X, y)
        
        # 예측 및 평가
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # 잔차 분석
        residuals = y - y_pred
        
        results = {
            "model_coefficients": dict(zip(feature_names, model.coef_)),
            "intercept": model.intercept_,
            "r_squared": r2,
            "adjusted_r_squared": 1 - (1 - r2) * (len(y) - 1) / (len(y) - len(feature_columns) - 1),
            "residuals": {
                "mean": residuals.mean(),
                "std": residuals.std(),
                "min": residuals.min(),
                "max": residuals.max()
            },
            "sample_size": len(analysis_df),
            "feature_importance": dict(zip(feature_names, abs(model.coef_)))
        }
        
        return results

    def _interpret_skewness(self, skewness: float) -> str:
        """왜도를 해석합니다."""
        if abs(skewness) < 0.5:
            return "대칭에 가까움"
        elif abs(skewness) < 1:
            return "약간 치우침"
        else:
            return "심하게 치우침"

    def _interpret_kurtosis(self, kurtosis: float) -> str:
        """첨도를 해석합니다."""
        if kurtosis < -1:
            return "매우 평평함 (platykurtic)"
        elif kurtosis < 1:
            return "정상 첨도 (mesokurtic)"
        else:
            return "매우 뾰족함 (leptokurtic)"

    def _interpret_correlation_strength(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        """상관관계 강도를 해석합니다."""
        strength_matrix = corr_matrix.copy()
        
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix.columns)):
                corr = abs(corr_matrix.iloc[i, j])
                if corr < 0.1:
                    strength_matrix.iloc[i, j] = "무시할 수 있음"
                elif corr < 0.3:
                    strength_matrix.iloc[i, j] = "약함"
                elif corr < 0.5:
                    strength_matrix.iloc[i, j] = "보통"
                elif corr < 0.7:
                    strength_matrix.iloc[i, j] = "강함"
                else:
                    strength_matrix.iloc[i, j] = "매우 강함"
        
        return strength_matrix

    def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """높은 상관관계를 찾습니다."""
        high_correlations = []
        
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) >= threshold:
                    high_correlations.append({
                        "column1": corr_matrix.index[i],
                        "column2": corr_matrix.columns[j],
                        "correlation": corr,
                        "strength": "강함" if abs(corr) >= 0.7 else "보통"
                    })
        
        return high_correlations

    def _classify_distribution(self, series: pd.Series) -> str:
        """분포 유형을 분류합니다."""
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        if abs(skewness) < 0.5 and abs(kurtosis) < 1:
            return "정규분포에 가까움"
        elif skewness > 1:
            return "오른쪽 치우침 (right-skewed)"
        elif skewness < -1:
            return "왼쪽 치우침 (left-skewed)"
        elif kurtosis > 3:
            return "뾰족한 분포 (leptokurtic)"
        elif kurtosis < -1:
            return "평평한 분포 (platykurtic)"
        else:
            return "복합 분포"

    def _t_test(self, df: pd.DataFrame, column1: str, column2: str, alpha: float) -> Dict[str, Any]:
        """t-검정을 수행합니다."""
        group1 = df[column1].dropna()
        group2 = df[column2].dropna()
        
        statistic, p_value = ttest_ind(group1, group2)
        
        return {
            "test_type": "Independent t-test",
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < alpha,
            "group1_mean": group1.mean(),
            "group2_mean": group2.mean(),
            "group1_std": group1.std(),
            "group2_std": group2.std()
        }

    def _mann_whitney_test(self, df: pd.DataFrame, column1: str, column2: str, alpha: float) -> Dict[str, Any]:
        """Mann-Whitney U 검정을 수행합니다."""
        group1 = df[column1].dropna()
        group2 = df[column2].dropna()
        
        statistic, p_value = mannwhitneyu(group1, group2)
        
        return {
            "test_type": "Mann-Whitney U test",
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < alpha,
            "group1_median": group1.median(),
            "group2_median": group2.median()
        }

    def _anova_test(self, df: pd.DataFrame, column: str, group_column: str, alpha: float) -> Dict[str, Any]:
        """ANOVA 검정을 수행합니다."""
        groups = [group[column].dropna() for name, group in df.groupby(group_column)]
        
        statistic, p_value = f_oneway(*groups)
        
        group_stats = {}
        for name, group in df.groupby(group_column):
            group_stats[name] = {
                "mean": group[column].mean(),
                "std": group[column].std(),
                "count": len(group[column].dropna())
            }
        
        return {
            "test_type": "One-way ANOVA",
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < alpha,
            "group_statistics": group_stats
        }

    def _kruskal_wallis_test(self, df: pd.DataFrame, column: str, group_column: str, alpha: float) -> Dict[str, Any]:
        """Kruskal-Wallis 검정을 수행합니다."""
        groups = [group[column].dropna() for name, group in df.groupby(group_column)]
        
        statistic, p_value = kruskal(*groups)
        
        group_stats = {}
        for name, group in df.groupby(group_column):
            group_stats[name] = {
                "median": group[column].median(),
                "count": len(group[column].dropna())
            }
        
        return {
            "test_type": "Kruskal-Wallis test",
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < alpha,
            "group_statistics": group_stats
        }

    def _chi_square_test(self, df: pd.DataFrame, column1: str, column2: str, alpha: float) -> Dict[str, Any]:
        """카이제곱 검정을 수행합니다."""
        contingency_table = pd.crosstab(df[column1], df[column2])
        
        statistic, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            "test_type": "Chi-square test",
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < alpha,
            "degrees_of_freedom": dof,
            "contingency_table": contingency_table.to_dict(),
            "expected_frequencies": expected.tolist()
        }

    def __repr__(self) -> str:
        """문자열 표현."""
        return "AdvancedStatistics()"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
