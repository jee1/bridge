"""통계적 검정 모듈.

CA 마일스톤 3.2: 고급 통계 분석 및 시각화
통계적 검정 및 A/B 테스트 기능을 제공하는 모듈입니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    kruskal,
    mannwhitneyu,
    pearsonr,
    spearmanr,
    ttest_1samp,
    ttest_ind,
    ttest_rel,
    wilcoxon
)

from bridge.analytics.core.data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)


class StatisticalTests:
    """통계적 검정 클래스.
    
    다양한 통계적 검정을 수행합니다.
    """

    def __init__(self, alpha: float = 0.05):
        """StatisticalTests를 초기화합니다.
        
        Args:
            alpha: 유의수준 (기본값: 0.05)
        """
        self.alpha = alpha

    def t_test(
        self,
        data: UnifiedDataFrame,
        test_type: str,
        column: str,
        value: Optional[float] = None,
        group_column: Optional[str] = None,
        paired: bool = False
    ) -> Dict[str, Any]:
        """t-검정을 수행합니다.
        
        Args:
            data: 분석할 데이터
            test_type: 검정 유형 ("one_sample", "two_sample", "paired")
            column: 분석할 컬럼
            value: 일표본 t-검정에서의 가설값
            group_column: 두표본 t-검정에서의 그룹 컬럼
            paired: 대응표본 t-검정 여부
            
        Returns:
            Dict[str, Any]: t-검정 결과
        """
        df = data.to_pandas()
        
        if test_type == "one_sample":
            return self._one_sample_t_test(df, column, value)
        elif test_type == "two_sample":
            return self._two_sample_t_test(df, column, group_column)
        elif test_type == "paired":
            return self._paired_t_test(df, column, group_column)
        else:
            return {"error": f"지원하지 않는 t-검정 유형: {test_type}"}

    def non_parametric_test(
        self,
        data: UnifiedDataFrame,
        test_type: str,
        column: str,
        group_column: Optional[str] = None,
        value: Optional[float] = None
    ) -> Dict[str, Any]:
        """비모수 검정을 수행합니다.
        
        Args:
            data: 분석할 데이터
            test_type: 검정 유형 ("mann_whitney", "wilcoxon", "kruskal_wallis")
            column: 분석할 컬럼
            group_column: 그룹 컬럼
            value: 가설값 (wilcoxon 검정에서)
            
        Returns:
            Dict[str, Any]: 비모수 검정 결과
        """
        df = data.to_pandas()
        
        if test_type == "mann_whitney":
            return self._mann_whitney_test(df, column, group_column)
        elif test_type == "wilcoxon":
            return self._wilcoxon_test(df, column, value)
        elif test_type == "kruskal_wallis":
            return self._kruskal_wallis_test(df, column, group_column)
        else:
            return {"error": f"지원하지 않는 비모수 검정 유형: {test_type}"}

    def anova_test(
        self,
        data: UnifiedDataFrame,
        test_type: str,
        column: str,
        group_column: str,
        block_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """ANOVA 검정을 수행합니다.
        
        Args:
            data: 분석할 데이터
            test_type: 검정 유형 ("one_way", "two_way")
            column: 분석할 컬럼
            group_column: 그룹 컬럼
            block_column: 블록 컬럼 (이원분산분석에서)
            
        Returns:
            Dict[str, Any]: ANOVA 검정 결과
        """
        df = data.to_pandas()
        
        if test_type == "one_way":
            return self._one_way_anova(df, column, group_column)
        elif test_type == "two_way":
            return self._two_way_anova(df, column, group_column, block_column)
        else:
            return {"error": f"지원하지 않는 ANOVA 유형: {test_type}"}

    def chi_square_test(
        self,
        data: UnifiedDataFrame,
        column1: str,
        column2: str,
        test_type: str = "independence"
    ) -> Dict[str, Any]:
        """카이제곱 검정을 수행합니다.
        
        Args:
            data: 분석할 데이터
            column1: 첫 번째 컬럼
            column2: 두 번째 컬럼
            test_type: 검정 유형 ("independence", "goodness_of_fit")
            
        Returns:
            Dict[str, Any]: 카이제곱 검정 결과
        """
        df = data.to_pandas()
        
        if test_type == "independence":
            return self._chi_square_independence_test(df, column1, column2)
        elif test_type == "goodness_of_fit":
            return self._chi_square_goodness_of_fit_test(df, column1, column2)
        else:
            return {"error": f"지원하지 않는 카이제곱 검정 유형: {test_type}"}

    def correlation_test(
        self,
        data: UnifiedDataFrame,
        column1: str,
        column2: str,
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """상관관계 검정을 수행합니다.
        
        Args:
            data: 분석할 데이터
            column1: 첫 번째 컬럼
            column2: 두 번째 컬럼
            method: 상관관계 방법 ("pearson", "spearman")
            
        Returns:
            Dict[str, Any]: 상관관계 검정 결과
        """
        df = data.to_pandas()
        
        if method == "pearson":
            return self._pearson_correlation_test(df, column1, column2)
        elif method == "spearman":
            return self._spearman_correlation_test(df, column1, column2)
        else:
            return {"error": f"지원하지 않는 상관관계 방법: {method}"}

    def ab_test_analysis(
        self,
        data: UnifiedDataFrame,
        group_column: str,
        metric_column: str,
        test_type: str = "t_test",
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """A/B 테스트 분석을 수행합니다.
        
        Args:
            data: 분석할 데이터
            group_column: 그룹 컬럼 (A, B)
            metric_column: 지표 컬럼
            test_type: 검정 유형 ("t_test", "mann_whitney")
            confidence_level: 신뢰수준
            
        Returns:
            Dict[str, Any]: A/B 테스트 분석 결과
        """
        df = data.to_pandas()
        
        # 그룹별 데이터 분리
        groups = df[group_column].unique()
        if len(groups) != 2:
            return {"error": "A/B 테스트를 위해서는 정확히 2개의 그룹이 필요합니다."}
        
        group_a = df[df[group_column] == groups[0]][metric_column].dropna()
        group_b = df[df[group_column] == groups[1]][metric_column].dropna()
        
        # 기본 통계
        stats_a = {
            "mean": group_a.mean(),
            "std": group_a.std(),
            "count": len(group_a),
            "median": group_a.median()
        }
        
        stats_b = {
            "mean": group_b.mean(),
            "std": group_b.std(),
            "count": len(group_b),
            "median": group_b.median()
        }
        
        # 효과 크기 계산
        pooled_std = np.sqrt(((len(group_a) - 1) * group_a.var() + (len(group_b) - 1) * group_b.var()) / 
                            (len(group_a) + len(group_b) - 2))
        cohens_d = (group_b.mean() - group_a.mean()) / pooled_std
        
        # 검정 수행
        if test_type == "t_test":
            test_result = self._two_sample_t_test(df, metric_column, group_column)
        elif test_type == "mann_whitney":
            test_result = self._mann_whitney_test(df, metric_column, group_column)
        else:
            return {"error": f"지원하지 않는 검정 유형: {test_type}"}
        
        # 신뢰구간 계산
        alpha = 1 - confidence_level
        se_diff = pooled_std * np.sqrt(1/len(group_a) + 1/len(group_b))
        margin_error = stats.t.ppf(1 - alpha/2, len(group_a) + len(group_b) - 2) * se_diff
        mean_diff = group_b.mean() - group_a.mean()
        
        confidence_interval = {
            "lower": mean_diff - margin_error,
            "upper": mean_diff + margin_error,
            "level": confidence_level
        }
        
        # 실용적 유의성 판단
        practical_significance = abs(cohens_d) >= 0.2  # Cohen's d >= 0.2 (작은 효과)
        
        return {
            "test_type": test_type,
            "groups": {
                groups[0]: stats_a,
                groups[1]: stats_b
            },
            "test_result": test_result,
            "effect_size": {
                "cohens_d": cohens_d,
                "interpretation": self._interpret_cohens_d(cohens_d)
            },
            "confidence_interval": confidence_interval,
            "practical_significance": practical_significance,
            "recommendation": self._get_ab_test_recommendation(
                test_result.get("significant", False),
                practical_significance,
                cohens_d
            )
        }

    def power_analysis(
        self,
        effect_size: float,
        alpha: float = 0.05,
        power: float = 0.8,
        test_type: str = "t_test"
    ) -> Dict[str, Any]:
        """검정력을 분석합니다.
        
        Args:
            effect_size: 효과 크기 (Cohen's d)
            alpha: 유의수준
            power: 검정력
            test_type: 검정 유형
            
        Returns:
            Dict[str, Any]: 검정력 분석 결과
        """
        from statsmodels.stats.power import ttest_power
        
        if test_type == "t_test":
            # 필요한 표본 크기 계산
            n_required = ttest_power(effect_size, alpha=alpha, power=power, alternative='two-sided')
            
            # 현재 검정력으로 계산된 표본 크기
            current_power = ttest_power(effect_size, alpha=alpha, nobs=n_required, alternative='two-sided')
            
            return {
                "effect_size": effect_size,
                "alpha": alpha,
                "target_power": power,
                "required_sample_size": int(np.ceil(n_required)),
                "actual_power": current_power,
                "test_type": test_type,
                "interpretation": self._interpret_power_analysis(effect_size, n_required, current_power)
            }
        else:
            return {"error": f"지원하지 않는 검정 유형: {test_type}"}

    def _one_sample_t_test(self, df: pd.DataFrame, column: str, value: float) -> Dict[str, Any]:
        """일표본 t-검정을 수행합니다."""
        data = df[column].dropna()
        
        if len(data) == 0:
            return {"error": f"컬럼 '{column}'에 유효한 데이터가 없습니다."}
        
        statistic, p_value = ttest_1samp(data, value)
        
        return {
            "test_type": "One-sample t-test",
            "column": column,
            "hypothesized_value": value,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "sample_mean": data.mean(),
            "sample_std": data.std(),
            "sample_size": len(data),
            "degrees_of_freedom": len(data) - 1
        }

    def _two_sample_t_test(self, df: pd.DataFrame, column: str, group_column: str) -> Dict[str, Any]:
        """두표본 t-검정을 수행합니다."""
        groups = df.groupby(group_column)[column].apply(lambda x: x.dropna())
        
        if len(groups) != 2:
            return {"error": f"그룹 컬럼 '{group_column}'에 정확히 2개의 그룹이 필요합니다."}
        
        group1, group2 = groups.iloc[0], groups.iloc[1]
        
        if len(group1) == 0 or len(group2) == 0:
            return {"error": "그룹에 유효한 데이터가 없습니다."}
        
        statistic, p_value = ttest_ind(group1, group2)
        
        return {
            "test_type": "Two-sample t-test",
            "column": column,
            "group_column": group_column,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "group1": {
                "name": groups.index[0],
                "mean": group1.mean(),
                "std": group1.std(),
                "count": len(group1)
            },
            "group2": {
                "name": groups.index[1],
                "mean": group2.mean(),
                "std": group2.std(),
                "count": len(group2)
            },
            "degrees_of_freedom": len(group1) + len(group2) - 2
        }

    def _paired_t_test(self, df: pd.DataFrame, column: str, group_column: str) -> Dict[str, Any]:
        """대응표본 t-검정을 수행합니다."""
        groups = df.groupby(group_column)[column].apply(lambda x: x.dropna())
        
        if len(groups) != 2:
            return {"error": f"그룹 컬럼 '{group_column}'에 정확히 2개의 그룹이 필요합니다."}
        
        group1, group2 = groups.iloc[0], groups.iloc[1]
        
        if len(group1) != len(group2):
            return {"error": "대응표본 t-검정을 위해서는 두 그룹의 크기가 같아야 합니다."}
        
        statistic, p_value = ttest_rel(group1, group2)
        
        return {
            "test_type": "Paired t-test",
            "column": column,
            "group_column": group_column,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "group1": {
                "name": groups.index[0],
                "mean": group1.mean(),
                "std": group1.std(),
                "count": len(group1)
            },
            "group2": {
                "name": groups.index[1],
                "mean": group2.mean(),
                "std": group2.std(),
                "count": len(group2)
            },
            "difference": {
                "mean": (group2 - group1).mean(),
                "std": (group2 - group1).std()
            },
            "degrees_of_freedom": len(group1) - 1
        }

    def _mann_whitney_test(self, df: pd.DataFrame, column: str, group_column: str) -> Dict[str, Any]:
        """Mann-Whitney U 검정을 수행합니다."""
        groups = df.groupby(group_column)[column].apply(lambda x: x.dropna())
        
        if len(groups) != 2:
            return {"error": f"그룹 컬럼 '{group_column}'에 정확히 2개의 그룹이 필요합니다."}
        
        group1, group2 = groups.iloc[0], groups.iloc[1]
        
        if len(group1) == 0 or len(group2) == 0:
            return {"error": "그룹에 유효한 데이터가 없습니다."}
        
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        
        return {
            "test_type": "Mann-Whitney U test",
            "column": column,
            "group_column": group_column,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "group1": {
                "name": groups.index[0],
                "median": group1.median(),
                "count": len(group1)
            },
            "group2": {
                "name": groups.index[1],
                "median": group2.median(),
                "count": len(group2)
            }
        }

    def _wilcoxon_test(self, df: pd.DataFrame, column: str, value: float) -> Dict[str, Any]:
        """Wilcoxon 부호순위 검정을 수행합니다."""
        data = df[column].dropna()
        
        if len(data) == 0:
            return {"error": f"컬럼 '{column}'에 유효한 데이터가 없습니다."}
        
        statistic, p_value = wilcoxon(data - value)
        
        return {
            "test_type": "Wilcoxon signed-rank test",
            "column": column,
            "hypothesized_value": value,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "sample_median": data.median(),
            "sample_size": len(data)
        }

    def _kruskal_wallis_test(self, df: pd.DataFrame, column: str, group_column: str) -> Dict[str, Any]:
        """Kruskal-Wallis 검정을 수행합니다."""
        groups = [group[column].dropna() for name, group in df.groupby(group_column)]
        
        if len(groups) < 2:
            return {"error": f"그룹 컬럼 '{group_column}'에 최소 2개의 그룹이 필요합니다."}
        
        statistic, p_value = kruskal(*groups)
        
        group_stats = {}
        for name, group in df.groupby(group_column):
            group_data = group[column].dropna()
            group_stats[name] = {
                "median": group_data.median(),
                "count": len(group_data)
            }
        
        return {
            "test_type": "Kruskal-Wallis test",
            "column": column,
            "group_column": group_column,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "group_statistics": group_stats,
            "degrees_of_freedom": len(groups) - 1
        }

    def _one_way_anova(self, df: pd.DataFrame, column: str, group_column: str) -> Dict[str, Any]:
        """일원분산분석을 수행합니다."""
        groups = [group[column].dropna() for name, group in df.groupby(group_column)]
        
        if len(groups) < 2:
            return {"error": f"그룹 컬럼 '{group_column}'에 최소 2개의 그룹이 필요합니다."}
        
        statistic, p_value = f_oneway(*groups)
        
        group_stats = {}
        for name, group in df.groupby(group_column):
            group_data = group[column].dropna()
            group_stats[name] = {
                "mean": group_data.mean(),
                "std": group_data.std(),
                "count": len(group_data)
            }
        
        return {
            "test_type": "One-way ANOVA",
            "column": column,
            "group_column": group_column,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "group_statistics": group_stats,
            "degrees_of_freedom": {
                "between_groups": len(groups) - 1,
                "within_groups": sum(len(group) for group in groups) - len(groups)
            }
        }

    def _two_way_anova(self, df: pd.DataFrame, column: str, group_column: str, block_column: str) -> Dict[str, Any]:
        """이원분산분석을 수행합니다."""
        # 간단한 이원분산분석 구현 (실제로는 더 복잡한 구현이 필요)
        return {"error": "이원분산분석은 아직 구현되지 않았습니다."}

    def _chi_square_independence_test(self, df: pd.DataFrame, column1: str, column2: str) -> Dict[str, Any]:
        """카이제곱 독립성 검정을 수행합니다."""
        contingency_table = pd.crosstab(df[column1], df[column2])
        
        statistic, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            "test_type": "Chi-square test of independence",
            "column1": column1,
            "column2": column2,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "degrees_of_freedom": dof,
            "contingency_table": contingency_table.to_dict(),
            "expected_frequencies": expected.tolist()
        }

    def _chi_square_goodness_of_fit_test(self, df: pd.DataFrame, column1: str, column2: str) -> Dict[str, Any]:
        """카이제곱 적합도 검정을 수행합니다."""
        observed = df[column1].value_counts()
        expected = df[column2].value_counts()
        
        # 빈도 정규화
        expected = expected * (observed.sum() / expected.sum())
        
        statistic, p_value = stats.chisquare(observed, expected)
        
        return {
            "test_type": "Chi-square goodness of fit test",
            "column1": column1,
            "column2": column2,
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "observed": observed.to_dict(),
            "expected": expected.to_dict(),
            "degrees_of_freedom": len(observed) - 1
        }

    def _pearson_correlation_test(self, df: pd.DataFrame, column1: str, column2: str) -> Dict[str, Any]:
        """피어슨 상관관계 검정을 수행합니다."""
        data1 = df[column1].dropna()
        data2 = df[column2].dropna()
        
        if len(data1) != len(data2):
            return {"error": "두 컬럼의 길이가 같아야 합니다."}
        
        correlation, p_value = pearsonr(data1, data2)
        
        return {
            "test_type": "Pearson correlation test",
            "column1": column1,
            "column2": column2,
            "correlation": correlation,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "sample_size": len(data1),
            "interpretation": self._interpret_correlation(correlation)
        }

    def _spearman_correlation_test(self, df: pd.DataFrame, column1: str, column2: str) -> Dict[str, Any]:
        """스피어만 상관관계 검정을 수행합니다."""
        data1 = df[column1].dropna()
        data2 = df[column2].dropna()
        
        if len(data1) != len(data2):
            return {"error": "두 컬럼의 길이가 같아야 합니다."}
        
        correlation, p_value = spearmanr(data1, data2)
        
        return {
            "test_type": "Spearman correlation test",
            "column1": column1,
            "column2": column2,
            "correlation": correlation,
            "p_value": p_value,
            "significant": p_value < self.alpha,
            "sample_size": len(data1),
            "interpretation": self._interpret_correlation(correlation)
        }

    def _interpret_cohens_d(self, d: float) -> str:
        """Cohen's d를 해석합니다."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "작은 효과"
        elif abs_d < 0.5:
            return "중간 효과"
        elif abs_d < 0.8:
            return "큰 효과"
        else:
            return "매우 큰 효과"

    def _interpret_correlation(self, r: float) -> str:
        """상관관계를 해석합니다."""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "무시할 수 있음"
        elif abs_r < 0.3:
            return "약한 상관관계"
        elif abs_r < 0.5:
            return "보통 상관관계"
        elif abs_r < 0.7:
            return "강한 상관관계"
        else:
            return "매우 강한 상관관계"

    def _get_ab_test_recommendation(self, significant: bool, practical_significance: bool, cohens_d: float) -> str:
        """A/B 테스트 권장사항을 제공합니다."""
        if significant and practical_significance:
            return "통계적으로 유의하고 실용적으로도 의미있는 차이가 있습니다. 변경사항을 적용하는 것을 권장합니다."
        elif significant and not practical_significance:
            return "통계적으로는 유의하지만 실용적으로는 의미가 작습니다. 추가 검토가 필요합니다."
        elif not significant and practical_significance:
            return "실용적으로는 의미가 있지만 통계적으로는 유의하지 않습니다. 표본 크기를 늘려 재검정을 권장합니다."
        else:
            return "통계적으로도 실용적으로도 의미있는 차이가 없습니다. 현재 상태를 유지하는 것을 권장합니다."

    def _interpret_power_analysis(self, effect_size: float, n_required: float, actual_power: float) -> str:
        """검정력 분석 결과를 해석합니다."""
        if actual_power >= 0.8:
            return f"검정력이 충분합니다 (power = {actual_power:.3f}). 효과 크기 {effect_size:.3f}를 검출할 수 있습니다."
        elif actual_power >= 0.5:
            return f"검정력이 보통입니다 (power = {actual_power:.3f}). 더 큰 표본이 필요할 수 있습니다."
        else:
            return f"검정력이 부족합니다 (power = {actual_power:.3f}). 표본 크기를 늘리거나 효과 크기를 크게 해야 합니다."

    def __repr__(self) -> str:
        """문자열 표현."""
        return f"StatisticalTests(alpha={self.alpha})"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
