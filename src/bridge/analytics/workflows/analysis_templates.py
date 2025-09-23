"""분석 템플릿 시스템.

CA 마일스톤 3.4: 워크플로 및 자동화 시스템
- 고객 분석, 매출 분석, A/B 테스트 등 사전 정의된 템플릿
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from bridge.analytics.core.data_integration import UnifiedDataFrame
from bridge.analytics.core.statistics import StatisticsAnalyzer
from bridge.analytics.quality.comprehensive_metrics import ComprehensiveQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class AnalysisTemplate:
    """분석 템플릿을 정의하는 클래스"""

    name: str
    description: str
    required_columns: List[str]
    optional_columns: List[str]
    parameters: Dict[str, Any]
    output_format: str


@dataclass
class AnalysisResult:
    """분석 결과를 담는 데이터 클래스"""

    template_name: str
    success: bool
    results: Dict[str, Any]
    visualizations: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]


class BaseAnalysisTemplate(ABC):
    """분석 템플릿 기본 클래스"""

    def __init__(self):
        """분석 템플릿 초기화"""
        self.logger = logging.getLogger(__name__)
        self.statistics_analyzer = StatisticsAnalyzer()
        self.quality_calculator = ComprehensiveQualityMetrics()

    @abstractmethod
    def get_template_info(self) -> AnalysisTemplate:
        """템플릿 정보 반환"""
        pass

    @abstractmethod
    def validate_data(self, data: UnifiedDataFrame) -> bool:
        """데이터 검증"""
        pass

    @abstractmethod
    def execute_analysis(
        self, data: UnifiedDataFrame, parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """분석 실행"""
        pass


class CustomerAnalysisTemplate(BaseAnalysisTemplate):
    """고객 분석 템플릿"""

    def get_template_info(self) -> AnalysisTemplate:
        """템플릿 정보 반환"""
        return AnalysisTemplate(
            name="customer_analysis",
            description="고객 세그멘테이션 및 행동 분석",
            required_columns=["customer_id", "purchase_amount", "purchase_date"],
            optional_columns=["age", "gender", "location", "product_category"],
            parameters={"segmentation_method": "rfm", "clusters": 5, "time_period": "12M"},
            output_format="json",
        )

    def validate_data(self, data: UnifiedDataFrame) -> bool:
        """데이터 검증"""
        try:
            df = data.to_pandas()
            required_cols = ["customer_id", "purchase_amount", "purchase_date"]

            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"필수 컬럼 {col}이 없습니다.")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"데이터 검증 실패: {e}")
            return False

    def execute_analysis(
        self, data: UnifiedDataFrame, parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """고객 분석 실행"""
        try:
            if not self.validate_data(data):
                return AnalysisResult(
                    template_name="customer_analysis",
                    success=False,
                    results={},
                    visualizations=[],
                    recommendations=["데이터 검증에 실패했습니다."],
                    metadata={"error": "데이터 검증 실패"},
                )

            df = data.to_pandas()

            # RFM 분석
            rfm_results = self._perform_rfm_analysis(df, parameters)

            # 고객 세그멘테이션
            segmentation_results = self._perform_customer_segmentation(df, parameters)

            # 구매 패턴 분석
            pattern_results = self._analyze_purchase_patterns(df, parameters)

            # 시각화 생성
            visualizations = self._create_visualizations(df, rfm_results, segmentation_results)

            # 권장사항 생성
            recommendations = self._generate_recommendations(rfm_results, segmentation_results)

            return AnalysisResult(
                template_name="customer_analysis",
                success=True,
                results={
                    "rfm_analysis": rfm_results,
                    "segmentation": segmentation_results,
                    "purchase_patterns": pattern_results,
                },
                visualizations=visualizations,
                recommendations=recommendations,
                metadata={
                    "total_customers": len(df["customer_id"].unique()),
                    "analysis_period": parameters.get("time_period", "12M"),
                    "segmentation_method": parameters.get("segmentation_method", "rfm"),
                },
            )

        except Exception as e:
            self.logger.error(f"고객 분석 실행 실패: {e}")
            return AnalysisResult(
                template_name="customer_analysis",
                success=False,
                results={},
                visualizations=[],
                recommendations=[f"분석 실행 실패: {str(e)}"],
                metadata={"error": str(e)},
            )

    def _perform_rfm_analysis(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """RFM 분석 수행"""
        try:
            # 최근 구매일 계산
            df["purchase_date"] = pd.to_datetime(df["purchase_date"])
            recency = df.groupby("customer_id")["purchase_date"].max()

            # 구매 빈도 계산
            frequency = df.groupby("customer_id").size()

            # 구매 금액 계산
            monetary = df.groupby("customer_id")["purchase_amount"].sum()

            # RFM 점수 계산
            rfm_df = pd.DataFrame(
                {
                    "customer_id": recency.index,
                    "recency": (df["purchase_date"].max() - recency).dt.days,
                    "frequency": frequency,
                    "monetary": monetary,
                }
            )

            # RFM 점수 정규화 (1-5 스케일)
            # 중복된 값이 있을 경우를 처리하기 위해 try-except 사용
            try:
                rfm_df["r_score"] = pd.qcut(rfm_df["recency"], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
            except ValueError:
                # recency 값이 모두 동일한 경우
                rfm_df["r_score"] = 3  # 중간값으로 설정
            
            try:
                rfm_df["f_score"] = pd.qcut(rfm_df["frequency"], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            except ValueError:
                # frequency 값이 모두 동일한 경우
                rfm_df["f_score"] = 3  # 중간값으로 설정
            
            try:
                rfm_df["m_score"] = pd.qcut(rfm_df["monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            except ValueError:
                # monetary 값이 모두 동일한 경우
                rfm_df["m_score"] = 3  # 중간값으로 설정

            # RFM 점수 합계
            rfm_df["rfm_score"] = (
                rfm_df["r_score"].astype(int)
                + rfm_df["f_score"].astype(int)
                + rfm_df["m_score"].astype(int)
            )

            return {
                "rfm_data": rfm_df.to_dict("records"),
                "recency_stats": rfm_df["recency"].describe().to_dict(),
                "frequency_stats": rfm_df["frequency"].describe().to_dict(),
                "monetary_stats": rfm_df["monetary"].describe().to_dict(),
                "rfm_score_distribution": rfm_df["rfm_score"].value_counts().to_dict(),
            }

        except Exception as e:
            self.logger.error(f"RFM 분석 실패: {e}")
            return {"error": str(e)}

    def _perform_customer_segmentation(
        self, df: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """고객 세그멘테이션 수행"""
        try:
            # 고객별 집계
            customer_summary = (
                df.groupby("customer_id")
                .agg({
                    "purchase_amount": ["sum", "mean", "count"], 
                    "purchase_date": ["min", "max"]
                })
                .reset_index()
            )

            # 컬럼명을 평평하게 만들기 (튜플 형태 제거)
            customer_summary.columns = [
                "customer_id",
                "total_spent",
                "avg_purchase", 
                "purchase_count",
                "first_purchase",
                "last_purchase",
            ]

            # 세그멘테이션 규칙 적용
            segments = []
            for _, row in customer_summary.iterrows():
                if row["total_spent"] > customer_summary["total_spent"].quantile(0.8):
                    if row["purchase_count"] > customer_summary["purchase_count"].quantile(0.8):
                        segments.append("VIP")
                    else:
                        segments.append("High Value")
                elif row["total_spent"] > customer_summary["total_spent"].quantile(0.5):
                    segments.append("Medium Value")
                else:
                    segments.append("Low Value")

            customer_summary["segment"] = segments

            # 세그먼트별 통계 계산 (튜플 컬럼명 문제 해결)
            segment_stats = customer_summary.groupby("segment").agg({
                "total_spent": ["mean", "std"], 
                "purchase_count": ["mean", "std"]
            })
            
            # 컬럼명을 평평하게 만들기
            segment_stats.columns = [
                "total_spent_mean", "total_spent_std",
                "purchase_count_mean", "purchase_count_std"
            ]
            
            return {
                "segmentation_data": customer_summary.to_dict("records"),
                "segment_distribution": pd.Series(segments).value_counts().to_dict(),
                "segment_stats": segment_stats.to_dict(),
            }

        except Exception as e:
            self.logger.error(f"고객 세그멘테이션 실패: {e}")
            return {"error": str(e)}

    def _analyze_purchase_patterns(
        self, df: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """구매 패턴 분석"""
        try:
            df["purchase_date"] = pd.to_datetime(df["purchase_date"])
            df["month"] = df["purchase_date"].dt.month
            df["day_of_week"] = df["purchase_date"].dt.dayofweek

            # 월별 구매 패턴
            monthly_pattern = (
                df.groupby("month")["purchase_amount"].agg(["sum", "mean", "count"]).to_dict()
            )

            # 요일별 구매 패턴
            daily_pattern = (
                df.groupby("day_of_week")["purchase_amount"].agg(["sum", "mean", "count"]).to_dict()
            )

            # 고객별 구매 패턴
            customer_patterns_df = (
                df.groupby("customer_id")
                .agg({"purchase_amount": ["sum", "mean", "std"], "purchase_date": "count"})
            )
            
            # 컬럼명을 평평하게 만들기
            customer_patterns_df.columns = [
                "purchase_amount_sum", "purchase_amount_mean", "purchase_amount_std", "purchase_count"
            ]
            
            customer_patterns = customer_patterns_df.to_dict()

            return {
                "monthly_patterns": monthly_pattern,
                "daily_patterns": daily_pattern,
                "customer_patterns": customer_patterns,
            }

        except Exception as e:
            self.logger.error(f"구매 패턴 분석 실패: {e}")
            return {"error": str(e)}

    def _create_visualizations(
        self, df: pd.DataFrame, rfm_results: Dict[str, Any], segmentation_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """시각화 생성"""
        try:
            visualizations = []

            # RFM 분포 차트
            if "rfm_data" in rfm_results:
                rfm_df = pd.DataFrame(rfm_results["rfm_data"])
                visualizations.append(
                    {
                        "type": "histogram",
                        "title": "RFM 점수 분포",
                        "data": rfm_df["rfm_score"].value_counts().to_dict(),
                        "x_label": "RFM 점수",
                        "y_label": "고객 수",
                    }
                )

            # 세그먼트 분포 차트
            if "segment_distribution" in segmentation_results:
                visualizations.append(
                    {
                        "type": "pie",
                        "title": "고객 세그먼트 분포",
                        "data": segmentation_results["segment_distribution"],
                    }
                )

            return visualizations

        except Exception as e:
            self.logger.error(f"시각화 생성 실패: {e}")
            return []

    def _generate_recommendations(
        self, rfm_results: Dict[str, Any], segmentation_results: Dict[str, Any]
    ) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        try:
            # RFM 기반 권장사항
            if "rfm_score_distribution" in rfm_results:
                rfm_dist = rfm_results["rfm_score_distribution"]
                high_value_customers = sum(
                    count for score, count in rfm_dist.items() if score >= 12
                )
                total_customers = sum(rfm_dist.values())

                if high_value_customers / total_customers < 0.2:
                    recommendations.append(
                        "고가치 고객 비율이 낮습니다. 고객 유지 전략을 강화하세요."
                    )

            # 세그먼트 기반 권장사항
            if "segment_distribution" in segmentation_results:
                segment_dist = segmentation_results["segment_distribution"]
                if "Low Value" in segment_dist and segment_dist["Low Value"] > segment_dist.get(
                    "VIP", 0
                ):
                    recommendations.append(
                        "저가치 고객이 많습니다. 고객 가치 향상 프로그램을 도입하세요."
                    )

            return recommendations

        except Exception as e:
            self.logger.error(f"권장사항 생성 실패: {e}")
            return ["권장사항 생성 중 오류가 발생했습니다."]


class SalesAnalysisTemplate(BaseAnalysisTemplate):
    """매출 분석 템플릿"""

    def get_template_info(self) -> AnalysisTemplate:
        """템플릿 정보 반환"""
        return AnalysisTemplate(
            name="sales_analysis",
            description="매출 트렌드 및 계절성 분석",
            required_columns=["sales_date", "sales_amount"],
            optional_columns=["product_category", "region", "salesperson"],
            parameters={"time_period": "12M", "trend_analysis": True, "seasonality_analysis": True},
            output_format="json",
        )

    def validate_data(self, data: UnifiedDataFrame) -> bool:
        """데이터 검증"""
        try:
            df = data.to_pandas()
            required_cols = ["sales_date", "sales_amount"]

            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"필수 컬럼 {col}이 없습니다.")
                    return False

            return True
        except Exception as e:
            self.logger.error(f"데이터 검증 실패: {e}")
            return False

    def execute_analysis(
        self, data: UnifiedDataFrame, parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """매출 분석 실행"""
        try:
            if not self.validate_data(data):
                return AnalysisResult(
                    template_name="sales_analysis",
                    success=False,
                    results={},
                    visualizations=[],
                    recommendations=["데이터 검증에 실패했습니다."],
                    metadata={"error": "데이터 검증 실패"},
                )

            df = data.to_pandas()

            # 매출 트렌드 분석
            trend_results = self._analyze_sales_trends(df, parameters)

            # 계절성 분석
            seasonality_results = self._analyze_seasonality(df, parameters)

            # 카테고리별 분석
            category_results = self._analyze_by_category(df, parameters)

            # 시각화 생성
            visualizations = self._create_sales_visualizations(
                df, trend_results, seasonality_results
            )

            # 권장사항 생성
            recommendations = self._generate_sales_recommendations(
                trend_results, seasonality_results
            )

            return AnalysisResult(
                template_name="sales_analysis",
                success=True,
                results={
                    "trend_analysis": trend_results,
                    "seasonality_analysis": seasonality_results,
                    "category_analysis": category_results,
                },
                visualizations=visualizations,
                recommendations=recommendations,
                metadata={
                    "total_sales": df["sales_amount"].sum(),
                    "analysis_period": parameters.get("time_period", "12M"),
                    "total_records": len(df),
                },
            )

        except Exception as e:
            self.logger.error(f"매출 분석 실행 실패: {e}")
            return AnalysisResult(
                template_name="sales_analysis",
                success=False,
                results={},
                visualizations=[],
                recommendations=[f"분석 실행 실패: {str(e)}"],
                metadata={"error": str(e)},
            )

    def _analyze_sales_trends(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """매출 트렌드 분석"""
        try:
            df["sales_date"] = pd.to_datetime(df["sales_date"])
            df["month"] = df["sales_date"].dt.to_period("M")

            # 월별 매출 집계
            monthly_sales = df.groupby("month")["sales_amount"].sum().reset_index()
            monthly_sales["month"] = monthly_sales["month"].astype(str)

            # 트렌드 계산 (선형 회귀)
            X = np.array(range(len(monthly_sales))).reshape(-1, 1)
            y = monthly_sales["sales_amount"].values

            from sklearn.linear_model import LinearRegression

            model = LinearRegression()
            model.fit(X, y)

            trend_slope = model.coef_[0]
            trend_direction = "상승" if trend_slope > 0 else "하락" if trend_slope < 0 else "안정"

            return {
                "monthly_sales": monthly_sales.to_dict("records"),
                "trend_slope": trend_slope,
                "trend_direction": trend_direction,
                "r_squared": model.score(X, y),
            }

        except Exception as e:
            self.logger.error(f"매출 트렌드 분석 실패: {e}")
            return {"error": str(e)}

    def _analyze_seasonality(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """계절성 분석"""
        try:
            df["sales_date"] = pd.to_datetime(df["sales_date"])
            df["month"] = df["sales_date"].dt.month
            df["quarter"] = df["sales_date"].dt.quarter

            # 월별 계절성
            monthly_seasonality = df.groupby("month")["sales_amount"].mean().to_dict()

            # 분기별 계절성
            quarterly_seasonality = df.groupby("quarter")["sales_amount"].mean().to_dict()

            # 계절성 지수 계산
            overall_mean = df["sales_amount"].mean()
            seasonal_indices = {
                month: sales / overall_mean for month, sales in monthly_seasonality.items()
            }

            return {
                "monthly_seasonality": monthly_seasonality,
                "quarterly_seasonality": quarterly_seasonality,
                "seasonal_indices": seasonal_indices,
            }

        except Exception as e:
            self.logger.error(f"계절성 분석 실패: {e}")
            return {"error": str(e)}

    def _analyze_by_category(self, df: pd.DataFrame, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """카테고리별 분석"""
        try:
            if "product_category" not in df.columns:
                return {"message": "product_category 컬럼이 없습니다."}

            category_analysis = (
                df.groupby("product_category")["sales_amount"]
                .agg(["sum", "mean", "count", "std"])
                .to_dict()
            )

            return category_analysis

        except Exception as e:
            self.logger.error(f"카테고리별 분석 실패: {e}")
            return {"error": str(e)}

    def _create_sales_visualizations(
        self, df: pd.DataFrame, trend_results: Dict[str, Any], seasonality_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """매출 시각화 생성"""
        try:
            visualizations = []

            # 월별 매출 트렌드 차트
            if "monthly_sales" in trend_results:
                visualizations.append(
                    {
                        "type": "line",
                        "title": "월별 매출 트렌드",
                        "data": trend_results["monthly_sales"],
                        "x_label": "월",
                        "y_label": "매출액",
                    }
                )

            # 계절성 차트
            if "monthly_seasonality" in seasonality_results:
                visualizations.append(
                    {
                        "type": "bar",
                        "title": "월별 계절성",
                        "data": seasonality_results["monthly_seasonality"],
                        "x_label": "월",
                        "y_label": "평균 매출액",
                    }
                )

            return visualizations

        except Exception as e:
            self.logger.error(f"매출 시각화 생성 실패: {e}")
            return []

    def _generate_sales_recommendations(
        self, trend_results: Dict[str, Any], seasonality_results: Dict[str, Any]
    ) -> List[str]:
        """매출 권장사항 생성"""
        recommendations = []

        try:
            # 트렌드 기반 권장사항
            if "trend_direction" in trend_results:
                if trend_results["trend_direction"] == "하락":
                    recommendations.append("매출이 하락하고 있습니다. 마케팅 전략을 점검하세요.")
                elif trend_results["trend_direction"] == "상승":
                    recommendations.append("매출이 상승하고 있습니다. 현재 전략을 유지하세요.")

            # 계절성 기반 권장사항
            if "seasonal_indices" in seasonality_results:
                seasonal_indices = seasonality_results["seasonal_indices"]
                peak_month = max(seasonal_indices, key=seasonal_indices.get)
                low_month = min(seasonal_indices, key=seasonal_indices.get)

                recommendations.append(
                    f"{peak_month}월에 매출이 최고조에 달합니다. 재고 준비를 강화하세요."
                )
                recommendations.append(f"{low_month}월에 매출이 저조합니다. 프로모션을 고려하세요.")

            return recommendations

        except Exception as e:
            self.logger.error(f"매출 권장사항 생성 실패: {e}")
            return ["권장사항 생성 중 오류가 발생했습니다."]


class ABTestAnalysisTemplate(BaseAnalysisTemplate):
    """A/B 테스트 분석 템플릿"""

    def get_template_info(self) -> AnalysisTemplate:
        """템플릿 정보 반환"""
        return AnalysisTemplate(
            name="ab_test_analysis",
            description="A/B 테스트 통계적 유의성 분석",
            required_columns=["group", "metric"],
            optional_columns=["user_id", "conversion", "revenue"],
            parameters={
                "test_type": "conversion",
                "confidence_level": 0.95,
                "minimum_effect_size": 0.05,
            },
            output_format="json",
        )

    def validate_data(self, data: UnifiedDataFrame) -> bool:
        """데이터 검증"""
        try:
            df = data.to_pandas()
            required_cols = ["group", "metric"]

            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"필수 컬럼 {col}이 없습니다.")
                    return False

            # 그룹이 2개인지 확인
            if len(df["group"].unique()) != 2:
                self.logger.error("그룹이 정확히 2개여야 합니다.")
                return False

            return True
        except Exception as e:
            self.logger.error(f"데이터 검증 실패: {e}")
            return False

    def execute_analysis(
        self, data: UnifiedDataFrame, parameters: Dict[str, Any]
    ) -> AnalysisResult:
        """A/B 테스트 분석 실행"""
        try:
            if not self.validate_data(data):
                return AnalysisResult(
                    template_name="ab_test_analysis",
                    success=False,
                    results={},
                    visualizations=[],
                    recommendations=["데이터 검증에 실패했습니다."],
                    metadata={"error": "데이터 검증 실패"},
                )

            df = data.to_pandas()

            # 기본 통계 분석
            basic_stats = self._calculate_basic_statistics(df)

            # 통계적 유의성 검정
            significance_test = self._perform_significance_test(df, parameters)

            # 효과 크기 계산
            effect_size = self._calculate_effect_size(df)

            # 시각화 생성
            visualizations = self._create_ab_test_visualizations(df, basic_stats)

            # 권장사항 생성
            recommendations = self._generate_ab_test_recommendations(significance_test, effect_size)

            return AnalysisResult(
                template_name="ab_test_analysis",
                success=True,
                results={
                    "basic_statistics": basic_stats,
                    "significance_test": significance_test,
                    "effect_size": effect_size,
                },
                visualizations=visualizations,
                recommendations=recommendations,
                metadata={
                    "test_type": parameters.get("test_type", "conversion"),
                    "confidence_level": parameters.get("confidence_level", 0.95),
                    "total_samples": len(df),
                },
            )

        except Exception as e:
            self.logger.error(f"A/B 테스트 분석 실행 실패: {e}")
            return AnalysisResult(
                template_name="ab_test_analysis",
                success=False,
                results={},
                visualizations=[],
                recommendations=[f"분석 실행 실패: {str(e)}"],
                metadata={"error": str(e)},
            )

    def _calculate_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """기본 통계 계산"""
        try:
            groups = df["group"].unique()
            stats = {}

            for group in groups:
                group_data = df[df["group"] == group]["metric"]
                stats[group] = {
                    "count": len(group_data),
                    "mean": group_data.mean(),
                    "std": group_data.std(),
                    "median": group_data.median(),
                    "min": group_data.min(),
                    "max": group_data.max(),
                }

            return stats

        except Exception as e:
            self.logger.error(f"기본 통계 계산 실패: {e}")
            return {"error": str(e)}

    def _perform_significance_test(
        self, df: pd.DataFrame, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """통계적 유의성 검정"""
        try:
            from scipy import stats

            groups = df["group"].unique()
            group_a = df[df["group"] == groups[0]]["metric"]
            group_b = df[df["group"] == groups[1]]["metric"]

            # t-검정 수행
            t_stat, p_value = stats.ttest_ind(group_a, group_b)

            # 신뢰도 수준
            confidence_level = parameters.get("confidence_level", 0.95)
            alpha = 1 - confidence_level

            # 유의성 판정
            is_significant = p_value < alpha

            return {
                "t_statistic": t_stat,
                "p_value": p_value,
                "is_significant": is_significant,
                "confidence_level": confidence_level,
                "alpha": alpha,
            }

        except Exception as e:
            self.logger.error(f"통계적 유의성 검정 실패: {e}")
            return {"error": str(e)}

    def _calculate_effect_size(self, df: pd.DataFrame) -> Dict[str, Any]:
        """효과 크기 계산"""
        try:
            groups = df["group"].unique()
            group_a = df[df["group"] == groups[0]]["metric"]
            group_b = df[df["group"] == groups[1]]["metric"]

            # Cohen's d 계산
            pooled_std = np.sqrt(
                ((len(group_a) - 1) * group_a.var() + (len(group_b) - 1) * group_b.var())
                / (len(group_a) + len(group_b) - 2)
            )
            cohens_d = (group_a.mean() - group_b.mean()) / pooled_std

            # 효과 크기 해석
            if abs(cohens_d) < 0.2:
                effect_size_interpretation = "작은 효과"
            elif abs(cohens_d) < 0.5:
                effect_size_interpretation = "중간 효과"
            else:
                effect_size_interpretation = "큰 효과"

            return {
                "cohens_d": cohens_d,
                "interpretation": effect_size_interpretation,
                "group_a_mean": group_a.mean(),
                "group_b_mean": group_b.mean(),
                "difference": group_a.mean() - group_b.mean(),
            }

        except Exception as e:
            self.logger.error(f"효과 크기 계산 실패: {e}")
            return {"error": str(e)}

    def _create_ab_test_visualizations(
        self, df: pd.DataFrame, basic_stats: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """A/B 테스트 시각화 생성"""
        try:
            visualizations = []

            # 그룹별 분포 히스토그램
            groups = df["group"].unique()
            for group in groups:
                group_data = df[df["group"] == group]["metric"]
                visualizations.append(
                    {
                        "type": "histogram",
                        "title": f"그룹 {group} 분포",
                        "data": group_data.tolist(),
                        "x_label": "메트릭 값",
                        "y_label": "빈도",
                    }
                )

            # 그룹별 박스플롯
            visualizations.append(
                {
                    "type": "boxplot",
                    "title": "그룹별 메트릭 비교",
                    "data": {
                        group: df[df["group"] == group]["metric"].tolist() for group in groups
                    },
                    "x_label": "그룹",
                    "y_label": "메트릭 값",
                }
            )

            return visualizations

        except Exception as e:
            self.logger.error(f"A/B 테스트 시각화 생성 실패: {e}")
            return []

    def _generate_ab_test_recommendations(
        self, significance_test: Dict[str, Any], effect_size: Dict[str, Any]
    ) -> List[str]:
        """A/B 테스트 권장사항 생성"""
        recommendations = []

        try:
            # 유의성 기반 권장사항
            if "is_significant" in significance_test:
                if significance_test["is_significant"]:
                    recommendations.append(
                        "통계적으로 유의한 차이가 있습니다. 승리 그룹을 선택하세요."
                    )
                else:
                    recommendations.append(
                        "통계적으로 유의한 차이가 없습니다. 더 많은 데이터를 수집하거나 테스트를 연장하세요."
                    )

            # 효과 크기 기반 권장사항
            if "interpretation" in effect_size:
                effect_interpretation = effect_size["interpretation"]
                if effect_interpretation == "큰 효과":
                    recommendations.append("효과 크기가 큽니다. 즉시 적용을 고려하세요.")
                elif effect_interpretation == "중간 효과":
                    recommendations.append("중간 정도의 효과입니다. 신중하게 검토 후 적용하세요.")
                else:
                    recommendations.append(
                        "효과 크기가 작습니다. 비즈니스 임팩트를 고려하여 결정하세요."
                    )

            return recommendations

        except Exception as e:
            self.logger.error(f"A/B 테스트 권장사항 생성 실패: {e}")
            return ["권장사항 생성 중 오류가 발생했습니다."]


class AnalysisTemplateManager:
    """분석 템플릿 관리자"""

    def __init__(self):
        """템플릿 관리자 초기화"""
        self.logger = logging.getLogger(__name__)
        self.templates: Dict[str, BaseAnalysisTemplate] = {}
        self._register_default_templates()

    def _register_default_templates(self):
        """기본 템플릿 등록"""
        self.templates["customer_analysis"] = CustomerAnalysisTemplate()
        self.templates["sales_analysis"] = SalesAnalysisTemplate()
        self.templates["ab_test_analysis"] = ABTestAnalysisTemplate()

    def register_template(self, name: str, template: BaseAnalysisTemplate):
        """템플릿 등록

        Args:
            name: 템플릿 이름
            template: 템플릿 인스턴스
        """
        self.templates[name] = template
        self.logger.info(f"템플릿 {name}이 등록되었습니다.")

    def get_template(self, name: str) -> Optional[BaseAnalysisTemplate]:
        """템플릿 조회

        Args:
            name: 템플릿 이름

        Returns:
            템플릿 인스턴스 또는 None
        """
        return self.templates.get(name)

    def list_templates(self) -> List[Dict[str, Any]]:
        """템플릿 목록 반환

        Returns:
            템플릿 목록
        """
        from dataclasses import asdict
        
        return [
            {"name": name, "info": asdict(template.get_template_info())}
            for name, template in self.templates.items()
        ]

    def execute_template(
        self, name: str, data: UnifiedDataFrame, parameters: Dict[str, Any] = None
    ) -> AnalysisResult:
        """템플릿 실행

        Args:
            name: 템플릿 이름
            data: 분석할 데이터
            parameters: 분석 매개변수

        Returns:
            분석 결과
        """
        try:
            template = self.get_template(name)
            if not template:
                return AnalysisResult(
                    template_name=name,
                    success=False,
                    results={},
                    visualizations=[],
                    recommendations=[f"템플릿 {name}을 찾을 수 없습니다."],
                    metadata={"error": "템플릿을 찾을 수 없음"},
                )

            if parameters is None:
                parameters = {}

            return template.execute_analysis(data, parameters)

        except Exception as e:
            self.logger.error(f"템플릿 {name} 실행 실패: {e}")
            return AnalysisResult(
                template_name=name,
                success=False,
                results={},
                visualizations=[],
                recommendations=[f"템플릿 실행 실패: {str(e)}"],
                metadata={"error": str(e)},
            )
