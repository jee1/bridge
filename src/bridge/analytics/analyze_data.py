"""통합 데이터 분석 함수.

Bridge Analytics의 모든 기능을 통합하여 사용하기 쉬운 단일 인터페이스를 제공합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa

from bridge.analytics.core.data_integration import UnifiedDataFrame
from bridge.analytics.core.integrated_data_layer import IntegratedDataLayer
from bridge.analytics.core.statistics import StatisticsAnalyzer
from bridge.analytics.core.visualization import ChartGenerator, DashboardGenerator, ReportGenerator
from bridge.analytics.quality.comprehensive_metrics import ComprehensiveQualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """분석 설정을 담는 데이터 클래스"""

    # 데이터 통합 설정
    enable_data_integration: bool = True
    auto_schema_mapping: bool = True
    merge_strategy: str = "union"
    enable_streaming: bool = True

    # 분석 설정
    include_descriptive_stats: bool = True
    include_correlation_analysis: bool = True
    include_distribution_analysis: bool = True
    include_quality_metrics: bool = True

    # 시각화 설정
    generate_charts: bool = True
    generate_dashboard: bool = True
    generate_report: bool = True
    chart_columns: Optional[List[str]] = None

    # 품질 검증 설정
    quality_threshold: float = 0.8
    include_outlier_detection: bool = True

    # 출력 설정
    verbose: bool = True
    save_charts: bool = False
    output_dir: Optional[str] = None


@dataclass
class AnalysisResult:
    """분석 결과를 담는 데이터 클래스"""

    # 기본 정보
    success: bool
    data_summary: Dict[str, Any]
    analysis_time: float

    # 통계 분석 결과
    descriptive_stats: Optional[Dict[str, Any]] = None
    correlation_analysis: Optional[Dict[str, Any]] = None
    distribution_analysis: Optional[Dict[str, Any]] = None

    # 품질 메트릭
    quality_metrics: Optional[Dict[str, Any]] = None

    # 시각화
    charts: Optional[Dict[str, Any]] = None
    dashboard: Optional[Any] = None
    report: Optional[Dict[str, Any]] = None

    # 오류 정보
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None


def analyze_data(
    data: Union[
        pd.DataFrame,
        pa.Table,
        List[Dict[str, Any]],
        Dict[str, Any],
        UnifiedDataFrame,
        Dict[str, Union[pd.DataFrame, pa.Table, List[Dict[str, Any]]]],
    ],
    config: Optional[AnalysisConfig] = None,
    **kwargs
) -> AnalysisResult:
    """통합 데이터 분석을 수행합니다.

    이 함수는 Bridge Analytics의 모든 기능을 통합하여
    다양한 데이터 소스에 대한 종합적인 분석을 제공합니다.

    Args:
        data: 분석할 데이터
            - pandas DataFrame
            - Arrow Table
            - 딕셔너리 리스트
            - 단일 딕셔너리
            - UnifiedDataFrame
            - 다중 소스 딕셔너리 {소스명: 데이터}
        config: 분석 설정 (선택사항)
        **kwargs: 추가 설정 (config를 덮어씀)

    Returns:
        AnalysisResult: 종합 분석 결과

    Examples:
        >>> import pandas as pd
        >>> from bridge.analytics.analyze_data import analyze_data
        >>>
        >>> # 단일 데이터프레임 분석
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> result = analyze_data(df)
        >>>
        >>> # 다중 소스 데이터 분석
        >>> data_sources = {
        ...     'sales': sales_df,
        ...     'customers': customers_df
        ... }
        >>> result = analyze_data(data_sources)
        >>>
        >>> # 커스텀 설정으로 분석
        >>> config = AnalysisConfig(
        ...     include_quality_metrics=True,
        ...     generate_charts=True,
        ...     quality_threshold=0.9
        ... )
        >>> result = analyze_data(df, config)
    """
    import time

    start_time = time.time()
    errors = []
    warnings = []

    # 설정 초기화
    if config is None:
        config = AnalysisConfig()

    # kwargs로 설정 덮어쓰기
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    try:
        logger.info("데이터 분석 시작")

        # 1. 데이터 통합
        unified_data = _integrate_data(data, config, errors, warnings)

        # 2. 데이터 요약
        data_summary = _get_data_summary(unified_data)

        # 3. 통계 분석
        descriptive_stats = None
        correlation_analysis = None
        distribution_analysis = None

        if config.include_descriptive_stats or config.include_correlation_analysis or config.include_distribution_analysis:
            stats_analyzer = StatisticsAnalyzer()
            
            if config.include_descriptive_stats:
                try:
                    descriptive_stats = stats_analyzer.generate_summary_report(
                        unified_data, config.chart_columns
                    )
                except Exception as e:
                    error_msg = f"기술 통계 분석 실패: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            if config.include_correlation_analysis:
                try:
                    corr_result = stats_analyzer.calculate_correlation(
                        unified_data, config.chart_columns
                    )
                    correlation_analysis = {
                        "correlation_matrix": corr_result.correlation_matrix.to_dict(),
                        "strong_correlations": corr_result.strong_correlations,
                        "moderate_correlations": corr_result.moderate_correlations,
                    }
                except Exception as e:
                    error_msg = f"상관관계 분석 실패: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

            if config.include_distribution_analysis:
                try:
                    distribution_analysis = stats_analyzer.calculate_distribution_stats(
                        unified_data, config.chart_columns
                    )
                except Exception as e:
                    error_msg = f"분포 분석 실패: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)

        # 4. 품질 메트릭
        quality_metrics = None
        if config.include_quality_metrics:
            try:
                quality_analyzer = ComprehensiveQualityMetrics()
                quality_result = quality_analyzer.calculate_overall_quality(unified_data)
                quality_metrics = {
                    "overall_score": quality_result.overall_score,
                    "completeness": quality_result.completeness,
                    "accuracy": quality_result.accuracy,
                    "consistency": quality_result.consistency,
                    "validity": quality_result.validity,
                    "missing_ratio": quality_result.missing_ratio,
                    "duplicate_ratio": quality_result.duplicate_ratio,
                    "constraint_violations": quality_result.constraint_violations,
                    "data_type_consistency": quality_result.data_type_consistency,
                }
                
                # 품질 경고
                if quality_result.overall_score < config.quality_threshold:
                    warnings.append(f"데이터 품질이 임계값({config.quality_threshold}) 미만입니다: {quality_result.overall_score:.3f}")
                    
            except Exception as e:
                error_msg = f"품질 메트릭 계산 실패: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        # 5. 시각화
        charts = None
        dashboard = None
        report = None

        if config.generate_charts or config.generate_dashboard or config.generate_report:
            try:
                chart_generator = ChartGenerator()
                dashboard_generator = DashboardGenerator()
                report_generator = ReportGenerator()

                if config.generate_charts:
                    charts = _generate_charts(unified_data, chart_generator, config)

                if config.generate_dashboard:
                    dashboard = dashboard_generator.create_analytics_dashboard(unified_data)

                if config.generate_report:
                    report = report_generator.generate_analytics_report(unified_data)

            except Exception as e:
                error_msg = f"시각화 생성 실패: {e}"
                errors.append(error_msg)
                logger.error(error_msg)

        # 6. 결과 생성
        analysis_time = time.time() - start_time

        result = AnalysisResult(
            success=len(errors) == 0,
            data_summary=data_summary,
            analysis_time=analysis_time,
            descriptive_stats=descriptive_stats,
            correlation_analysis=correlation_analysis,
            distribution_analysis=distribution_analysis,
            quality_metrics=quality_metrics,
            charts=charts,
            dashboard=dashboard,
            report=report,
            errors=errors if errors else None,
            warnings=warnings if warnings else None,
        )

        if config.verbose:
            _print_analysis_summary(result)

        logger.info(f"데이터 분석 완료 (소요시간: {analysis_time:.2f}초)")
        return result

    except Exception as e:
        error_msg = f"데이터 분석 중 오류 발생: {e}"
        errors.append(error_msg)
        logger.error(error_msg)

        return AnalysisResult(
            success=False,
            data_summary={},
            analysis_time=time.time() - start_time,
            errors=errors,
        )


def _integrate_data(
    data: Union[
        pd.DataFrame,
        pa.Table,
        List[Dict[str, Any]],
        Dict[str, Any],
        UnifiedDataFrame,
        Dict[str, Union[pd.DataFrame, pa.Table, List[Dict[str, Any]]]],
    ],
    config: AnalysisConfig,
    errors: List[str],
    warnings: List[str],
) -> UnifiedDataFrame:
    """데이터를 통합합니다."""
    try:
        # 이미 UnifiedDataFrame인 경우
        if isinstance(data, UnifiedDataFrame):
            return data

        # 다중 소스 데이터인 경우
        if isinstance(data, dict) and not _is_single_data_dict(data):
            if config.enable_data_integration:
                integrated_layer = IntegratedDataLayer(
                    auto_schema_mapping=config.auto_schema_mapping
                )
                return integrated_layer.integrate_data_sources(
                    data,
                    merge_strategy=config.merge_strategy,
                    enable_streaming=config.enable_streaming,
                )
            else:
                warnings.append("다중 소스 데이터가 감지되었지만 데이터 통합이 비활성화되어 있습니다.")

        # 단일 데이터 소스를 UnifiedDataFrame으로 변환
        return UnifiedDataFrame(data)

    except Exception as e:
        error_msg = f"데이터 통합 실패: {e}"
        errors.append(error_msg)
        raise


def _is_single_data_dict(data: Dict[str, Any]) -> bool:
    """딕셔너리가 단일 데이터인지 다중 소스인지 판단합니다."""
    if not data:
        return True
    
    # 첫 번째 값이 데이터 구조인지 확인
    first_value = next(iter(data.values()))
    # DataFrame이나 Table이면 다중 소스, 그 외에는 단일 데이터
    return not isinstance(first_value, (pd.DataFrame, pa.Table))


def _get_data_summary(unified_data: UnifiedDataFrame) -> Dict[str, Any]:
    """데이터 요약 정보를 생성합니다."""
    try:
        pandas_df = unified_data.to_pandas()
        
        return {
            "total_rows": int(len(pandas_df)),
            "total_columns": int(len(pandas_df.columns)),
            "column_names": list(pandas_df.columns),
            "column_types": {
                col: str(dtype) for col, dtype in pandas_df.dtypes.items()
            },
            "numeric_columns": int(len(pandas_df.select_dtypes(include=["number"]).columns)),
            "categorical_columns": int(len(pandas_df.select_dtypes(include=["object"]).columns)),
            "missing_values": int(pandas_df.isnull().sum().sum()),
            "duplicate_rows": int(pandas_df.duplicated().sum()),
            "memory_usage_mb": round(float(pandas_df.memory_usage(deep=True).sum()) / 1024 / 1024, 2),
        }
    except Exception as e:
        logger.error(f"데이터 요약 생성 실패: {e}")
        return {}


def _generate_charts(
    unified_data: UnifiedDataFrame, 
    chart_generator: ChartGenerator, 
    config: AnalysisConfig
) -> Dict[str, Any]:
    """차트를 생성합니다."""
    try:
        pandas_df = unified_data.to_pandas()
        charts = {}

        # 숫자형 컬럼 선택
        numeric_columns = pandas_df.select_dtypes(include=["number"]).columns.tolist()
        categorical_columns = pandas_df.select_dtypes(include=["object"]).columns.tolist()

        # 분석할 컬럼 결정
        target_columns = config.chart_columns or numeric_columns

        # 히스토그램 (첫 번째 숫자형 컬럼)
        if numeric_columns:
            charts["histogram"] = chart_generator.create_histogram(
                unified_data, numeric_columns[0]
            )

        # 막대 차트 (첫 번째 범주형 컬럼)
        if categorical_columns:
            charts["bar_chart"] = chart_generator.create_bar_chart(
                unified_data, categorical_columns[0]
            )

        # 산점도 (두 개의 숫자형 컬럼)
        if len(numeric_columns) >= 2:
            charts["scatter_plot"] = chart_generator.create_scatter_plot(
                unified_data, numeric_columns[0], numeric_columns[1]
            )

        # 박스 플롯
        if numeric_columns:
            charts["box_plot"] = chart_generator.create_box_plot(
                unified_data, None, numeric_columns[0]
            )

        # 상관관계 히트맵
        if len(numeric_columns) >= 2:
            charts["heatmap"] = chart_generator.create_heatmap(
                unified_data, columns=numeric_columns
            )

        return charts

    except Exception as e:
        logger.error(f"차트 생성 실패: {e}")
        return {}


def _print_analysis_summary(result: AnalysisResult) -> None:
    """분석 결과 요약을 출력합니다."""
    print("\n" + "="*60)
    print("📊 Bridge Analytics - 데이터 분석 결과")
    print("="*60)
    
    # 기본 정보
    print(f"✅ 분석 성공: {'예' if result.success else '아니오'}")
    print(f"⏱️  분석 시간: {result.analysis_time:.2f}초")
    
    # 데이터 요약
    if result.data_summary:
        print(f"\n📋 데이터 요약:")
        print(f"   • 총 행 수: {result.data_summary.get('total_rows', 0):,}")
        print(f"   • 총 열 수: {result.data_summary.get('total_columns', 0)}")
        print(f"   • 숫자형 컬럼: {result.data_summary.get('numeric_columns', 0)}")
        print(f"   • 범주형 컬럼: {result.data_summary.get('categorical_columns', 0)}")
        print(f"   • 결측값: {result.data_summary.get('missing_values', 0):,}")
        print(f"   • 중복 행: {result.data_summary.get('duplicate_rows', 0):,}")
        print(f"   • 메모리 사용량: {result.data_summary.get('memory_usage_mb', 0)} MB")
    
    # 품질 메트릭
    if result.quality_metrics:
        print(f"\n🔍 데이터 품질:")
        print(f"   • 전체 점수: {result.quality_metrics.get('overall_score', 0):.3f}")
        print(f"   • 완전성: {result.quality_metrics.get('completeness', 0):.3f}")
        print(f"   • 정확성: {result.quality_metrics.get('accuracy', 0):.3f}")
        print(f"   • 일관성: {result.quality_metrics.get('consistency', 0):.3f}")
        print(f"   • 유효성: {result.quality_metrics.get('validity', 0):.3f}")
    
    # 경고 및 오류
    if result.warnings:
        print(f"\n⚠️  경고 ({len(result.warnings)}개):")
        for warning in result.warnings:
            print(f"   • {warning}")
    
    if result.errors:
        print(f"\n❌ 오류 ({len(result.errors)}개):")
        for error in result.errors:
            print(f"   • {error}")
    
    print("="*60)


# 편의 함수들
def quick_analysis(data: Union[pd.DataFrame, pa.Table, List[Dict[str, Any]], Dict[str, Any]]) -> AnalysisResult:
    """빠른 분석을 수행합니다 (기본 설정 사용)."""
    return analyze_data(data, AnalysisConfig(verbose=True))


def comprehensive_analysis(
    data: Union[pd.DataFrame, pa.Table, List[Dict[str, Any]], Dict[str, Any]]
) -> AnalysisResult:
    """종합 분석을 수행합니다 (모든 기능 활성화)."""
    config = AnalysisConfig(
        enable_data_integration=True,
        include_descriptive_stats=True,
        include_correlation_analysis=True,
        include_distribution_analysis=True,
        include_quality_metrics=True,
        generate_charts=True,
        generate_dashboard=True,
        generate_report=True,
        quality_threshold=0.8,
        verbose=True,
    )
    return analyze_data(data, config)


def quality_focused_analysis(
    data: Union[pd.DataFrame, pa.Table, List[Dict[str, Any]], Dict[str, Any]]
) -> AnalysisResult:
    """품질 중심 분석을 수행합니다."""
    config = AnalysisConfig(
        include_descriptive_stats=True,
        include_quality_metrics=True,
        quality_threshold=0.9,
        generate_charts=False,
        generate_dashboard=False,
        generate_report=False,
        verbose=True,
    )
    return analyze_data(data, config)


def visualization_focused_analysis(
    data: Union[pd.DataFrame, pa.Table, List[Dict[str, Any]], Dict[str, Any]]
) -> AnalysisResult:
    """시각화 중심 분석을 수행합니다."""
    config = AnalysisConfig(
        include_descriptive_stats=True,
        include_correlation_analysis=True,
        include_quality_metrics=False,
        generate_charts=True,
        generate_dashboard=True,
        generate_report=True,
        verbose=True,
    )
    return analyze_data(data, config)
