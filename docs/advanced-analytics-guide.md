# Bridge 고급 통계 분석 및 시각화 가이드

## 🚀 개요

Bridge 고급 통계 분석 및 시각화는 **CA 마일스톤 3.2**로 구현된 핵심 기능으로, 고급 통계 분석, 인터랙티브 시각화, 통계적 검정, 시계열 분석을 제공합니다.

## 📊 주요 구성 요소

### 1. AdvancedStatistics - 고급 통계 분석
고급 통계 분석 기능을 제공하는 모듈입니다.

```python
from bridge.analytics.core import AdvancedStatistics, UnifiedDataFrame

# 고급 통계 분석기 초기화
analyzer = AdvancedStatistics()

# 기술 통계 분석
descriptive_result = analyzer.descriptive_statistics(
    data=df,
    columns=["value1", "value2"],
    include_percentiles=True,
    include_skewness=True,
    include_kurtosis=True
)

# 상관관계 분석
correlation_result = analyzer.correlation_analysis(
    data=df,
    columns=["value1", "value2", "value3"],
    method="pearson"
)

# 분포 분석
distribution_result = analyzer.distribution_analysis(
    data=df,
    columns=["value1"],
    test_normality=True
)
```

### 2. AdvancedVisualization - 고급 시각화
인터랙티브 차트 및 시각화를 제공하는 모듈입니다.

```python
from bridge.analytics.core import AdvancedVisualization

# 고급 시각화 생성기 초기화
viz = AdvancedVisualization()

# 고급 차트 생성
chart_result = viz.create_advanced_chart(
    data=df,
    chart_type="bar",
    x_column="category",
    y_column="value",
    hue_column="group",
    title="Category vs Value by Group"
)

# 통계적 플롯 생성
statistical_plots = viz.create_statistical_plots(
    data=df,
    column="value",
    plot_types=["histogram", "qq", "box", "violin", "density", "ecdf"]
)

# 시계열 플롯 생성
time_series_plot = viz.create_time_series_plot(
    data=df,
    time_column="timestamp",
    value_columns=["value1", "value2"],
    title="Time Series Analysis"
)
```

### 3. StatisticalTests - 통계적 검정
통계적 검정 기능을 제공하는 모듈입니다.

```python
from bridge.analytics.core import StatisticalTests

# 통계적 검정 수행기 초기화
tester = StatisticalTests()

# t-검정 수행
t_test_result = tester.t_test(
    data=df,
    test_type="two_sample",
    column="value",
    group_column="group",
    alternative="two-sided"
)

# ANOVA 검정 수행
anova_result = tester.anova_test(
    data=df,
    column="value",
    group_column="category"
)

# 회귀분석 수행
regression_result = tester.regression_analysis(
    data=df,
    dependent_var="value",
    independent_vars=["value1", "value2"],
    method="linear"
)

# A/B 테스트 수행
ab_test_result = tester.ab_test(
    data=df,
    metric_column="conversion_rate",
    group_column="variant",
    confidence_level=0.95
)
```

### 4. TimeSeriesAnalysis - 시계열 분석
시계열 분석 및 예측을 제공하는 모듈입니다.

```python
from bridge.analytics.core import TimeSeriesAnalysis

# 시계열 분석기 초기화
ts_analyzer = TimeSeriesAnalysis()

# 트렌드 분석
trend_result = ts_analyzer.detect_trend(
    data=df,
    time_column="timestamp",
    value_column="value",
    trend_type="linear"
)

# 계절성 분석
seasonality_result = ts_analyzer.analyze_seasonality(
    data=df,
    time_column="timestamp",
    value_column="value",
    period="monthly"
)

# 이상치 탐지
anomaly_result = ts_analyzer.detect_anomalies(
    data=df,
    time_column="timestamp",
    value_column="value",
    method="isolation_forest"
)

# 예측 수행
forecast_result = ts_analyzer.forecast(
    data=df,
    time_column="timestamp",
    value_column="value",
    periods=30,
    model="arima"
)
```

## 🔧 MCP 도구 사용법

### advanced_statistics 도구
```json
{
  "tool": "advanced_statistics",
  "arguments": {
    "data": {...},
    "analysis_type": "descriptive",
    "columns": ["value1", "value2"],
    "include_percentiles": true,
    "include_skewness": true,
    "include_kurtosis": true
  }
}
```

### interactive_charts 도구
```json
{
  "tool": "interactive_charts",
  "arguments": {
    "data": {...},
    "chart_type": "bar",
    "x_column": "category",
    "y_column": "value",
    "hue_column": "group",
    "title": "Category vs Value",
    "config": {
      "figsize": [12, 8],
      "style": "whitegrid"
    }
  }
}
```

### statistical_tests 도구
```json
{
  "tool": "statistical_tests",
  "arguments": {
    "data": {...},
    "test_type": "t_test",
    "column": "value",
    "group_column": "group",
    "alternative": "two_sided",
    "confidence_level": 0.95
  }
}
```

### time_series_analysis 도구
```json
{
  "tool": "time_series_analysis",
  "arguments": {
    "data": {...},
    "analysis_type": "trend",
    "time_column": "timestamp",
    "value_column": "value",
    "trend_type": "linear",
    "periods": 30
  }
}
```

## 📋 고급 분석 패턴

### 1. 종합적 데이터 분석 워크플로우
```python
# 1단계: 기술 통계 분석
descriptive = analyzer.descriptive_statistics(df, ["value1", "value2"])

# 2단계: 상관관계 분석
correlation = analyzer.correlation_analysis(df, ["value1", "value2"])

# 3단계: 분포 분석
distribution = analyzer.distribution_analysis(df, ["value1"])

# 4단계: 통계적 검정
t_test = tester.t_test(df, "two_sample", "value1", "group")

# 5단계: 시각화
chart = viz.create_advanced_chart(df, "scatter", "value1", "value2")
```

### 2. 시계열 분석 워크플로우
```python
# 1단계: 트렌드 분석
trend = ts_analyzer.detect_trend(df, "timestamp", "value", "linear")

# 2단계: 계절성 분석
seasonality = ts_analyzer.analyze_seasonality(df, "timestamp", "value")

# 3단계: 이상치 탐지
anomalies = ts_analyzer.detect_anomalies(df, "timestamp", "value")

# 4단계: 예측
forecast = ts_analyzer.forecast(df, "timestamp", "value", periods=30)

# 5단계: 시각화
ts_plot = viz.create_time_series_plot(df, "timestamp", ["value"])
```

### 3. A/B 테스트 분석 워크플로우
```python
# 1단계: A/B 테스트 수행
ab_result = tester.ab_test(df, "conversion_rate", "variant")

# 2단계: 통계적 유의성 확인
if ab_result["p_value"] < 0.05:
    print("통계적으로 유의한 차이가 있습니다")
    print(f"효과 크기: {ab_result['effect_size']}")

# 3단계: 시각화
ab_chart = viz.create_advanced_chart(
    df, "bar", "variant", "conversion_rate",
    title="A/B Test Results"
)
```

## 📊 실제 사용 예시

### 1. 고객 행동 분석
```python
# 고객 데이터 분석
customer_data = UnifiedDataFrame(customer_df)

# 기술 통계 분석
stats = analyzer.descriptive_statistics(
    customer_data, 
    ["age", "income", "spending"]
)

# 상관관계 분석
correlation = analyzer.correlation_analysis(
    customer_data,
    ["age", "income", "spending"]
)

# 시각화
age_income_chart = viz.create_advanced_chart(
    customer_data, "scatter", "age", "income",
    hue_column="segment",
    title="Age vs Income by Segment"
)
```

### 2. 매출 트렌드 분석
```python
# 매출 데이터 시계열 분석
sales_data = UnifiedDataFrame(sales_df)

# 트렌드 분석
trend = ts_analyzer.detect_trend(
    sales_data, "date", "revenue", "linear"
)

# 계절성 분석
seasonality = ts_analyzer.analyze_seasonality(
    sales_data, "date", "revenue", "monthly"
)

# 예측
forecast = ts_analyzer.forecast(
    sales_data, "date", "revenue", periods=12
)

# 시계열 시각화
ts_plot = viz.create_time_series_plot(
    sales_data, "date", ["revenue"],
    title="Sales Revenue Trend"
)
```

### 3. 제품 성능 A/B 테스트
```python
# A/B 테스트 데이터 분석
ab_data = UnifiedDataFrame(ab_test_df)

# A/B 테스트 수행
ab_result = tester.ab_test(
    ab_data, "conversion_rate", "variant"
)

# 통계적 검정
t_test = tester.t_test(
    ab_data, "two_sample", "conversion_rate", "variant"
)

# 결과 시각화
ab_chart = viz.create_advanced_chart(
    ab_data, "bar", "variant", "conversion_rate",
    title="A/B Test: Conversion Rate by Variant"
)
```

## ⚡ 성능 최적화

### 메모리 효율적인 처리
```python
# 대용량 데이터 처리
def process_large_dataset(data_chunks):
    results = []
    for chunk in data_chunks:
        # 청크별 분석
        chunk_result = analyzer.descriptive_statistics(chunk)
        results.append(chunk_result)
    
    # 결과 통합
    return combine_results(results)
```

### 병렬 처리
```python
from multiprocessing import Pool

def analyze_parallel(data_chunks):
    with Pool(processes=4) as pool:
        results = pool.map(analyze_chunk, data_chunks)
    return results
```

## 🚨 에러 처리

### 통계 분석 에러 처리
```python
try:
    result = analyzer.descriptive_statistics(df, columns)
except StatisticalError as e:
    logger.error(f"통계 분석 실패: {e}")
    # 대체 분석 방법
    result = analyzer.basic_statistics(df, columns)
```

### 시각화 에러 처리
```python
try:
    chart = viz.create_advanced_chart(df, chart_type, x_col, y_col)
except VisualizationError as e:
    logger.error(f"시각화 생성 실패: {e}")
    # 기본 차트로 대체
    chart = viz.create_basic_chart(df, x_col, y_col)
```

## 🔍 주의사항

1. **데이터 품질**: 분석 전에 데이터 품질 확인 필수
2. **통계적 가정**: 검정 전에 가정 조건 확인
3. **샘플 크기**: 통계적 검정의 신뢰성을 위한 충분한 샘플 크기
4. **다중 검정**: 여러 검정 시 보정 방법 적용
5. **해석 주의**: 통계적 유의성과 실용적 유의성 구분
6. **시각화 명확성**: 차트의 명확성과 해석 가능성 고려

## 📚 추가 리소스

- [ML 사용 가이드](ml-user-guide.md) - 머신러닝 기능 사용법
- [통합 데이터 분석 가이드](integrated-data-layer-guide.md) - 통합 데이터 분석 레이어 사용법
- [API 참조 문서](api-reference.md) - REST API 완전 참조
- [개발자 가이드](developer-guide.md) - 개발 환경 설정 및 기여 방법
