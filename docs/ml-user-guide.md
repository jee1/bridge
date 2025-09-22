# Bridge ML 사용 가이드

## 🚀 개요

Bridge ML은 시계열 분석, 클러스터링, 이상치 탐지, 모델 관리 등 다양한 머신러닝 기능을 제공합니다. 이 가이드는 각 기능의 사용법과 예시를 설명합니다.

## 📊 시계열 분석

### TimeSeriesAnalyzer 사용법

```python
from bridge.ml.algorithms.time_series import TimeSeriesAnalyzer, TimeSeriesResult, ForecastResult
import pandas as pd

# 시계열 분석기 초기화
analyzer = TimeSeriesAnalyzer()

# 데이터 준비 (pandas DataFrame)
data = pd.read_csv('sales_data.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')

# 정상성 검사
is_stationary, adf_stat, adf_pvalue = analyzer.analyze_stationarity(data['sales'])
print(f"정상성: {is_stationary}, ADF 통계: {adf_stat:.4f}, p-value: {adf_pvalue:.4f}")

# ARIMA 모델 훈련
arima_result = analyzer.fit_arima(data['sales'], order=(1, 1, 1))
print(f"ARIMA 모델 훈련 완료: {arima_result.model_type}")

# Prophet 모델 훈련
prophet_result = analyzer.fit_prophet(data['sales'], seasonality_mode='multiplicative')
print(f"Prophet 모델 훈련 완료: {prophet_result.model_type}")

# LSTM 모델 훈련
lstm_result = analyzer.fit_lstm(data['sales'], sequence_length=30, epochs=100)
print(f"LSTM 모델 훈련 완료: {lstm_result.model_type}")

# 예측 수행
forecast = analyzer.forecast(arima_result, periods=30)
print(f"예측 완료: {len(forecast.forecast_values)}개 기간")
```

### 시계열 분석 결과 활용

```python
# 결과 검증
if arima_result.is_stationary:
    print("데이터가 정상성을 만족합니다")

# 신뢰구간 확인
if arima_result.confidence_intervals:
    lower, upper = arima_result.confidence_intervals
    print(f"신뢰구간: {lower} ~ {upper}")

# 계절성 성분 확인
if prophet_result.seasonal_components:
    for component, values in prophet_result.seasonal_components.items():
        print(f"{component}: {values}")

# 모델 메트릭 확인
if arima_result.model_metrics:
    for metric, value in arima_result.model_metrics.items():
        print(f"{metric}: {value:.4f}")
```

## 🔍 이상치 탐지

### AnomalyDetector 사용법

```python
from bridge.ml.algorithms.anomaly_detection import AnomalyDetector
import numpy as np

# 이상치 탐지기 초기화
detector = AnomalyDetector()

# 데이터 준비
data = np.random.randn(1000)
data[100:110] += 5  # 이상치 추가

# IQR 방법으로 이상치 탐지
iqr_anomalies = detector.detect_iqr(data, threshold=1.5)
print(f"IQR 이상치 수: {len(iqr_anomalies)}")

# Z-score 방법으로 이상치 탐지
zscore_anomalies = detector.detect_zscore(data, threshold=3.0)
print(f"Z-score 이상치 수: {len(zscore_anomalies)}")

# Isolation Forest로 이상치 탐지
isolation_anomalies = detector.detect_isolation_forest(data, contamination=0.1)
print(f"Isolation Forest 이상치 수: {len(isolation_anomalies)}")

# One-Class SVM으로 이상치 탐지
svm_anomalies = detector.detect_one_class_svm(data, nu=0.1)
print(f"One-Class SVM 이상치 수: {len(svm_anomalies)}")
```

## 🎯 클러스터링

### ClusteringAnalyzer 사용법

```python
from bridge.ml.algorithms.clustering import ClusteringAnalyzer
import numpy as np

# 클러스터링 분석기 초기화
clustering = ClusteringAnalyzer()

# 데이터 준비
data = np.random.randn(100, 2)

# K-means 클러스터링
kmeans_clusters = clustering.kmeans(data, n_clusters=3)
print(f"K-means 클러스터 수: {len(np.unique(kmeans_clusters))}")

# DBSCAN 클러스터링
dbscan_clusters = clustering.dbscan(data, eps=0.5, min_samples=5)
print(f"DBSCAN 클러스터 수: {len(np.unique(dbscan_clusters))}")

# 계층적 클러스터링
hierarchical_clusters = clustering.hierarchical(data, n_clusters=3)
print(f"계층적 클러스터 수: {len(np.unique(hierarchical_clusters))}")

# 클러스터 평가
silhouette_score = clustering.evaluate_silhouette(data, kmeans_clusters)
print(f"실루엣 점수: {silhouette_score:.4f}")
```

## 📉 차원 축소

### DimensionalityReducer 사용법

```python
from bridge.ml.algorithms.dimensionality_reduction import DimensionalityReducer
import numpy as np

# 차원 축소기 초기화
reducer = DimensionalityReducer()

# 데이터 준비
data = np.random.randn(100, 10)  # 10차원 데이터

# PCA 차원 축소
pca_data = reducer.pca(data, n_components=2)
print(f"PCA 결과: {pca_data.shape}")

# t-SNE 차원 축소
tsne_data = reducer.tsne(data, n_components=2)
print(f"t-SNE 결과: {tsne_data.shape}")

# UMAP 차원 축소
umap_data = reducer.umap(data, n_components=2)
print(f"UMAP 결과: {umap_data.shape}")

# 설명 분산 비율 확인
explained_variance = reducer.get_explained_variance_ratio(data, n_components=2)
print(f"설명 분산 비율: {explained_variance}")
```

## 🤖 모델 관리

### ModelRegistry 사용법

```python
from bridge.ml.models.registry import ModelRegistry
from bridge.governance.contracts import ModelContract, ModelType, ModelStatus

# 모델 레지스트리 초기화
registry = ModelRegistry(storage_path="models")

# 모델 계약 생성
model_contract = ModelContract(
    id="churn_model_001",
    name="고객 이탈 예측 모델",
    model_type=ModelType.CLASSIFICATION,
    version="1.0.0",
    status=ModelStatus.READY,
    metadata={
        "algorithm": "RandomForest",
        "accuracy": 0.85,
        "features": ["age", "income", "usage_days"]
    }
)

# 모델 등록
registry.register_model(model_contract)
print("모델 등록 완료")

# 모델 조회
model = registry.get_model("churn_model_001")
print(f"모델 조회: {model.name}")

# 모든 모델 목록 조회
models = registry.list_models()
print(f"등록된 모델 수: {len(models)}")

# 모델 검색
search_results = registry.search_models(
    model_type=ModelType.CLASSIFICATION,
    status=ModelStatus.READY
)
print(f"검색 결과: {len(search_results)}개")
```

### 모델 추론

```python
from bridge.ml.models.inference import ModelInference

# 모델 추론기 초기화
inference = ModelInference(registry)

# 모델 로드
model = inference.load_model("churn_model_001", version="1.0.0")

# 예측 수행
test_data = [[25, 50000, 30], [35, 75000, 60]]
predictions = inference.predict(
    model_id="churn_model_001",
    data=test_data,
    version="1.0.0"
)
print(f"예측 결과: {predictions}")

# 배치 예측
batch_predictions = inference.batch_predict(
    model_id="churn_model_001",
    data_list=[test_data, test_data]
)
print(f"배치 예측 결과: {len(batch_predictions)}개")
```

## 📋 거버넌스 계약

### DataContract 사용법

```python
from bridge.governance.contracts import DataContract, DataType, QualityRule

# 데이터 계약 생성
contract = DataContract(
    id="customer_data_contract",
    name="고객 데이터 계약",
    version="1.0.0",
    schema={
        "customer_id": DataType.INTEGER,
        "name": DataType.STRING,
        "email": DataType.STRING,
        "age": DataType.INTEGER,
        "created_at": DataType.DATETIME
    },
    quality_rules=[
        QualityRule(
            field="customer_id",
            rule_type="not_null",
            description="고객 ID는 필수입니다"
        ),
        QualityRule(
            field="email",
            rule_type="email_format",
            description="이메일 형식이 올바르야 합니다"
        ),
        QualityRule(
            field="age",
            rule_type="range",
            parameters={"min": 0, "max": 120},
            description="나이는 0-120 사이여야 합니다"
        )
    ]
)

# 데이터 검증
test_data = {
    "customer_id": 1,
    "name": "홍길동",
    "email": "hong@example.com",
    "age": 30,
    "created_at": "2024-01-01T00:00:00Z"
}

validation_result = contract.validate_data(test_data)
if validation_result.is_valid:
    print("데이터가 계약을 만족합니다")
else:
    print(f"검증 실패: {validation_result.errors}")
```

## 🔧 고급 사용법

### 데이터 전처리

```python
# 시계열 데이터 전처리
cleaned_data = analyzer.handle_missing_values(data, method='interpolate')
cleaned_data = analyzer.remove_outliers(cleaned_data, method='iqr')
normalized_data = analyzer.normalize(cleaned_data, method='minmax')
differenced_data = analyzer.difference(normalized_data, periods=1)
```

### 모델 평가

```python
# 교차 검증
cv_scores = analyzer.cross_validate(data, model_type='arima', cv_folds=5)
print(f"교차 검증 점수: {cv_scores}")

# 예측 정확도 평가
accuracy = analyzer.evaluate_forecast(actual, predicted)
print(f"예측 정확도: {accuracy:.4f}")
```

### 성능 최적화

```python
# 대용량 데이터 처리
for chunk in analyzer.process_in_chunks(data, chunk_size=1000):
    result = analyzer.fit_arima(chunk)
    # 결과 저장

# 병렬 처리
results = analyzer.fit_multiple_models(data, models=['arima', 'prophet', 'lstm'])
```

## 🚨 주의사항

1. **데이터 품질**: ML 분석 전에 데이터 품질을 반드시 확인
2. **정상성 검사**: ARIMA 모델 사용 전에 정상성 검사 필수
3. **계절성 고려**: 계절성이 있는 데이터는 Prophet 모델 권장
4. **메모리 관리**: 대용량 데이터는 청크 단위로 처리
5. **모델 검증**: 교차 검증을 통한 모델 성능 검증 필수
6. **계약 준수**: 데이터 계약을 통한 품질 보장

## 📚 추가 리소스

- [시계열 분석 가이드](docs/time-series-analysis.md)
- [모델 관리 가이드](docs/model-management.md)
- [거버넌스 계약 가이드](docs/governance-contracts.md)
- [API 참조 문서](docs/api-reference.md)
