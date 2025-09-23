# Bridge 통합 데이터 분석 레이어 가이드

## 🚀 개요

Bridge 통합 데이터 분석 레이어는 **CA 마일스톤 3.1**로 구현된 핵심 기능으로, 다중 소스 데이터를 통합하고 표준화된 분석을 제공합니다.

## 📊 주요 구성 요소

### 1. DataUnifier - 데이터 통합
다중 소스 데이터를 표준 테이블 형태로 통합하는 핵심 모듈입니다.

```python
from bridge.analytics.core import DataUnifier

# 데이터 통합기 초기화
unifier = DataUnifier()

# 다중 소스 데이터 통합
data_sources = {
    "postgres": postgres_data,
    "mongodb": mongo_data,
    "elasticsearch": es_data
}

unified_data = unifier.unify_data_sources(
    data_sources=data_sources,
    schema_mapping=schema_mapping,
    merge_strategy="union"
)
```

### 2. SchemaMapper - 스키마 매핑
다양한 데이터 소스의 스키마를 표준화하고 매핑하는 시스템입니다.

```python
from bridge.analytics.core import SchemaMapper, SchemaMapping, ColumnMapping

# 스키마 매퍼 초기화
mapper = SchemaMapper()

# 스키마 매핑 정의
schema_mapping = SchemaMapping(
    source_name="postgres",
    target_schema={
        "user_id": ColumnMapping(
            source_column="id",
            target_column="user_id",
            data_type="integer",
            transformation="identity"
        ),
        "email": ColumnMapping(
            source_column="email_address",
            target_column="email",
            data_type="string",
            transformation="lowercase"
        )
    }
)

# 스키마 매핑 적용
mapped_data = mapper.apply_schema_mapping(data, schema_mapping)
```

### 3. TypeConverter - 타입 변환
고급 데이터 타입 변환을 수행하는 도구입니다.

```python
from bridge.analytics.core import TypeConverter, ConversionRule

# 타입 변환기 초기화
converter = TypeConverter()

# 변환 규칙 정의
conversion_rules = [
    ConversionRule(
        source_type="string",
        target_type="datetime",
        format_pattern="%Y-%m-%d",
        error_handling="coerce"
    ),
    ConversionRule(
        source_type="integer",
        target_type="float",
        transformation="divide_by_100"
    )
]

# 데이터 타입 변환
converted_data = converter.convert_types(data, conversion_rules)
```

### 4. StreamingProcessor - 스트리밍 처리
대용량 데이터를 메모리 효율적으로 처리하는 스트리밍 프로세서입니다.

```python
from bridge.analytics.core import StreamingProcessor

# 스트리밍 프로세서 초기화
processor = StreamingProcessor(
    chunk_size=10000,
    memory_limit_mb=1000
)

# 대용량 데이터 스트리밍 처리
def process_chunk(chunk):
    # 청크별 처리 로직
    return chunk.apply_transformation()

processed_data = processor.process_streaming(
    data_stream=data_stream,
    processor_func=process_chunk
)
```

### 5. IntegratedDataLayer - 통합 레이어
모든 구성 요소를 통합하여 사용하는 메인 클래스입니다.

```python
from bridge.analytics.core import IntegratedDataLayer

# 통합 데이터 레이어 초기화
layer = IntegratedDataLayer(
    chunk_size=10000,
    memory_limit_mb=1000,
    auto_schema_mapping=True
)

# 다중 소스 데이터 통합
unified_data = layer.integrate_data_sources(
    data_sources=data_sources,
    schema_mapping=schema_mapping,
    merge_strategy="union",
    enable_streaming=True
)
```

## 🔧 MCP 도구 사용법

### data_unifier 도구
```json
{
  "tool": "data_unifier",
  "arguments": {
    "action": "unify",
    "data_sources": {
      "source1": {"data": [...]},
      "source2": {"data": [...]}
    },
    "merge_strategy": "union"
  }
}
```

### schema_mapper 도구
```json
{
  "tool": "schema_mapper",
  "arguments": {
    "action": "map_schema",
    "data": {...},
    "mapping_rules": {
      "source_column": "target_column"
    }
  }
}
```

### type_converter 도구
```json
{
  "tool": "type_converter",
  "arguments": {
    "action": "convert",
    "data": {...},
    "conversion_rules": [
      {
        "source_type": "string",
        "target_type": "datetime",
        "format": "%Y-%m-%d"
      }
    ]
  }
}
```

### streaming_processor 도구
```json
{
  "tool": "streaming_processor",
  "arguments": {
    "action": "process",
    "data": {...},
    "processor_func": "transform_data",
    "chunk_size": 10000,
    "memory_limit_mb": 1000
  }
}
```

### integrated_data_layer 도구
```json
{
  "tool": "integrated_data_layer",
  "arguments": {
    "action": "integrate",
    "data_sources": {...},
    "transformations": [...],
    "export_format": "arrow"
  }
}
```

## 📋 데이터 통합 전략

### Union 전략
모든 소스의 컬럼을 포함하며, 누락된 값은 NULL로 처리합니다.

```python
unified_data = layer.integrate_data_sources(
    data_sources=data_sources,
    merge_strategy="union"
)
```

### Intersection 전략
공통 컬럼만 포함하여 데이터를 통합합니다.

```python
unified_data = layer.integrate_data_sources(
    data_sources=data_sources,
    merge_strategy="intersection"
)
```

### Custom 전략
사용자 정의 통합 로직을 사용합니다.

```python
def custom_merge_logic(source1, source2):
    # 커스텀 통합 로직
    return merged_data

unified_data = layer.integrate_data_sources(
    data_sources=data_sources,
    merge_strategy="custom",
    custom_merge_func=custom_merge_logic
)
```

## ⚡ 성능 최적화

### 메모리 효율적인 처리
```python
# 스트리밍 처리 활성화
layer = IntegratedDataLayer(
    chunk_size=5000,  # 작은 청크 크기
    memory_limit_mb=500,  # 낮은 메모리 제한
    auto_schema_mapping=True
)

# 대용량 데이터 처리
unified_data = layer.integrate_data_sources(
    data_sources=large_data_sources,
    enable_streaming=True
)
```

### 병렬 처리
```python
# 멀티프로세싱을 통한 병렬 처리
from multiprocessing import Pool

def process_source(source_data):
    return layer.process_single_source(source_data)

with Pool(processes=4) as pool:
    results = pool.map(process_source, data_sources.values())
```

## 🚨 에러 처리

### 데이터 통합 에러 처리
```python
try:
    unified_data = layer.integrate_data_sources(data_sources)
except DataIntegrationError as e:
    logger.error(f"데이터 통합 실패: {e}")
    # 부분 통합 시도
    unified_data = layer.integrate_data_sources_partial(data_sources)
```

### 스키마 매핑 에러 처리
```python
try:
    mapped_data = mapper.apply_schema_mapping(data, schema_mapping)
except SchemaMappingError as e:
    logger.error(f"스키마 매핑 실패: {e}")
    # 자동 매핑으로 대체
    auto_mapping = mapper.create_auto_mapping(data)
    mapped_data = mapper.apply_schema_mapping(data, auto_mapping)
```

## 📊 실제 사용 예시

### 1. 다중 데이터베이스 통합
```python
# PostgreSQL, MongoDB, Elasticsearch 데이터 통합
data_sources = {
    "users": postgres_users_data,
    "orders": postgres_orders_data,
    "products": mongo_products_data,
    "logs": es_logs_data
}

# 스키마 매핑 정의
schema_mapping = {
    "users": {
        "id": "user_id",
        "email": "email_address",
        "created_at": "registration_date"
    },
    "orders": {
        "user_id": "customer_id",
        "order_date": "purchase_date",
        "total": "order_amount"
    }
}

# 데이터 통합
unified_data = layer.integrate_data_sources(
    data_sources=data_sources,
    schema_mapping=schema_mapping,
    merge_strategy="union"
)

# 통합된 데이터 분석
summary = layer.get_data_summary()
print(f"통합된 데이터: {summary['total_rows']}행, {summary['total_columns']}열")
```

### 2. 실시간 데이터 스트리밍
```python
# 실시간 데이터 스트리밍 처리
def process_realtime_data(data_chunk):
    # 실시간 데이터 처리 로직
    processed_chunk = data_chunk.filter(lambda x: x['status'] == 'active')
    return processed_chunk

# 스트리밍 처리
streaming_processor = StreamingProcessor(chunk_size=1000)
processed_stream = streaming_processor.process_streaming(
    data_stream=realtime_data_stream,
    processor_func=process_realtime_data
)
```

### 3. 데이터 품질 검사
```python
# 통합된 데이터 품질 검사
from bridge.analytics.core import QualityChecker

quality_checker = QualityChecker()
quality_report = quality_checker.check_data_quality(unified_data)

# 품질 리포트 출력
print(f"데이터 품질 점수: {quality_report.overall_score}")
print(f"결측값 비율: {quality_report.missing_value_ratio}")
print(f"중복값 비율: {quality_report.duplicate_ratio}")
```

## 🔍 주의사항

1. **메모리 관리**: 대용량 데이터는 스트리밍 처리 사용
2. **스키마 일관성**: 통합 전에 스키마 매핑 규칙 정의
3. **데이터 품질**: 통합 후 데이터 품질 검사 수행
4. **성능 모니터링**: 처리 시간과 메모리 사용량 모니터링
5. **에러 복구**: 부분 실패 시 대체 전략 준비
6. **데이터 검증**: 통합된 데이터의 정확성 검증

## 📚 추가 리소스

- [ML 사용 가이드](ml-user-guide.md) - 머신러닝 기능 사용법
- [API 참조 문서](api-reference.md) - REST API 완전 참조
- [개발자 가이드](developer-guide.md) - 개발 환경 설정 및 기여 방법
