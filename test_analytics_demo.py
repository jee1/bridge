#!/usr/bin/env python3
"""Bridge Analytics 기능 데모 테스트 스크립트."""

import asyncio
import time
import random
from bridge.analytics.core import (
    UnifiedDataFrame, 
    TypeNormalizer, 
    ConnectorAdapter, 
    CrossSourceJoiner
)
from bridge.connectors.mock import MockConnector


def test_unified_dataframe():
    """UnifiedDataFrame 기능 테스트."""
    print("=== UnifiedDataFrame 테스트 ===")
    
    # 딕셔너리 리스트로 테스트
    data = [
        {"id": 1, "name": "Alice", "age": 30, "salary": 50000.5},
        {"id": 2, "name": "Bob", "age": 25, "salary": 45000.0},
        {"id": 3, "name": "Charlie", "age": 35, "salary": 60000.0}
    ]
    
    df = UnifiedDataFrame(data)
    print(f"✅ 데이터 로드: {df.num_rows}행, {df.num_columns}열")
    print(f"✅ 컬럼: {df.column_names}")
    print(f"✅ 스키마 정보: {df.get_schema_info()}")
    
    # 컬럼 선택 테스트
    selected = df.select_columns(["name", "salary"])
    print(f"✅ 컬럼 선택: {selected.column_names}")
    
    # Pandas 변환 테스트
    pandas_df = df.to_pandas()
    print(f"✅ Pandas 변환: {pandas_df.shape}")
    
    return df


def test_type_normalizer():
    """TypeNormalizer 기능 테스트."""
    print("\n=== TypeNormalizer 테스트 ===")
    
    normalizer = TypeNormalizer()
    
    # 타입 감지 테스트
    data = [
        {"id": 1, "name": "test", "value": 1.5, "active": True, "score": 85},
        {"id": 2, "name": "demo", "value": 2.3, "active": False, "score": 92}
    ]
    
    types = normalizer.detect_types(data)
    print(f"✅ 감지된 타입: {types}")
    
    # 데이터 정규화 테스트
    df = UnifiedDataFrame(data)
    normalized_table = normalizer.normalize_data(df.table)
    print(f"✅ 정규화된 스키마: {normalized_table.schema}")
    
    return normalizer


def test_cross_source_joiner():
    """CrossSourceJoiner 기능 테스트."""
    print("\n=== CrossSourceJoiner 테스트 ===")
    
    joiner = CrossSourceJoiner()
    
    # 테이블 데이터 준비
    users_data = [
        {"user_id": 1, "name": "Alice", "department": "Engineering"},
        {"user_id": 2, "name": "Bob", "department": "Marketing"},
        {"user_id": 3, "name": "Charlie", "department": "Engineering"}
    ]
    
    orders_data = [
        {"order_id": 101, "user_id": 1, "amount": 150.0, "status": "completed"},
        {"order_id": 102, "user_id": 2, "amount": 75.5, "status": "pending"},
        {"order_id": 103, "user_id": 1, "amount": 200.0, "status": "completed"}
    ]
    
    # 테이블 등록
    users_df = UnifiedDataFrame(users_data)
    orders_df = UnifiedDataFrame(orders_data)
    
    joiner.register_table("users", users_df)
    joiner.register_table("orders", orders_df)
    
    print(f"✅ 등록된 테이블: {joiner.get_registered_tables()}")
    
    # 조인 실행
    result = joiner.join_tables("users", "orders", "users.user_id = orders.user_id")
    print(f"✅ 조인 결과: {result.num_rows}행, {result.num_columns}열")
    print(f"✅ 조인 컬럼: {result.column_names}")
    
    # 결과 출력
    result_df = result.to_pandas()
    print("✅ 조인 결과 데이터:")
    print(result_df)
    
    joiner.close()
    return result


async def test_connector_adapter():
    """ConnectorAdapter 기능 테스트."""
    print("\n=== ConnectorAdapter 테스트 ===")
    
    # Mock 커넥터 생성
    mock_connector = MockConnector("test_db", sample_rows=[
        {"id": 1, "product": "Laptop", "price": 1200.0, "category": "Electronics"},
        {"id": 2, "product": "Mouse", "price": 25.0, "category": "Accessories"},
        {"id": 3, "product": "Keyboard", "price": 75.0, "category": "Accessories"}
    ])
    
    # 어댑터로 래핑
    adapter = ConnectorAdapter(mock_connector, normalize_types=True)
    
    # 연결 테스트
    connection_ok = await adapter.test_connection()
    print(f"✅ 연결 테스트: {'성공' if connection_ok else '실패'}")
    
    # 메타데이터 조회
    metadata = await adapter.get_metadata()
    print(f"✅ 메타데이터: {metadata['connector_name']}")
    
    # 쿼리 실행
    result = await adapter.execute_query("SELECT * FROM products")
    print(f"✅ 쿼리 결과: {result.num_rows}행, {result.num_columns}열")
    print(f"✅ 결과 컬럼: {result.column_names}")
    
    # 결과 출력
    result_df = result.to_pandas()
    print("✅ 쿼리 결과 데이터:")
    print(result_df)
    
    return result


def test_performance():
    """성능 테스트."""
    print("\n=== 성능 테스트 ===")
    
    # 대용량 데이터 생성
    print("대용량 데이터 생성 중...")
    large_data = [
        {
            "id": i,
            "value": random.randint(1, 1000),
            "category": random.choice(["A", "B", "C", "D"]),
            "score": random.uniform(0, 100),
            "active": random.choice([True, False])
        }
        for i in range(10000)
    ]
    
    # 성능 측정
    start_time = time.time()
    df = UnifiedDataFrame(large_data)
    end_time = time.time()
    
    print(f"✅ 10,000행 데이터 처리 시간: {end_time - start_time:.3f}초")
    print(f"✅ 메모리 사용량: {df.table.nbytes / 1024 / 1024:.2f} MB")
    print(f"✅ 데이터 타입: {df.get_schema_info()['column_types']}")
    
    return df


async def main():
    """메인 테스트 함수."""
    print("🚀 Bridge Analytics 기능 테스트 시작\n")
    
    try:
        # 1. UnifiedDataFrame 테스트
        df = test_unified_dataframe()
        
        # 2. TypeNormalizer 테스트
        normalizer = test_type_normalizer()
        
        # 3. CrossSourceJoiner 테스트
        join_result = test_cross_source_joiner()
        
        # 4. ConnectorAdapter 테스트
        adapter_result = await test_connector_adapter()
        
        # 5. 성능 테스트
        perf_df = test_performance()
        
        print("\n🎉 모든 테스트가 성공적으로 완료되었습니다!")
        print("\n📊 테스트 요약:")
        print(f"  - UnifiedDataFrame: ✅ {df.num_rows}행 처리")
        print(f"  - TypeNormalizer: ✅ 타입 정규화 완료")
        print(f"  - CrossSourceJoiner: ✅ {join_result.num_rows}행 조인")
        print(f"  - ConnectorAdapter: ✅ {adapter_result.num_rows}행 쿼리")
        print(f"  - 성능 테스트: ✅ {perf_df.num_rows}행 처리")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
