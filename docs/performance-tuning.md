# Bridge 성능 튜닝 가이드

## 📖 개요

이 문서는 Bridge 시스템의 성능을 최적화하기 위한 종합적인 가이드입니다. 데이터베이스 쿼리 최적화, 캐싱 전략, 비동기 처리, 메모리 관리, 그리고 모니터링을 통한 성능 개선 방법을 다룹니다.

## 🚀 성능 최적화 전략

### 1. 성능 측정 및 모니터링

#### 성능 메트릭 정의

```python
from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps

# 메트릭 정의
REQUEST_COUNT = Counter('bridge_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('bridge_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('bridge_active_connections', 'Active connections')
DATABASE_QUERY_DURATION = Histogram('bridge_db_query_duration_seconds', 'Database query duration', ['query_type'])
CACHE_HIT_RATIO = Gauge('bridge_cache_hit_ratio', 'Cache hit ratio')
MEMORY_USAGE = Gauge('bridge_memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('bridge_cpu_usage_percent', 'CPU usage percentage')

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def track_request(self, method: str, endpoint: str, duration: float, status: str):
        """요청 추적"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def track_database_query(self, query_type: str, duration: float):
        """데이터베이스 쿼리 추적"""
        DATABASE_QUERY_DURATION.labels(query_type=query_type).observe(duration)
    
    def update_system_metrics(self):
        """시스템 메트릭 업데이트"""
        import psutil
        process = psutil.Process()
        
        MEMORY_USAGE.set(process.memory_info().rss)
        CPU_USAGE.set(process.cpu_percent())
        ACTIVE_CONNECTIONS.set(len(process.connections()))
    
    def get_performance_summary(self) -> dict:
        """성능 요약 조회"""
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": uptime,
            "memory_usage_mb": MEMORY_USAGE._value.get() / 1024 / 1024,
            "cpu_usage_percent": CPU_USAGE._value.get(),
            "active_connections": ACTIVE_CONNECTIONS._value.get()
        }

# 성능 측정 데코레이터
def measure_performance(metric_name: str):
    """성능 측정 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                # 메트릭 기록
                return result
            except Exception as e:
                duration = time.time() - start_time
                # 에러 메트릭 기록
                raise
        return wrapper
    return decorator
```

#### 실시간 성능 대시보드

```python
from fastapi import FastAPI, WebSocket
from typing import Dict, Any
import asyncio
import json

class PerformanceDashboard:
    """성능 대시보드"""
    
    def __init__(self):
        self.connections = set()
        self.monitor = PerformanceMonitor()
    
    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket 엔드포인트"""
        await websocket.accept()
        self.connections.add(websocket)
        
        try:
            while True:
                # 성능 데이터 수집
                performance_data = await self._collect_performance_data()
                
                # 클라이언트에게 전송
                await websocket.send_text(json.dumps(performance_data))
                await asyncio.sleep(1)  # 1초마다 업데이트
                
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.connections.discard(websocket)
    
    async def _collect_performance_data(self) -> Dict[str, Any]:
        """성능 데이터 수집"""
        self.monitor.update_system_metrics()
        
        return {
            "timestamp": time.time(),
            "system": {
                "memory_usage_mb": MEMORY_USAGE._value.get() / 1024 / 1024,
                "cpu_usage_percent": CPU_USAGE._value.get(),
                "active_connections": ACTIVE_CONNECTIONS._value.get()
            },
            "requests": {
                "total_requests": REQUEST_COUNT._value.sum(),
                "avg_duration": REQUEST_DURATION._value.sum() / max(REQUEST_COUNT._value.sum(), 1)
            },
            "database": {
                "avg_query_duration": DATABASE_QUERY_DURATION._value.sum() / max(DATABASE_QUERY_DURATION._value.count(), 1)
            }
        }
```

### 2. 데이터베이스 성능 최적화

#### 쿼리 최적화

```python
from sqlalchemy import text, select, func
from sqlalchemy.orm import sessionmaker
from typing import List, Dict, Any
import time

class QueryOptimizer:
    """쿼리 최적화기"""
    
    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.query_cache = {}
        self.query_stats = {}
    
    async def execute_optimized_query(self, query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """최적화된 쿼리 실행"""
        start_time = time.time()
        
        try:
            # 쿼리 분석
            query_plan = await self._analyze_query(query)
            
            # 인덱스 힌트 추가
            optimized_query = await self._add_index_hints(query, query_plan)
            
            # 실행 계획 확인
            execution_plan = await self._get_execution_plan(optimized_query)
            
            # 쿼리 실행
            with self.session_factory() as session:
                result = session.execute(text(optimized_query), params or {})
                rows = [dict(row) for row in result]
            
            duration = time.time() - start_time
            
            # 성능 통계 기록
            self._record_query_stats(query, duration, len(rows))
            
            return rows
            
        except Exception as e:
            duration = time.time() - start_time
            self._record_query_stats(query, duration, 0, error=str(e))
            raise
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """쿼리 분석"""
        # 쿼리 복잡도 분석
        complexity_score = self._calculate_complexity(query)
        
        # 테이블 및 컬럼 분석
        tables = self._extract_tables(query)
        columns = self._extract_columns(query)
        
        # 조인 분석
        joins = self._analyze_joins(query)
        
        return {
            "complexity_score": complexity_score,
            "tables": tables,
            "columns": columns,
            "joins": joins
        }
    
    def _calculate_complexity(self, query: str) -> int:
        """쿼리 복잡도 계산"""
        complexity = 0
        
        # SELECT 절 복잡도
        complexity += query.upper().count('SELECT') * 1
        
        # JOIN 복잡도
        complexity += query.upper().count('JOIN') * 2
        
        # 서브쿼리 복잡도
        complexity += query.upper().count('(SELECT') * 3
        
        # 집계 함수 복잡도
        aggregate_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'GROUP_CONCAT']
        for func in aggregate_functions:
            complexity += query.upper().count(func) * 1
        
        return complexity
    
    async def _add_index_hints(self, query: str, query_plan: Dict[str, Any]) -> str:
        """인덱스 힌트 추가"""
        # 테이블별 인덱스 힌트 추가
        for table in query_plan.get("tables", []):
            # 가장 적합한 인덱스 선택
            best_index = await self._select_best_index(table, query_plan)
            if best_index:
                query = query.replace(f"FROM {table}", f"FROM {table} USE INDEX ({best_index})")
        
        return query
    
    async def _select_best_index(self, table: str, query_plan: Dict[str, Any]) -> str:
        """최적 인덱스 선택"""
        # 테이블의 인덱스 정보 조회
        indexes = await self._get_table_indexes(table)
        
        # 쿼리 조건과 매칭되는 인덱스 선택
        for index in indexes:
            if self._is_index_suitable(index, query_plan):
                return index["name"]
        
        return None
    
    async def _get_table_indexes(self, table: str) -> List[Dict[str, Any]]:
        """테이블 인덱스 정보 조회"""
        # 실제 구현에서는 데이터베이스에서 인덱스 정보 조회
        return []
    
    def _is_index_suitable(self, index: Dict[str, Any], query_plan: Dict[str, Any]) -> bool:
        """인덱스 적합성 검사"""
        # 인덱스 컬럼과 쿼리 조건 매칭
        return True
    
    async def _get_execution_plan(self, query: str) -> Dict[str, Any]:
        """실행 계획 조회"""
        # EXPLAIN 쿼리 실행
        with self.session_factory() as session:
            result = session.execute(text(f"EXPLAIN {query}"))
            return [dict(row) for row in result]
    
    def _record_query_stats(self, query: str, duration: float, row_count: int, error: str = None):
        """쿼리 통계 기록"""
        query_hash = hash(query)
        
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = {
                "query": query,
                "execution_count": 0,
                "total_duration": 0,
                "total_rows": 0,
                "error_count": 0
            }
        
        stats = self.query_stats[query_hash]
        stats["execution_count"] += 1
        stats["total_duration"] += duration
        stats["total_rows"] += row_count
        
        if error:
            stats["error_count"] += 1
```

#### 연결 풀 최적화

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.engine import Engine
import asyncio

class DatabaseConnectionManager:
    """데이터베이스 연결 관리자"""
    
    def __init__(self, database_url: str, max_connections: int = 20):
        self.database_url = database_url
        self.max_connections = max_connections
        self.engine = self._create_optimized_engine()
        self.session_factory = sessionmaker(bind=self.engine)
    
    def _create_optimized_engine(self) -> Engine:
        """최적화된 엔진 생성"""
        engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,  # 기본 연결 수
            max_overflow=20,  # 추가 연결 수
            pool_pre_ping=True,  # 연결 상태 확인
            pool_recycle=3600,  # 1시간마다 연결 재생성
            echo=False,  # SQL 로깅 비활성화
            connect_args={
                "connect_timeout": 10,
                "application_name": "bridge_analytics"
            }
        )
        return engine
    
    async def get_connection(self):
        """연결 획득"""
        return self.engine.connect()
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None):
        """쿼리 실행"""
        async with self.get_connection() as conn:
            result = await conn.execute(text(query), params or {})
            return [dict(row) for row in result]
    
    def get_pool_status(self) -> Dict[str, Any]:
        """연결 풀 상태 조회"""
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
```

### 3. 캐싱 전략

#### 다층 캐싱 시스템

```python
import redis
import json
import pickle
from typing import Any, Optional, Union
from functools import wraps
import hashlib

class MultiLayerCache:
    """다층 캐싱 시스템"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url)
        self.local_cache = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0
        }
    
    def cache_key(self, func_name: str, *args, **kwargs) -> str:
        """캐시 키 생성"""
        key_data = f"{func_name}:{args}:{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """캐시에서 데이터 조회"""
        # 1단계: 로컬 캐시 확인
        if key in self.local_cache:
            self.cache_stats["hits"] += 1
            self.cache_stats["local_hits"] += 1
            return self.local_cache[key]
        
        # 2단계: Redis 캐시 확인
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                data = pickle.loads(cached_data)
                self.local_cache[key] = data  # 로컬 캐시에 저장
                self.cache_stats["hits"] += 1
                self.cache_stats["redis_hits"] += 1
                return data
        except Exception as e:
            print(f"Redis cache error: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, key: str, data: Any, ttl: int = 3600):
        """캐시에 데이터 저장"""
        # 로컬 캐시 저장
        self.local_cache[key] = data
        
        # Redis 캐시 저장
        try:
            serialized_data = pickle.dumps(data)
            self.redis_client.setex(key, ttl, serialized_data)
        except Exception as e:
            print(f"Redis cache set error: {e}")
    
    def invalidate(self, pattern: str):
        """패턴에 맞는 캐시 무효화"""
        # 로컬 캐시 무효화
        keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
        for key in keys_to_remove:
            del self.local_cache[key]
        
        # Redis 캐시 무효화
        try:
            keys = self.redis_client.keys(f"*{pattern}*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception as e:
            print(f"Redis cache invalidation error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_ratio = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            "hit_ratio": hit_ratio,
            "local_cache_size": len(self.local_cache)
        }

# 캐싱 데코레이터
def cached(ttl: int = 3600, cache_key_func: callable = None):
    """캐싱 데코레이터"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = MultiLayerCache()
            
            # 캐시 키 생성
            if cache_key_func:
                key = cache_key_func(*args, **kwargs)
            else:
                key = cache.cache_key(func.__name__, *args, **kwargs)
            
            # 캐시에서 조회
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행
            result = await func(*args, **kwargs)
            
            # 결과 캐싱
            cache.set(key, result, ttl)
            
            return result
        return wrapper
    return decorator
```

#### 쿼리 결과 캐싱

```python
class QueryResultCache:
    """쿼리 결과 캐싱"""
    
    def __init__(self, cache_manager: MultiLayerCache):
        self.cache_manager = cache_manager
        self.query_cache_ttl = {
            "metadata": 3600,  # 1시간
            "statistics": 1800,  # 30분
            "quality_check": 900,  # 15분
            "chart_data": 600,  # 10분
        }
    
    def cache_query_result(self, query_type: str, query: str, params: Dict[str, Any], result: Any):
        """쿼리 결과 캐싱"""
        cache_key = self._generate_cache_key(query_type, query, params)
        ttl = self.query_cache_ttl.get(query_type, 3600)
        self.cache_manager.set(cache_key, result, ttl)
    
    def get_cached_result(self, query_type: str, query: str, params: Dict[str, Any]) -> Optional[Any]:
        """캐시된 결과 조회"""
        cache_key = self._generate_cache_key(query_type, query, params)
        return self.cache_manager.get(cache_key)
    
    def _generate_cache_key(self, query_type: str, query: str, params: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        key_data = f"{query_type}:{query}:{params}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def invalidate_query_cache(self, query_type: str = None):
        """쿼리 캐시 무효화"""
        if query_type:
            self.cache_manager.invalidate(f"{query_type}:")
        else:
            self.cache_manager.invalidate("")
```

### 4. 비동기 처리 최적화

#### 비동기 작업 큐

```python
from celery import Celery
from celery.result import AsyncResult
from typing import Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncTaskManager:
    """비동기 작업 관리자"""
    
    def __init__(self, celery_app: Celery, max_workers: int = 4):
        self.celery_app = celery_app
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.active_tasks = {}
    
    async def submit_task(self, task_name: str, args: tuple = (), kwargs: dict = None) -> str:
        """작업 제출"""
        task = self.celery_app.send_task(task_name, args=args, kwargs=kwargs or {})
        self.active_tasks[task.id] = {
            "task": task,
            "status": "PENDING",
            "created_at": time.time()
        }
        return task.id
    
    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """작업 결과 조회"""
        if task_id not in self.active_tasks:
            return {"error": "Task not found"}
        
        task_info = self.active_tasks[task_id]
        task = task_info["task"]
        
        if task.ready():
            if task.successful():
                result = task.result
                task_info["status"] = "SUCCESS"
                return {
                    "status": "SUCCESS",
                    "result": result
                }
            else:
                error = str(task.result)
                task_info["status"] = "FAILURE"
                return {
                    "status": "FAILURE",
                    "error": error
                }
        else:
            return {
                "status": "PENDING",
                "progress": task_info.get("progress", 0)
            }
    
    async def execute_parallel_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """병렬 작업 실행"""
        async def execute_single_task(task_config):
            task_name = task_config["name"]
            args = task_config.get("args", ())
            kwargs = task_config.get("kwargs", {})
            
            return await self.submit_task(task_name, args, kwargs)
        
        # 모든 작업을 병렬로 제출
        task_ids = await asyncio.gather(*[execute_single_task(task) for task in tasks])
        
        # 결과 수집
        results = []
        for task_id in task_ids:
            result = await self.get_task_result(task_id)
            results.append({"task_id": task_id, **result})
        
        return results
    
    async def batch_process(self, items: List[Any], process_func: callable, 
                          batch_size: int = 100) -> List[Any]:
        """배치 처리"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # 배치를 병렬로 처리
            batch_tasks = [
                asyncio.create_task(
                    asyncio.get_event_loop().run_in_executor(
                        self.executor, process_func, item
                    )
                ) for item in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
```

#### 메모리 효율적인 데이터 처리

```python
import pandas as pd
import pyarrow as pa
from typing import Iterator, Any
import gc

class MemoryEfficientProcessor:
    """메모리 효율적인 데이터 처리기"""
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
    
    def process_large_dataset(self, data_source: str, process_func: callable) -> Iterator[Any]:
        """대용량 데이터셋 처리"""
        # 청크 단위로 데이터 읽기
        for chunk in self._read_data_in_chunks(data_source):
            try:
                # 청크 처리
                processed_chunk = process_func(chunk)
                yield processed_chunk
                
                # 메모리 정리
                del chunk
                gc.collect()
                
            except Exception as e:
                print(f"Chunk processing error: {e}")
                continue
    
    def _read_data_in_chunks(self, data_source: str) -> Iterator[pd.DataFrame]:
        """청크 단위로 데이터 읽기"""
        # 실제 구현에서는 데이터베이스 커서나 파일 스트림 사용
        offset = 0
        
        while True:
            # 데이터베이스에서 청크 읽기
            chunk = self._read_chunk_from_db(data_source, offset, self.chunk_size)
            
            if chunk.empty:
                break
            
            yield chunk
            offset += self.chunk_size
    
    def _read_chunk_from_db(self, data_source: str, offset: int, limit: int) -> pd.DataFrame:
        """데이터베이스에서 청크 읽기"""
        # 실제 구현에서는 데이터베이스 쿼리 실행
        return pd.DataFrame()
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """메모리 사용량 최적화"""
        # 데이터 타입 최적화
        for col in df.columns:
            if df[col].dtype == 'object':
                # 문자열 컬럼 최적화
                df[col] = df[col].astype('category')
            elif df[col].dtype == 'int64':
                # 정수 컬럼 최적화
                if df[col].min() >= 0:
                    if df[col].max() < 255:
                        df[col] = df[col].astype('uint8')
                    elif df[col].max() < 65535:
                        df[col] = df[col].astype('uint16')
                    elif df[col].max() < 4294967295:
                        df[col] = df[col].astype('uint32')
                else:
                    if df[col].min() > -128 and df[col].max() < 127:
                        df[col] = df[col].astype('int8')
                    elif df[col].min() > -32768 and df[col].max() < 32767:
                        df[col] = df[col].astype('int16')
                    elif df[col].min() > -2147483648 and df[col].max() < 2147483647:
                        df[col] = df[col].astype('int32')
            elif df[col].dtype == 'float64':
                # 실수 컬럼 최적화
                df[col] = pd.to_numeric(df[col], downcast='float')
        
        return df
```

### 5. API 성능 최적화

#### 응답 압축

```python
from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
import gzip
import json

def setup_response_compression(app: FastAPI):
    """응답 압축 설정"""
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    @app.middleware("http")
    async def compress_response(request: Request, call_next):
        response = await call_next(request)
        
        # JSON 응답 압축
        if (response.headers.get("content-type", "").startswith("application/json") and
            len(response.body) > 1000):
            
            compressed_body = gzip.compress(response.body)
            response.body = compressed_body
            response.headers["content-encoding"] = "gzip"
            response.headers["content-length"] = str(len(compressed_body))
        
        return response
```

#### 페이지네이션 최적화

```python
from typing import List, Dict, Any, Optional
from sqlalchemy import select, func

class PaginatedQuery:
    """페이지네이션 쿼리"""
    
    def __init__(self, query, page: int = 1, page_size: int = 20):
        self.query = query
        self.page = max(1, page)
        self.page_size = min(100, max(1, page_size))  # 최대 100개로 제한
        self.offset = (self.page - 1) * self.page_size
    
    async def execute(self, session) -> Dict[str, Any]:
        """페이지네이션 쿼리 실행"""
        # 총 개수 조회
        count_query = select(func.count()).select_from(self.query.subquery())
        total_count = session.execute(count_query).scalar()
        
        # 페이지 데이터 조회
        paginated_query = self.query.offset(self.offset).limit(self.page_size)
        results = session.execute(paginated_query).fetchall()
        
        # 페이지 정보 계산
        total_pages = (total_count + self.page_size - 1) // self.page_size
        has_next = self.page < total_pages
        has_prev = self.page > 1
        
        return {
            "data": [dict(row) for row in results],
            "pagination": {
                "page": self.page,
                "page_size": self.page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev
            }
        }
```

#### API 응답 캐싱

```python
from fastapi import FastAPI, Request, Response
from typing import Callable
import hashlib
import json

class APICacheMiddleware:
    """API 캐싱 미들웨어"""
    
    def __init__(self, cache_manager: MultiLayerCache):
        self.cache_manager = cache_manager
        self.cacheable_endpoints = {
            "/api/v1/analytics/statistics": 1800,  # 30분
            "/api/v1/connectors/metadata": 3600,   # 1시간
            "/api/v1/dashboards": 600,            # 10분
        }
    
    def __call__(self, request: Request, call_next: Callable) -> Response:
        # 캐시 가능한 엔드포인트 확인
        if request.url.path not in self.cacheable_endpoints:
            return call_next(request)
        
        # 캐시 키 생성
        cache_key = self._generate_cache_key(request)
        
        # 캐시에서 응답 조회
        cached_response = self.cache_manager.get(cache_key)
        if cached_response:
            return Response(
                content=cached_response["content"],
                status_code=cached_response["status_code"],
                headers=cached_response["headers"]
            )
        
        # 원본 요청 처리
        response = call_next(request)
        
        # 응답 캐싱
        if response.status_code == 200:
            ttl = self.cacheable_endpoints[request.url.path]
            self.cache_manager.set(cache_key, {
                "content": response.body,
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }, ttl)
        
        return response
    
    def _generate_cache_key(self, request: Request) -> str:
        """캐시 키 생성"""
        key_data = f"{request.url.path}:{request.query_params}:{request.headers.get('authorization', '')}"
        return hashlib.md5(key_data.encode()).hexdigest()
```

### 6. 모니터링 및 알림

#### 성능 임계값 모니터링

```python
from typing import Dict, Any, List
import asyncio
import smtplib
from email.mime.text import MIMEText

class PerformanceAlertManager:
    """성능 알림 관리자"""
    
    def __init__(self):
        self.thresholds = {
            "response_time": 2.0,  # 2초
            "memory_usage": 80.0,  # 80%
            "cpu_usage": 90.0,     # 90%
            "error_rate": 5.0,     # 5%
            "cache_hit_ratio": 0.7  # 70%
        }
        self.alert_history = []
        self.notification_channels = []
    
    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """알림 채널 추가"""
        self.notification_channels.append({
            "type": channel_type,
            "config": config
        })
    
    async def check_performance_metrics(self, metrics: Dict[str, Any]):
        """성능 메트릭 확인"""
        alerts = []
        
        # 응답 시간 확인
        if metrics.get("avg_response_time", 0) > self.thresholds["response_time"]:
            alerts.append({
                "type": "high_response_time",
                "message": f"평균 응답 시간이 {metrics['avg_response_time']:.2f}초로 임계값을 초과했습니다",
                "severity": "warning"
            })
        
        # 메모리 사용량 확인
        memory_usage = metrics.get("memory_usage_percent", 0)
        if memory_usage > self.thresholds["memory_usage"]:
            alerts.append({
                "type": "high_memory_usage",
                "message": f"메모리 사용량이 {memory_usage:.1f}%로 임계값을 초과했습니다",
                "severity": "critical" if memory_usage > 95 else "warning"
            })
        
        # CPU 사용량 확인
        cpu_usage = metrics.get("cpu_usage_percent", 0)
        if cpu_usage > self.thresholds["cpu_usage"]:
            alerts.append({
                "type": "high_cpu_usage",
                "message": f"CPU 사용량이 {cpu_usage:.1f}%로 임계값을 초과했습니다",
                "severity": "critical" if cpu_usage > 95 else "warning"
            })
        
        # 에러율 확인
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.thresholds["error_rate"]:
            alerts.append({
                "type": "high_error_rate",
                "message": f"에러율이 {error_rate:.1f}%로 임계값을 초과했습니다",
                "severity": "critical"
            })
        
        # 캐시 히트율 확인
        cache_hit_ratio = metrics.get("cache_hit_ratio", 1.0)
        if cache_hit_ratio < self.thresholds["cache_hit_ratio"]:
            alerts.append({
                "type": "low_cache_hit_ratio",
                "message": f"캐시 히트율이 {cache_hit_ratio:.1%}로 임계값 미만입니다",
                "severity": "warning"
            })
        
        # 알림 전송
        for alert in alerts:
            await self._send_alert(alert)
    
    async def _send_alert(self, alert: Dict[str, Any]):
        """알림 전송"""
        for channel in self.notification_channels:
            try:
                if channel["type"] == "email":
                    await self._send_email_alert(alert, channel["config"])
                elif channel["type"] == "slack":
                    await self._send_slack_alert(alert, channel["config"])
            except Exception as e:
                print(f"Alert sending failed: {e}")
    
    async def _send_email_alert(self, alert: Dict[str, Any], config: Dict[str, Any]):
        """이메일 알림 전송"""
        msg = MIMEText(alert["message"])
        msg["Subject"] = f"Bridge Performance Alert: {alert['type']}"
        msg["From"] = config["from_email"]
        msg["To"] = config["to_email"]
        
        # SMTP 서버를 통한 이메일 전송
        # 실제 구현에서는 비동기 SMTP 클라이언트 사용
        pass
    
    async def _send_slack_alert(self, alert: Dict[str, Any], config: Dict[str, Any]):
        """슬랙 알림 전송"""
        # 슬랙 웹훅을 통한 알림 전송
        # 실제 구현에서는 aiohttp를 사용한 비동기 요청
        pass
```

## 📊 성능 벤치마킹

### 1. 부하 테스트

```python
import asyncio
import aiohttp
import time
from typing import List, Dict, Any
import statistics

class LoadTester:
    """부하 테스트 도구"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.results = []
    
    async def run_load_test(self, endpoint: str, concurrent_users: int, 
                           duration_seconds: int, request_data: Dict[str, Any] = None):
        """부하 테스트 실행"""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(concurrent_users):
                task = asyncio.create_task(
                    self._simulate_user(session, endpoint, request_data, end_time)
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
        
        return self._analyze_results()
    
    async def _simulate_user(self, session: aiohttp.ClientSession, endpoint: str, 
                           request_data: Dict[str, Any], end_time: float):
        """사용자 시뮬레이션"""
        while time.time() < end_time:
            start_time = time.time()
            
            try:
                if request_data:
                    async with session.post(f"{self.base_url}{endpoint}", 
                                          json=request_data) as response:
                        await response.text()
                else:
                    async with session.get(f"{self.base_url}{endpoint}") as response:
                        await response.text()
                
                duration = time.time() - start_time
                self.results.append({
                    "timestamp": start_time,
                    "duration": duration,
                    "status_code": response.status,
                    "success": response.status < 400
                })
                
            except Exception as e:
                duration = time.time() - start_time
                self.results.append({
                    "timestamp": start_time,
                    "duration": duration,
                    "status_code": 0,
                    "success": False,
                    "error": str(e)
                })
            
            # 요청 간 간격
            await asyncio.sleep(0.1)
    
    def _analyze_results(self) -> Dict[str, Any]:
        """결과 분석"""
        if not self.results:
            return {}
        
        successful_requests = [r for r in self.results if r["success"]]
        failed_requests = [r for r in self.results if not r["success"]]
        
        durations = [r["duration"] for r in successful_requests]
        
        return {
            "total_requests": len(self.results),
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / len(self.results) if self.results else 0,
            "avg_response_time": statistics.mean(durations) if durations else 0,
            "min_response_time": min(durations) if durations else 0,
            "max_response_time": max(durations) if durations else 0,
            "p95_response_time": self._percentile(durations, 95) if durations else 0,
            "p99_response_time": self._percentile(durations, 99) if durations else 0,
            "requests_per_second": len(self.results) / (max(r["timestamp"] for r in self.results) - min(r["timestamp"] for r in self.results)) if self.results else 0
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """백분위수 계산"""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
```

### 2. 성능 프로파일링

```python
import cProfile
import pstats
from io import StringIO
import memory_profiler
import line_profiler

class PerformanceProfiler:
    """성능 프로파일러"""
    
    @staticmethod
    def profile_function(func, *args, **kwargs):
        """함수 프로파일링"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # 결과 출력
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        print("Function Profile:")
        print(s.getvalue())
        
        return result
    
    @staticmethod
    def memory_profile_function(func, *args, **kwargs):
        """메모리 프로파일링"""
        @memory_profiler.profile
        def wrapper():
            return func(*args, **kwargs)
        
        return wrapper()
    
    @staticmethod
    def line_profile_function(func, *args, **kwargs):
        """라인 프로파일링"""
        profiler = line_profiler.LineProfiler()
        profiler.add_function(func)
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        profiler.print_stats()
        
        return result
```

## 🔧 성능 최적화 체크리스트

### 1. 데이터베이스 최적화

- [ ] 인덱스 최적화
- [ ] 쿼리 최적화
- [ ] 연결 풀 설정
- [ ] 쿼리 결과 캐싱
- [ ] 배치 처리 구현

### 2. 애플리케이션 최적화

- [ ] 비동기 처리 구현
- [ ] 메모리 사용량 최적화
- [ ] 캐싱 전략 구현
- [ ] 응답 압축 설정
- [ ] 페이지네이션 구현

### 3. 인프라 최적화

- [ ] 로드 밸런싱 설정
- [ ] CDN 사용
- [ ] 캐시 서버 구성
- [ ] 모니터링 시스템 구축
- [ ] 자동 스케일링 설정

### 4. 모니터링 및 알림

- [ ] 성능 메트릭 수집
- [ ] 임계값 설정
- [ ] 알림 시스템 구축
- [ ] 대시보드 구성
- [ ] 로그 분석

이 성능 튜닝 가이드를 따라 Bridge 시스템의 성능을 최적화할 수 있습니다. 성능 최적화는 지속적인 프로세스이므로 정기적으로 모니터링하고 개선해야 합니다.
