# Bridge 개발자 가이드

## 📖 개요

이 가이드는 Bridge 프로젝트의 개발 환경 설정, 코드 구조, 개발 워크플로우, 그리고 새로운 기능 추가 방법에 대해 설명합니다.

## 🛠️ 개발 환경 설정

### 필수 요구사항

- **Python**: 3.11 이상
- **Node.js**: 18 이상 (문서화용)
- **Docker**: 20.10 이상 (선택사항)
- **Git**: 2.30 이상

### 초기 설정

```bash
# 저장소 클론
git clone https://github.com/your-org/bridge.git
cd bridge

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
make install

# 개발 서버 실행
make dev
```

### IDE 설정

#### VS Code 설정

`.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "100"],
  "python.sortImports.args": ["--profile", "black"],
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

#### PyCharm 설정

1. **프로젝트 인터프리터 설정**:
   - File → Settings → Project → Python Interpreter
   - 가상환경의 Python 인터프리터 선택

2. **코드 스타일 설정**:
   - File → Settings → Editor → Code Style → Python
   - Scheme: Black (설치 필요)

## 🏗️ 프로젝트 구조

```
src/bridge/
├── analytics/                 # 분석 도구
│   ├── core/                 # 핵심 분석 기능
│   │   ├── data_integration.py    # 데이터 통합
│   │   ├── statistics.py          # 통계 분석
│   │   ├── quality.py             # 품질 검사
│   │   └── visualization.py       # 시각화
│   ├── utils/                # 분석 유틸리티
│   └── tests/                # 분석 테스트
├── connectors/               # 데이터 커넥터
│   ├── base.py              # 기본 커넥터 클래스
│   ├── postgres.py          # PostgreSQL 커넥터
│   ├── mongodb.py           # MongoDB 커넥터
│   ├── elasticsearch.py     # Elasticsearch 커넥터
│   └── mock.py              # Mock 커넥터
├── orchestrator/            # FastAPI 오케스트레이터
│   ├── app.py               # FastAPI 애플리케이션
│   ├── routers.py           # API 라우터
│   ├── tasks.py             # Celery 태스크
│   └── queries.py           # 쿼리 유틸리티
├── governance/              # 데이터 거버넌스
│   ├── contracts.py         # 데이터 계약
│   ├── metadata.py          # 메타데이터 카탈로그
│   ├── rbac.py              # RBAC 시스템
│   └── audit.py             # 감사 로그
├── automation/              # 자동화 파이프라인
│   ├── quality_monitor.py   # 품질 모니터링
│   ├── report_automation.py # 리포트 자동화
│   ├── notification_system.py # 알림 시스템
│   └── scheduler.py         # 작업 스케줄러
├── dashboard/               # 대시보드 시스템
│   ├── dashboard_manager.py # 대시보드 관리
│   ├── monitoring_dashboard.py # 모니터링 대시보드
│   ├── real_time_monitor.py # 실시간 모니터링
│   └── visualization_engine.py # 시각화 엔진
└── mcp_server_unified.py    # 통합 MCP 서버
```

## 🔧 개발 워크플로우

### 1. 브랜치 전략

```bash
# 기능 개발
git checkout -b feature/new-feature

# 버그 수정
git checkout -b fix/bug-description

# 핫픽스
git checkout -b hotfix/critical-fix
```

### 2. 코드 품질 관리

```bash
# 코드 포맷팅
make fmt

# 린터 실행
make lint

# 타입 체크
mypy src/

# 보안 검사
bandit -r src/
```

### 3. 테스트 실행

```bash
# 전체 테스트
make test

# 특정 모듈 테스트
pytest tests/analytics/test_statistics.py -v

# 커버리지 포함 테스트
make test -- --cov=src --cov-report=html

# 통합 테스트
docker-compose -f docker-compose.dev.yml run --rm test
```

### 4. 커밋 및 PR

```bash
# 변경사항 스테이징
git add .

# 커밋 (컨벤션 준수)
git commit -m "feat(analytics): add correlation analysis"

# 푸시
git push origin feature/new-feature

# PR 생성
gh pr create --title "새로운 기능 추가" --body "상세 설명"
```

## 🧩 새로운 기능 추가

### 1. 새로운 커넥터 추가

```python
# src/bridge/connectors/new_database.py
from typing import Dict, Any, AsyncGenerator
from .base import BaseConnector

class NewDatabaseConnector(BaseConnector):
    """새로운 데이터베이스 커넥터"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._connection = None
    
    async def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            # 실제 연결 테스트 로직
            return True
        except Exception as e:
            self.logger.error(f"연결 테스트 실패: {e}")
            return False
    
    async def get_metadata(self) -> Dict[str, Any]:
        """메타데이터 수집"""
        try:
            # 메타데이터 수집 로직
            return {
                "tables": ["table1", "table2"],
                "version": "1.0"
            }
        except Exception as e:
            self.logger.error(f"메타데이터 수집 실패: {e}")
            return {}
    
    async def run_query(self, query: str, params: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """쿼리 실행"""
        try:
            # 쿼리 실행 로직
            async for row in self._execute_query(query, params):
                yield row
        except Exception as e:
            self.logger.error(f"쿼리 실행 실패: {e}")
            raise
    
    async def _execute_query(self, query: str, params: Dict[str, Any] = None):
        """실제 쿼리 실행 로직"""
        # 구현
        pass
```

### 2. 새로운 분석 도구 추가

```python
# src/bridge/analytics/core/new_analyzer.py
from typing import Dict, Any, List
from ..data_integration import UnifiedDataFrame

class NewAnalyzer:
    """새로운 분석 도구"""
    
    def __init__(self):
        self.name = "New Analyzer"
        self.version = "1.0"
    
    def analyze(self, data: UnifiedDataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """분석 수행"""
        try:
            pandas_df = data.to_pandas()
            
            # 분석 로직 구현
            result = self._perform_analysis(pandas_df, config)
            
            return {
                "status": "success",
                "result": result,
                "metadata": {
                    "analyzer": self.name,
                    "version": self.version,
                    "data_shape": pandas_df.shape
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _perform_analysis(self, df, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """실제 분석 로직"""
        # 구현
        pass
```

### 3. 새로운 API 엔드포인트 추가

```python
# src/bridge/orchestrator/routers.py
from fastapi import APIRouter, HTTPException, Depends
from ..semantic.models import TaskRequest, TaskResponse
from ..auth import get_current_user

router = APIRouter()

@router.post("/new-endpoint", response_model=TaskResponse)
async def new_endpoint(
    request: TaskRequest,
    current_user: dict = Depends(get_current_user)
) -> TaskResponse:
    """새로운 API 엔드포인트"""
    try:
        # 비즈니스 로직 구현
        result = await process_new_request(request)
        
        return TaskResponse(
            status="success",
            data=result,
            message="요청이 성공적으로 처리되었습니다"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"요청 처리 중 오류가 발생했습니다: {str(e)}"
        )

async def process_new_request(request: TaskRequest) -> Dict[str, Any]:
    """새로운 요청 처리 로직"""
    # 구현
    pass
```

## 🧪 테스트 작성

### 1. 단위 테스트

```python
# tests/analytics/test_new_analyzer.py
import pytest
from src.bridge.analytics.core.new_analyzer import NewAnalyzer
from src.bridge.analytics.core.data_integration import UnifiedDataFrame
import pandas as pd

class TestNewAnalyzer:
    """NewAnalyzer 테스트 클래스"""
    
    def setup_method(self):
        """테스트 설정"""
        self.analyzer = NewAnalyzer()
        self.sample_data = pd.DataFrame({
            'value1': [1, 2, 3, 4, 5],
            'value2': [10, 20, 30, 40, 50]
        })
        self.unified_df = UnifiedDataFrame(self.sample_data)
    
    def test_analyze_success(self):
        """성공적인 분석 테스트"""
        result = self.analyzer.analyze(self.unified_df)
        
        assert result["status"] == "success"
        assert "result" in result
        assert "metadata" in result
    
    def test_analyze_with_config(self):
        """설정이 있는 분석 테스트"""
        config = {"param1": "value1"}
        result = self.analyzer.analyze(self.unified_df, config)
        
        assert result["status"] == "success"
    
    def test_analyze_error_handling(self):
        """에러 처리 테스트"""
        # 잘못된 데이터로 테스트
        invalid_data = UnifiedDataFrame(pd.DataFrame())
        result = self.analyzer.analyze(invalid_data)
        
        assert result["status"] == "error"
        assert "error" in result
```

### 2. 통합 테스트

```python
# tests/integration/test_new_feature_integration.py
import pytest
from fastapi.testclient import TestClient
from src.bridge.orchestrator.app import app

client = TestClient(app)

class TestNewFeatureIntegration:
    """새로운 기능 통합 테스트"""
    
    def test_new_endpoint_success(self):
        """새 엔드포인트 성공 테스트"""
        response = client.post(
            "/api/v1/new-endpoint",
            json={
                "intent": "테스트 요청",
                "sources": ["mock://test"],
                "required_tools": ["new_analyzer"]
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
    
    def test_new_endpoint_validation_error(self):
        """새 엔드포인트 검증 오류 테스트"""
        response = client.post(
            "/api/v1/new-endpoint",
            json={},  # 잘못된 요청
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 422
    
    def test_new_endpoint_authentication_error(self):
        """새 엔드포인트 인증 오류 테스트"""
        response = client.post(
            "/api/v1/new-endpoint",
            json={
                "intent": "테스트 요청",
                "sources": ["mock://test"]
            }
            # 인증 헤더 없음
        )
        
        assert response.status_code == 401
```

### 3. 성능 테스트

```python
# tests/performance/test_new_feature_performance.py
import pytest
import time
from src.bridge.analytics.core.new_analyzer import NewAnalyzer
from src.bridge.analytics.core.data_integration import UnifiedDataFrame
import pandas as pd

class TestNewFeaturePerformance:
    """새로운 기능 성능 테스트"""
    
    def test_analyzer_performance(self):
        """분석기 성능 테스트"""
        analyzer = NewAnalyzer()
        
        # 대용량 데이터 생성
        large_data = pd.DataFrame({
            'value1': range(10000),
            'value2': range(10000, 20000)
        })
        unified_df = UnifiedDataFrame(large_data)
        
        # 성능 측정
        start_time = time.time()
        result = analyzer.analyze(unified_df)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # 성능 검증 (1초 이내)
        assert execution_time < 1.0
        assert result["status"] == "success"
    
    def test_memory_usage(self):
        """메모리 사용량 테스트"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 대용량 데이터 처리
        analyzer = NewAnalyzer()
        large_data = pd.DataFrame({
            'value1': range(100000),
            'value2': range(100000, 200000)
        })
        unified_df = UnifiedDataFrame(large_data)
        
        result = analyzer.analyze(unified_df)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 메모리 증가량 검증 (100MB 이내)
        assert memory_increase < 100 * 1024 * 1024
```

## 🔍 디버깅

### 1. 로그 설정

```python
# src/bridge/logging_config.py
import logging
import sys
from pathlib import Path

def setup_logging(level: str = "INFO"):
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger("bridge")
    logger.setLevel(getattr(logging, level.upper()))
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_dir / "bridge.log")
    file_handler.setLevel(logging.INFO)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # 포맷터
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```

### 2. 디버깅 도구

```python
# 디버깅용 코드
import logging
from src.bridge.logging_config import setup_logging

# 로깅 설정
logger = setup_logging("DEBUG")

def debug_function():
    """디버깅 함수"""
    logger.debug("함수 시작")
    
    try:
        # 비즈니스 로직
        result = process_data()
        logger.debug(f"처리 결과: {result}")
        
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
        raise
    
    logger.debug("함수 완료")
    return result
```

### 3. 프로파일링

```python
# 성능 프로파일링
import cProfile
import pstats
from io import StringIO

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
    
    print(s.getvalue())
    return result

# 사용 예시
result = profile_function(analyzer.analyze, unified_df)
```

## 📚 문서화

### 1. 코드 문서화

```python
def analyze_data(data: UnifiedDataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    데이터를 분석합니다.
    
    Args:
        data: 분석할 데이터 (UnifiedDataFrame)
        config: 분석 설정 (선택사항)
            - param1: 첫 번째 매개변수
            - param2: 두 번째 매개변수
    
    Returns:
        분석 결과 딕셔너리:
            - status: 분석 상태 ("success" 또는 "error")
            - result: 분석 결과 데이터
            - metadata: 메타데이터
    
    Raises:
        ValueError: 잘못된 데이터가 제공된 경우
        RuntimeError: 분석 중 오류가 발생한 경우
    
    Example:
        >>> data = UnifiedDataFrame(pd.DataFrame({'value': [1, 2, 3]}))
        >>> result = analyze_data(data)
        >>> print(result['status'])
        'success'
    """
    # 구현
    pass
```

### 2. API 문서화

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

class AnalysisRequest(BaseModel):
    """분석 요청 모델"""
    data_source: str = Field(..., description="데이터 소스 URI")
    table_name: str = Field(..., description="테이블명")
    columns: List[str] = Field(..., description="분석할 컬럼 목록")
    config: Dict[str, Any] = Field(default={}, description="분석 설정")

class AnalysisResponse(BaseModel):
    """분석 응답 모델"""
    status: str = Field(..., description="분석 상태")
    result: Dict[str, Any] = Field(..., description="분석 결과")
    metadata: Dict[str, Any] = Field(..., description="메타데이터")

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_data_endpoint(
    request: AnalysisRequest,
    current_user: dict = Depends(get_current_user)
) -> AnalysisResponse:
    """
    데이터를 분석합니다.
    
    - **data_source**: 데이터베이스 연결 문자열
    - **table_name**: 분석할 테이블명
    - **columns**: 분석할 컬럼 목록
    - **config**: 분석 설정 (선택사항)
    
    Returns:
        AnalysisResponse: 분석 결과
    """
    # 구현
    pass
```

## 🚀 배포

### 1. Docker 이미지 빌드

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 의존성 설치
COPY pyproject.toml .
RUN pip install -e .

# 애플리케이션 코드 복사
COPY src/ ./src/
COPY scripts/ ./scripts/

# 포트 노출
EXPOSE 8000

# 실행 명령
CMD ["python", "-m", "src.bridge.orchestrator.app"]
```

### 2. Docker Compose 설정

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  bridge-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - BRIDGE_DATABASE_URL=postgresql://user:pass@db:5432/bridge
      - BRIDGE_REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
  
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=bridge
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### 3. Kubernetes 배포

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bridge-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bridge-api
  template:
    metadata:
      labels:
        app: bridge-api
    spec:
      containers:
      - name: bridge-api
        image: bridge:latest
        ports:
        - containerPort: 8000
        env:
        - name: BRIDGE_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: bridge-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## 🔧 유지보수

### 1. 의존성 업데이트

```bash
# 의존성 확인
pip list --outdated

# 특정 패키지 업데이트
pip install --upgrade package_name

# requirements.txt 업데이트
pip freeze > requirements.txt
```

### 2. 데이터베이스 마이그레이션

```python
# migrations/001_add_new_table.py
from sqlalchemy import create_engine, text

def upgrade(engine):
    """마이그레이션 업그레이드"""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE new_table (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()

def downgrade(engine):
    """마이그레이션 다운그레이드"""
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE new_table"))
        conn.commit()
```

### 3. 모니터링 설정

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# 메트릭 정의
REQUEST_COUNT = Counter('bridge_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('bridge_request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('bridge_active_connections', 'Active connections')

def track_request(func):
    """요청 추적 데코레이터"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            REQUEST_COUNT.labels(method='GET', endpoint=func.__name__).inc()
            return result
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
    
    return wrapper
```

## 🤝 기여 가이드

### 1. 코드 리뷰 체크리스트

- [ ] 코드가 PEP 8 스타일 가이드를 따르는가?
- [ ] 타입 힌트가 적절히 사용되었는가?
- [ ] 테스트 코드가 작성되었는가?
- [ ] 문서화가 충분한가?
- [ ] 에러 처리가 적절한가?
- [ ] 성능에 문제가 없는가?

### 2. PR 템플릿

```markdown
## 변경사항
- [ ] 새로운 기능 추가
- [ ] 버그 수정
- [ ] 문서 업데이트
- [ ] 테스트 추가

## 설명
변경사항에 대한 자세한 설명

## 테스트
- [ ] 단위 테스트 통과
- [ ] 통합 테스트 통과
- [ ] 수동 테스트 완료

## 체크리스트
- [ ] 코드 스타일 준수
- [ ] 타입 힌트 포함
- [ ] 문서화 완료
- [ ] 테스트 커버리지 유지
```

이 가이드를 따라 개발하면 Bridge 프로젝트에 효과적으로 기여할 수 있습니다. 추가 질문이나 도움이 필요하면 언제든 문의하세요!
