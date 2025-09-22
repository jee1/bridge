# 통합된 Bridge MCP 서버

## 개요

1개 통합 서버 + 7개 개별 서버를 환경 변수 기반으로 다양한 모드를 지원하는 통합된 서버로 구성합니다.

## 환경 변수 설정

### 기본 설정
- `BRIDGE_MCP_MODE`: 서버 모드 (`development`, `production`, `real`, `mock`)
- `BRIDGE_MCP_USE_SDK`: MCP SDK 사용 여부 (`true`, `false`)
- `BRIDGE_MCP_ERROR_RECOVERY`: 에러 복구 기능 (`true`, `false`)
- `BRIDGE_MCP_SERVER_NAME`: 서버 이름 (기본값: `bridge-mcp-unified`)
- `BRIDGE_MCP_VERSION`: 서버 버전 (기본값: `1.0.0`)

### 데이터베이스 연결 설정
- `BRIDGE_POSTGRES_HOST`, `BRIDGE_POSTGRES_PORT`, `BRIDGE_POSTGRES_DB`, `BRIDGE_POSTGRES_USER`, `BRIDGE_POSTGRES_PASSWORD`
- `BRIDGE_MYSQL_HOST`, `BRIDGE_MYSQL_PORT`, `BRIDGE_MYSQL_DB`, `BRIDGE_MYSQL_USER`, `BRIDGE_MYSQL_PASSWORD`
- `BRIDGE_ELASTICSEARCH_HOST`, `BRIDGE_ELASTICSEARCH_PORT`, `BRIDGE_ELASTICSEARCH_USE_SSL`, `BRIDGE_ELASTICSEARCH_USERNAME`, `BRIDGE_ELASTICSEARCH_PASSWORD`, `BRIDGE_ELASTICSEARCH_URL`

## 지원 모드

### 1. Development 모드
```bash
BRIDGE_MCP_MODE=development
BRIDGE_MCP_USE_SDK=true
BRIDGE_MCP_ERROR_RECOVERY=false
```
- MCP SDK 사용
- 모의 응답
- 빠른 개발/테스트

### 2. Production 모드
```bash
BRIDGE_MCP_MODE=production
BRIDGE_MCP_USE_SDK=true
BRIDGE_MCP_ERROR_RECOVERY=true
```
- MCP SDK 사용
- 실제 데이터베이스 연동
- 에러 복구 및 시그널 핸들링

### 3. Real 모드
```bash
BRIDGE_MCP_MODE=real
BRIDGE_MCP_USE_SDK=false
BRIDGE_MCP_ERROR_RECOVERY=true
```
- 직접 JSON-RPC 구현
- 실제 데이터베이스 연동
- 에러 복구 기능

### 4. Mock 모드
```bash
BRIDGE_MCP_MODE=mock
BRIDGE_MCP_USE_SDK=false
BRIDGE_MCP_ERROR_RECOVERY=false
```
- 직접 JSON-RPC 구현
- 모의 응답
- 간단한 테스트

## 실행 방법

### 직접 실행
```bash
# 기본 설정으로 실행
python -m src.bridge.mcp_server_unified

# 환경 변수와 함께 실행
BRIDGE_MCP_MODE=real python -m src.bridge.mcp_server_unified
```

### Docker Compose 사용
```bash
# 개발용
docker-compose -f docker-compose.mcp-modes.yml --profile mcp-dev up

# 프로덕션용
docker-compose -f docker-compose.mcp-modes.yml --profile mcp-prod up

# 실제 DB 연동
docker-compose -f docker-compose.mcp-modes.yml --profile mcp-real up

# 간단한 모드
docker-compose -f docker-compose.mcp-modes.yml --profile mcp-simple up
```

## 지원 도구

1. **query_database**: 데이터베이스 쿼리 실행
2. **get_schema**: 데이터베이스 스키마 정보 조회
3. **analyze_data**: 데이터 분석 및 인사이트 생성
4. **list_connectors**: 사용 가능한 커넥터 목록 조회

## 마이그레이션 가이드

### 기존 서버에서 통합 서버로 전환

1. **mcp_server.py**: 하위 호환성 유지 (통합 서버로 리다이렉트)
2. **mcp_server_robust.py**: `BRIDGE_MCP_MODE=production` + `BRIDGE_MCP_ERROR_RECOVERY=true`
3. **mcp_server_real.py**: `BRIDGE_MCP_MODE=real` + `BRIDGE_MCP_USE_SDK=false`
4. **mcp_server_simple.py**: `BRIDGE_MCP_MODE=mock` + `BRIDGE_MCP_USE_SDK=false`

### 삭제된 파일들
- `mcp_server_fixed.py` (mcp_server.py와 중복)
- `mcp_server_working.py` (mcp_server_minimal.py와 중복)
- `mcp_server_minimal.py` (mcp_server_simple.py와 중복)

## 장점

1. **코드 중복 제거**: 8개 개별 파일 → 1개 통합 파일
2. **유지보수 용이**: 한 곳에서 모든 기능 관리
3. **설정 기반**: 환경 변수로 모드 변경
4. **확장성**: 새로운 모드 추가 용이
5. **하위 호환성**: 기존 코드와 호환
