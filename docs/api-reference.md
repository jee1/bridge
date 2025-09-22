# Bridge API 참조 문서

## 📖 개요

Bridge API는 RESTful 인터페이스를 통해 데이터 분석, 커넥터 관리, 작업 스케줄링 등의 기능을 제공합니다.

## 🔗 기본 정보

- **Base URL**: `http://localhost:8000`
- **API 버전**: v1
- **인증**: Bearer Token (API Key)
- **Content-Type**: `application/json`

## 🔐 인증

모든 API 요청에는 Authorization 헤더가 필요합니다:

```http
Authorization: Bearer YOUR_API_KEY
```

## 📊 데이터 분석 API

### 통계 분석

#### POST /api/v1/analytics/statistics

기술 통계를 계산합니다.

**요청 본문:**
```json
{
  "data_source": "postgres://analytics_db",
  "table_name": "sales",
  "columns": ["amount", "profit"],
  "analysis_type": "descriptive"
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "amount": {
      "count": 1000,
      "mean": 1250.50,
      "std": 300.25,
      "min": 100.00,
      "max": 2500.00,
      "median": 1200.00
    },
    "profit": {
      "count": 1000,
      "mean": 250.10,
      "std": 60.05,
      "min": 20.00,
      "max": 500.00,
      "median": 240.00
    }
  }
}
```

#### POST /api/v1/analytics/correlation

상관관계 분석을 수행합니다.

**요청 본문:**
```json
{
  "data_source": "postgres://analytics_db",
  "table_name": "sales",
  "columns": ["amount", "profit", "quantity"]
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "correlation_matrix": {
      "amount": {
        "profit": 0.85,
        "quantity": 0.72
      },
      "profit": {
        "amount": 0.85,
        "quantity": 0.68
      },
      "quantity": {
        "amount": 0.72,
        "profit": 0.68
      }
    }
  }
}
```

### 데이터 품질 검사

#### POST /api/v1/analytics/quality

데이터 품질을 검사합니다.

**요청 본문:**
```json
{
  "data_source": "postgres://analytics_db",
  "table_name": "customers",
  "checks": ["missing_values", "outliers", "consistency"]
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "overall_score": 85.5,
    "missing_values": {
      "total_missing": 50,
      "missing_ratio": 0.05,
      "columns_affected": ["email", "phone"]
    },
    "outliers": {
      "total_outliers": 25,
      "outlier_ratio": 0.025,
      "columns_affected": ["age", "income"]
    },
    "consistency": {
      "consistency_score": 0.92,
      "issues": ["duplicate_emails", "invalid_phone_format"]
    },
    "recommendations": [
      "이메일 필드의 결측값을 처리하세요",
      "나이 필드의 이상치를 검토하세요"
    ]
  }
}
```

### 시각화

#### POST /api/v1/analytics/charts

차트를 생성합니다.

**요청 본문:**
```json
{
  "data_source": "postgres://analytics_db",
  "table_name": "sales",
  "chart_type": "bar",
  "x_axis": "region",
  "y_axis": "amount",
  "title": "지역별 매출",
  "width": 800,
  "height": 600
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "chart_id": "chart_123",
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "metadata": {
      "chart_type": "bar",
      "data_points": 5,
      "generated_at": "2024-01-15T10:30:00Z"
    }
  }
}
```

#### POST /api/v1/analytics/dashboards

대시보드를 생성합니다.

**요청 본문:**
```json
{
  "title": "매출 분석 대시보드",
  "layout": "grid",
  "grid_columns": 2,
  "grid_rows": 2,
  "widgets": [
    {
      "id": "sales_chart",
      "type": "chart",
      "title": "월별 매출",
      "position": {"x": 0, "y": 0, "width": 1, "height": 1},
      "config": {
        "chart_type": "line",
        "data_source": "monthly_sales"
      }
    },
    {
      "id": "region_metric",
      "type": "metric",
      "title": "활성 지역 수",
      "position": {"x": 1, "y": 0, "width": 1, "height": 1},
      "config": {
        "value": 12,
        "unit": "개"
      }
    }
  ]
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "dashboard_id": "dashboard_456",
    "html_content": "<html>...</html>",
    "widgets": [
      {
        "id": "sales_chart",
        "rendered": true,
        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
      },
      {
        "id": "region_metric",
        "rendered": true,
        "value": "12개"
      }
    ]
  }
}
```

## 🔌 커넥터 API

### 커넥터 관리

#### GET /api/v1/connectors

등록된 커넥터 목록을 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "connectors": [
      {
        "id": "postgres_analytics",
        "type": "postgres",
        "name": "Analytics Database",
        "host": "localhost",
        "port": 5432,
        "database": "analytics",
        "status": "connected",
        "last_checked": "2024-01-15T10:30:00Z"
      },
      {
        "id": "mongodb_logs",
        "type": "mongodb",
        "name": "Logs Database",
        "host": "localhost",
        "port": 27017,
        "database": "logs",
        "status": "connected",
        "last_checked": "2024-01-15T10:25:00Z"
      }
    ]
  }
}
```

#### POST /api/v1/connectors

새 커넥터를 등록합니다.

**요청 본문:**
```json
{
  "id": "new_postgres",
  "type": "postgres",
  "name": "New Database",
  "host": "localhost",
  "port": 5432,
  "database": "new_db",
  "username": "user",
  "password": "password"
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "connector_id": "new_postgres",
    "message": "커넥터가 성공적으로 등록되었습니다"
  }
}
```

#### GET /api/v1/connectors/{connector_id}

특정 커넥터의 상세 정보를 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "id": "postgres_analytics",
    "type": "postgres",
    "name": "Analytics Database",
    "host": "localhost",
    "port": 5432,
    "database": "analytics",
    "status": "connected",
    "last_checked": "2024-01-15T10:30:00Z",
    "metadata": {
      "tables": [
        {
          "name": "customers",
          "columns": 15,
          "rows": 10000
        },
        {
          "name": "sales",
          "columns": 8,
          "rows": 50000
        }
      ]
    }
  }
}
```

#### POST /api/v1/connectors/{connector_id}/test

커넥터 연결을 테스트합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "connected": true,
    "response_time_ms": 45,
    "message": "연결이 성공적으로 확인되었습니다"
  }
}
```

#### GET /api/v1/connectors/{connector_id}/metadata

커넥터의 메타데이터를 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "tables": [
      {
        "name": "customers",
        "schema": "public",
        "columns": [
          {
            "name": "customer_id",
            "type": "integer",
            "nullable": false,
            "primary_key": true
          },
          {
            "name": "email",
            "type": "varchar",
            "nullable": false,
            "primary_key": false
          }
        ],
        "row_count": 10000,
        "size_mb": 25.5
      }
    ],
    "last_updated": "2024-01-15T10:30:00Z"
  }
}
```

#### POST /api/v1/connectors/{connector_id}/query

커넥터를 통해 쿼리를 실행합니다.

**요청 본문:**
```json
{
  "query": "SELECT * FROM customers WHERE region = :region LIMIT 10",
  "params": {
    "region": "North"
  }
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "rows": [
      {
        "customer_id": 1,
        "email": "john@example.com",
        "region": "North"
      },
      {
        "customer_id": 2,
        "email": "jane@example.com",
        "region": "North"
      }
    ],
    "total_rows": 2,
    "execution_time_ms": 120
  }
}
```

## 📋 작업 관리 API

### 작업 계획

#### POST /api/v1/tasks/plan

새로운 분석 작업을 계획합니다.

**요청 본문:**
```json
{
  "intent": "고객 세그먼트 분석",
  "sources": ["postgres://analytics_db"],
  "required_tools": ["sql_executor", "statistics_analyzer"],
  "context": {
    "time_range": "2024-01-01 to 2024-12-31",
    "customer_segments": ["premium", "standard"]
  }
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "job_id": "2f7c18af-1234-5678-9abc-def012345678",
    "state": "PENDING",
    "ready": false,
    "estimated_duration": "5-10 minutes",
    "steps": [
      {
        "step_id": 1,
        "description": "고객 데이터 조회",
        "tool": "sql_executor",
        "status": "pending"
      },
      {
        "step_id": 2,
        "description": "세그먼트별 통계 분석",
        "tool": "statistics_analyzer",
        "status": "pending"
      }
    ]
  }
}
```

### 작업 상태 조회

#### GET /api/v1/tasks/{job_id}

작업의 현재 상태를 조회합니다.

**응답 (대기 중):**
```json
{
  "status": "success",
  "data": {
    "job_id": "2f7c18af-1234-5678-9abc-def012345678",
    "state": "PENDING",
    "ready": false,
    "successful": false,
    "progress": 0,
    "current_step": "작업 대기 중"
  }
}
```

**응답 (진행 중):**
```json
{
  "status": "success",
  "data": {
    "job_id": "2f7c18af-1234-5678-9abc-def012345678",
    "state": "PROGRESS",
    "ready": false,
    "successful": false,
    "progress": 50,
    "current_step": "세그먼트별 통계 분석 중"
  }
}
```

**응답 (완료):**
```json
{
  "status": "success",
  "data": {
    "job_id": "2f7c18af-1234-5678-9abc-def012345678",
    "state": "SUCCESS",
    "ready": true,
    "successful": true,
    "progress": 100,
    "result": {
      "segments": [
        {
          "name": "premium",
          "count": 1500,
          "avg_value": 2500.50,
          "growth_rate": 0.15
        },
        {
          "name": "standard",
          "count": 3500,
          "avg_value": 800.25,
          "growth_rate": 0.08
        }
      ],
      "insights": [
        "프리미엄 고객의 평균 가치가 표준 고객보다 3배 높습니다",
        "프리미엄 고객의 성장률이 표준 고객보다 2배 빠릅니다"
      ]
    }
  }
}
```

**응답 (실패):**
```json
{
  "status": "error",
  "data": {
    "job_id": "2f7c18af-1234-5678-9abc-def012345678",
    "state": "FAILURE",
    "ready": true,
    "successful": false,
    "error": "데이터베이스 연결 오류",
    "error_details": {
      "error_type": "ConnectionError",
      "error_code": "DB_CONNECTION_FAILED",
      "suggestions": [
        "데이터베이스 서버 상태를 확인하세요",
        "연결 설정을 검토하세요"
      ]
    }
  }
}
```

### 작업 목록 조회

#### GET /api/v1/tasks

작업 목록을 조회합니다.

**쿼리 파라미터:**
- `status`: 작업 상태 필터 (pending, progress, success, failure)
- `limit`: 결과 수 제한 (기본값: 20)
- `offset`: 결과 오프셋 (기본값: 0)

**응답:**
```json
{
  "status": "success",
  "data": {
    "tasks": [
      {
        "job_id": "2f7c18af-1234-5678-9abc-def012345678",
        "intent": "고객 세그먼트 분석",
        "state": "SUCCESS",
        "created_at": "2024-01-15T10:00:00Z",
        "completed_at": "2024-01-15T10:05:00Z",
        "duration_seconds": 300
      }
    ],
    "total": 1,
    "limit": 20,
    "offset": 0
  }
}
```

## 🔒 거버넌스 API

### 데이터 계약 관리

#### GET /api/v1/governance/contracts

데이터 계약 목록을 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "contracts": [
      {
        "id": "customer_contract",
        "name": "고객 데이터 계약",
        "version": "1.0",
        "data_source": "postgres://analytics_db",
        "table_name": "customers",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-15T10:30:00Z"
      }
    ]
  }
}
```

#### POST /api/v1/governance/contracts

새 데이터 계약을 생성합니다.

**요청 본문:**
```json
{
  "id": "sales_contract",
  "name": "매출 데이터 계약",
  "version": "1.0",
  "description": "매출 데이터 품질 계약",
  "data_source": "postgres://analytics_db",
  "table_name": "sales",
  "fields": [
    {
      "name": "sale_id",
      "data_type": "integer",
      "required": true,
      "description": "매출 ID"
    },
    {
      "name": "amount",
      "data_type": "decimal",
      "required": true,
      "description": "매출 금액"
    }
  ],
  "quality_rules": [
    {
      "field_name": "amount",
      "rule_type": "range",
      "rule_value": "0,1000000",
      "description": "매출 금액은 0 이상 1,000,000 이하여야 합니다"
    }
  ]
}
```

#### POST /api/v1/governance/contracts/{contract_id}/validate

데이터 계약을 검증합니다.

**요청 본문:**
```json
{
  "data_source": "postgres://analytics_db",
  "table_name": "sales",
  "sample_size": 1000
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "is_valid": true,
    "validation_score": 95.5,
    "issues": [],
    "recommendations": [
      "데이터 품질이 우수합니다"
    ]
  }
}
```

### RBAC 관리

#### GET /api/v1/governance/users

사용자 목록을 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "users": [
      {
        "id": "user123",
        "username": "john_doe",
        "email": "john@example.com",
        "roles": ["analyst", "viewer"],
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": "2024-01-15T09:30:00Z"
      }
    ]
  }
}
```

#### POST /api/v1/governance/users

새 사용자를 생성합니다.

**요청 본문:**
```json
{
  "username": "jane_smith",
  "email": "jane@example.com",
  "roles": ["analyst"]
}
```

#### GET /api/v1/governance/users/{user_id}/permissions

사용자의 권한을 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "user_id": "user123",
    "permissions": [
      {
        "resource_type": "table",
        "resource_id": "customers",
        "actions": ["read", "query"]
      },
      {
        "resource_type": "table",
        "resource_id": "sales",
        "actions": ["read", "query", "analyze"]
      }
    ]
  }
}
```

### 감사 로그

#### GET /api/v1/governance/audit/events

감사 이벤트를 조회합니다.

**쿼리 파라미터:**
- `user_id`: 사용자 ID 필터
- `event_type`: 이벤트 타입 필터
- `start_date`: 시작 날짜 (ISO 8601)
- `end_date`: 종료 날짜 (ISO 8601)
- `limit`: 결과 수 제한 (기본값: 100)

**응답:**
```json
{
  "status": "success",
  "data": {
    "events": [
      {
        "id": "event_123",
        "event_type": "data_access",
        "user_id": "user123",
        "resource_type": "table",
        "resource_id": "customers",
        "action": "query",
        "timestamp": "2024-01-15T10:30:00Z",
        "details": {
          "query": "SELECT * FROM customers WHERE region = 'North'",
          "rows_returned": 150
        }
      }
    ],
    "total": 1,
    "limit": 100
  }
}
```

## 🤖 자동화 API

### 품질 모니터링

#### GET /api/v1/automation/quality/status

품질 모니터링 상태를 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "monitoring_status": "running",
    "check_interval": 300,
    "thresholds_count": 5,
    "monitoring_tasks_count": 3,
    "alerts_count": 2,
    "unacknowledged_alerts": 1,
    "last_check": "2024-01-15T10:30:00Z"
  }
}
```

#### POST /api/v1/automation/quality/thresholds

품질 임계값을 추가합니다.

**요청 본문:**
```json
{
  "id": "cpu_threshold",
  "name": "CPU 사용률 임계값",
  "metric_type": "cpu_percent",
  "threshold_value": 80.0,
  "operator": "gt",
  "severity": "warning"
}
```

### 리포트 자동화

#### GET /api/v1/automation/reports/templates

리포트 템플릿 목록을 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "templates": [
      {
        "id": "daily_sales_report",
        "name": "일일 매출 리포트",
        "description": "매일 자동 생성되는 매출 분석 리포트",
        "schedule": "0 9 * * *",
        "last_executed": "2024-01-15T09:00:00Z",
        "next_execution": "2024-01-16T09:00:00Z"
      }
    ]
  }
}
```

#### POST /api/v1/automation/reports/execute

리포트를 수동으로 실행합니다.

**요청 본문:**
```json
{
  "template_id": "daily_sales_report",
  "parameters": {
    "date": "2024-01-15"
  }
}
```

**응답:**
```json
{
  "status": "success",
  "data": {
    "job_id": "report_job_789",
    "status": "started",
    "estimated_completion": "2024-01-15T10:35:00Z"
  }
}
```

## 📊 대시보드 API

### 대시보드 관리

#### GET /api/v1/dashboards

대시보드 목록을 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "dashboards": [
      {
        "id": "sales_dashboard",
        "name": "매출 분석 대시보드",
        "description": "매출 관련 지표 및 차트를 보여주는 대시보드",
        "layout_type": "grid",
        "widgets_count": 4,
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-15T10:30:00Z"
      }
    ]
  }
}
```

#### POST /api/v1/dashboards

새 대시보드를 생성합니다.

**요청 본문:**
```json
{
  "id": "new_dashboard",
  "name": "새 대시보드",
  "description": "새로 생성된 대시보드",
  "layout_type": "grid",
  "grid_columns": 2,
  "grid_rows": 2,
  "widgets": [
    {
      "id": "chart_widget",
      "widget_type": "chart",
      "title": "매출 차트",
      "position": {"x": 0, "y": 0, "width": 1, "height": 1},
      "config": {
        "chart_type": "line",
        "data_source": "monthly_sales"
      }
    }
  ]
}
```

#### GET /api/v1/dashboards/{dashboard_id}

특정 대시보드의 상세 정보를 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "id": "sales_dashboard",
    "name": "매출 분석 대시보드",
    "description": "매출 관련 지표 및 차트를 보여주는 대시보드",
    "layout_type": "grid",
    "grid_columns": 2,
    "grid_rows": 2,
    "widgets": [
      {
        "id": "chart_widget",
        "widget_type": "chart",
        "title": "매출 차트",
        "position": {"x": 0, "y": 0, "width": 1, "height": 1},
        "config": {
          "chart_type": "line",
          "data_source": "monthly_sales"
        },
        "rendered": true,
        "last_updated": "2024-01-15T10:30:00Z"
      }
    ],
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
  }
}
```

### 실시간 모니터링

#### GET /api/v1/monitoring/status

시스템 모니터링 상태를 조회합니다.

**응답:**
```json
{
  "status": "success",
  "data": {
    "system_metrics": {
      "cpu_percent": 45.2,
      "memory_percent": 67.8,
      "disk_percent": 34.5,
      "network_io": {
        "bytes_sent": 1024000,
        "bytes_recv": 2048000
      }
    },
    "application_metrics": {
      "active_connections": 25,
      "requests_per_minute": 150,
      "error_rate": 0.02,
      "average_response_time": 250
    },
    "alerts": [
      {
        "id": "alert_123",
        "level": "warning",
        "message": "메모리 사용률이 70%를 초과했습니다",
        "timestamp": "2024-01-15T10:25:00Z"
      }
    ]
  }
}
```

#### GET /api/v1/monitoring/metrics/history

메트릭 히스토리를 조회합니다.

**쿼리 파라미터:**
- `metric_type`: 메트릭 타입 (cpu, memory, disk, network)
- `hours`: 조회할 시간 범위 (기본값: 24)

**응답:**
```json
{
  "status": "success",
  "data": {
    "metric_type": "cpu",
    "time_range": "24h",
    "data_points": [
      {
        "timestamp": "2024-01-15T10:00:00Z",
        "value": 42.5
      },
      {
        "timestamp": "2024-01-15T10:05:00Z",
        "value": 45.2
      }
    ]
  }
}
```

## 🚨 에러 응답

모든 API는 일관된 에러 응답 형식을 사용합니다:

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "요청 데이터가 유효하지 않습니다",
    "details": {
      "field": "email",
      "issue": "이메일 형식이 올바르지 않습니다"
    }
  }
}
```

### 일반적인 에러 코드

- `VALIDATION_ERROR`: 요청 데이터 검증 실패
- `AUTHENTICATION_ERROR`: 인증 실패
- `AUTHORIZATION_ERROR`: 권한 부족
- `NOT_FOUND`: 리소스를 찾을 수 없음
- `CONNECTION_ERROR`: 데이터베이스 연결 실패
- `QUERY_ERROR`: 쿼리 실행 실패
- `INTERNAL_ERROR`: 내부 서버 오류

## 📝 요청/응답 예시

### cURL 예시

```bash
# 통계 분석 요청
curl -X POST "http://localhost:8000/api/v1/analytics/statistics" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "data_source": "postgres://analytics_db",
    "table_name": "sales",
    "columns": ["amount", "profit"],
    "analysis_type": "descriptive"
  }'

# 작업 상태 조회
curl -H "Authorization: Bearer YOUR_API_KEY" \
  "http://localhost:8000/api/v1/tasks/2f7c18af-1234-5678-9abc-def012345678"

# 대시보드 생성
curl -X POST "http://localhost:8000/api/v1/dashboards" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "id": "new_dashboard",
    "name": "새 대시보드",
    "layout_type": "grid",
    "grid_columns": 2,
    "grid_rows": 2
  }'
```

### Python 예시

```python
import requests

# API 클라이언트 설정
base_url = "http://localhost:8000/api/v1"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_API_KEY"
}

# 통계 분석 요청
response = requests.post(
    f"{base_url}/analytics/statistics",
    headers=headers,
    json={
        "data_source": "postgres://analytics_db",
        "table_name": "sales",
        "columns": ["amount", "profit"],
        "analysis_type": "descriptive"
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"평균 매출: {data['data']['amount']['mean']}")
else:
    print(f"오류: {response.json()['error']['message']}")

# 작업 계획 요청
response = requests.post(
    f"{base_url}/tasks/plan",
    headers=headers,
    json={
        "intent": "고객 세그먼트 분석",
        "sources": ["postgres://analytics_db"],
        "required_tools": ["sql_executor", "statistics_analyzer"]
    }
)

if response.status_code == 200:
    job_data = response.json()
    job_id = job_data['data']['job_id']
    print(f"작업 ID: {job_id}")
    
    # 작업 상태 폴링
    while True:
        status_response = requests.get(
            f"{base_url}/tasks/{job_id}",
            headers=headers
        )
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            if status_data['data']['ready']:
                print("작업 완료!")
                print(f"결과: {status_data['data']['result']}")
                break
            else:
                print(f"진행률: {status_data['data']['progress']}%")
                time.sleep(5)
```

## 🔄 웹훅 지원

Bridge는 웹훅을 통해 실시간 이벤트 알림을 지원합니다.

### 웹훅 등록

#### POST /api/v1/webhooks

새 웹훅을 등록합니다.

**요청 본문:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["task_completed", "quality_alert"],
  "secret": "your_webhook_secret"
}
```

### 웹훅 페이로드

```json
{
  "event_type": "task_completed",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "job_id": "2f7c18af-1234-5678-9abc-def012345678",
    "status": "success",
    "result": {
      "segments": [...]
    }
  }
}
```
