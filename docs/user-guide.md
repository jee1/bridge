# Bridge 사용자 가이드

## 📖 개요

Bridge는 다양한 데이터 소스에 대한 표준화된 접근을 제공하고, AI 에이전트가 엔터프라이즈 데이터를 안전하고 투명하게 활용할 수 있도록 지원하는 시스템입니다.

## 🚀 시작하기

### 1. 설치 및 설정

```bash
# 의존성 설치
make install

# 개발 서버 실행
make dev

# MCP 서버 실행
make mcp-server
```

### 2. 기본 사용법

```bash
# CLI를 통한 기본 분석
python cli.py "고객 세그먼트 분석"

# 특정 데이터 소스와 도구 지정
python cli.py "프리미엄 고객 분석" --sources postgres://analytics_db --tools sql_executor,statistics_analyzer
```

## 🔌 데이터 커넥터 사용법

### PostgreSQL 커넥터

```python
from src.bridge.connectors.postgres import PostgresConnector

# 커넥터 생성
connector = PostgresConnector(
    host="localhost",
    port=5432,
    database="analytics",
    username="user",
    password="password"
)

# 연결 테스트
await connector.test_connection()

# 메타데이터 조회
metadata = await connector.get_metadata()
print(f"테이블 수: {len(metadata['tables'])}")

# 쿼리 실행
async for row in connector.run_query("SELECT * FROM customers LIMIT 10"):
    print(row)
```

### MongoDB 커넥터

```python
from src.bridge.connectors.mongodb import MongoConnector

# 커넥터 생성
connector = MongoConnector(
    host="localhost",
    port=27017,
    database="analytics",
    username="user",
    password="password"
)

# 컬렉션 목록 조회
metadata = await connector.get_metadata()
print(f"컬렉션 수: {len(metadata['collections'])}")

# 집계 쿼리 실행
pipeline = [
    {"$group": {"_id": "$category", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
]
async for result in connector.run_query(pipeline):
    print(result)
```

### Elasticsearch 커넥터

```python
from src.bridge.connectors.elasticsearch import ElasticsearchConnector

# 커넥터 생성
connector = ElasticsearchConnector(
    host="localhost",
    port=9200,
    username="elastic",
    password="password"
)

# 인덱스 목록 조회
metadata = await connector.get_metadata()
print(f"인덱스 수: {len(metadata['indices'])}")

# 검색 쿼리 실행
query = {
    "query": {
        "match": {
            "title": "analytics"
        }
    }
}
async for result in connector.run_query(query):
    print(result)
```

## 📊 분석 도구 사용법

### 통계 분석

```python
from src.bridge.analytics.core import StatisticsAnalyzer, UnifiedDataFrame
import pandas as pd

# 데이터 준비
data = pd.DataFrame({
    'sales': [100, 200, 150, 300, 250],
    'profit': [20, 40, 30, 60, 50],
    'region': ['North', 'South', 'North', 'East', 'West']
})

unified_df = UnifiedDataFrame(data)

# 통계 분석기 생성
analyzer = StatisticsAnalyzer()

# 기술 통계 계산
stats = analyzer.calculate_descriptive_stats(unified_df, ['sales', 'profit'])
print(f"평균 매출: {stats['sales']['mean']:.2f}")
print(f"매출 표준편차: {stats['sales']['std']:.2f}")

# 상관관계 분석
correlation = analyzer.calculate_correlation(unified_df, ['sales', 'profit'])
print(f"매출-이익 상관계수: {correlation['sales']['profit']:.3f}")

# 분포 분석
distribution = analyzer.calculate_distribution_stats(unified_df, 'sales')
print(f"매출 왜도: {distribution['skewness']:.3f}")
print(f"매출 첨도: {distribution['kurtosis']:.3f}")
```

### 데이터 품질 검사

```python
from src.bridge.analytics.core import QualityChecker

# 품질 검사기 생성
checker = QualityChecker()

# 결측값 분석
missing_stats = checker.analyze_missing_values(unified_df)
print(f"결측값 비율: {missing_stats.overall_missing_ratio:.2%}")

# 이상치 탐지
outlier_stats = checker.detect_outliers(unified_df, 'sales', method='iqr')
print(f"이상치 개수: {outlier_stats.outlier_count}")

# 데이터 일관성 검사
consistency_stats = checker.check_consistency(unified_df)
print(f"일관성 점수: {consistency_stats.consistency_score:.2f}")

# 종합 품질 리포트
quality_report = checker.generate_quality_report(unified_df)
print(f"전체 품질 점수: {quality_report.overall_score:.2f}")
print(f"권장사항: {quality_report.recommendations}")
```

### 시각화

```python
from src.bridge.analytics.core import ChartGenerator, ChartConfig

# 차트 생성기 생성
chart_generator = ChartGenerator()

# 막대 차트 생성
bar_config = ChartConfig(
    chart_type="bar",
    title="지역별 매출",
    x_axis="region",
    y_axis="sales"
)
bar_chart = chart_generator.create_bar_chart(unified_df, bar_config)

# 선 차트 생성
line_config = ChartConfig(
    chart_type="line",
    title="매출 추이",
    x_axis="date",
    y_axis="sales"
)
line_chart = chart_generator.create_line_chart(unified_df, line_config)

# 산점도 생성
scatter_config = ChartConfig(
    chart_type="scatter",
    title="매출 vs 이익",
    x_axis="sales",
    y_axis="profit"
)
scatter_chart = chart_generator.create_scatter_plot(unified_df, scatter_config)

# 히스토그램 생성
hist_config = ChartConfig(
    chart_type="histogram",
    title="매출 분포",
    column="sales"
)
histogram = chart_generator.create_histogram(unified_df, hist_config)
```

### 대시보드 생성

```python
from src.bridge.analytics.core import DashboardGenerator, DashboardConfig

# 대시보드 생성기 생성
dashboard_generator = DashboardGenerator()

# 대시보드 설정
dashboard_config = DashboardConfig(
    title="매출 분석 대시보드",
    layout=(2, 2),
    figsize=(12, 8)
)

# 대시보드 생성
dashboard = dashboard_generator.create_analytics_dashboard(unified_df, dashboard_config)
```

### 리포트 생성

```python
from src.bridge.analytics.core import ReportGenerator, ReportConfig

# 리포트 생성기 생성
report_generator = ReportGenerator()

# 리포트 설정
report_config = ReportConfig(
    title="월간 매출 분석 리포트",
    author="분석팀",
    sections=['overview', 'statistics', 'quality', 'charts', 'dashboard']
)

# 리포트 생성
report_data = report_generator.generate_analytics_report(unified_df, report_config)
print(f"리포트 생성 완료: {report_data['title']}")
```

## 🔒 데이터 거버넌스 사용법

### 데이터 계약 관리

```python
from src.bridge.governance.contracts import DataContractManager, DataContract, DataField, QualityRule

# 계약 관리자 생성
contract_manager = DataContractManager()

# 데이터 계약 생성
contract = DataContract(
    id="customer_contract",
    name="고객 데이터 계약",
    version="1.0",
    description="고객 데이터 품질 계약",
    data_source="postgres://analytics_db",
    table_name="customers",
    fields=[
        DataField(
            name="customer_id",
            data_type="integer",
            required=True,
            description="고객 ID"
        ),
        DataField(
            name="email",
            data_type="string",
            required=True,
            description="이메일 주소"
        )
    ],
    quality_rules=[
        QualityRule(
            field_name="email",
            rule_type="format",
            rule_value="email",
            description="이메일 형식 검증"
        )
    ]
)

# 계약 등록
contract_manager.create_contract(contract)

# 계약 조회
retrieved_contract = contract_manager.get_contract("customer_contract")
print(f"계약명: {retrieved_contract.name}")

# 데이터 검증
validation_result = contract.validate_data(customer_data)
if validation_result.is_valid:
    print("데이터가 계약을 준수합니다")
else:
    print(f"검증 실패: {validation_result.errors}")
```

### 메타데이터 카탈로그

```python
from src.bridge.governance.metadata import MetadataCatalog, DataAsset, ColumnMetadata

# 메타데이터 카탈로그 생성
catalog = MetadataCatalog()

# 데이터 자산 등록
asset = DataAsset(
    id="customers_table",
    name="고객 테이블",
    asset_type="table",
    data_source="postgres://analytics_db",
    schema="public",
    table_name="customers",
    columns=[
        ColumnMetadata(
            name="customer_id",
            data_type="integer",
            is_primary_key=True,
            description="고객 고유 식별자"
        ),
        ColumnMetadata(
            name="email",
            data_type="varchar",
            is_nullable=False,
            description="고객 이메일 주소"
        )
    ],
    tags=["customer", "pii", "critical"]
)

# 자산 등록
catalog.add_asset(asset)

# 자산 조회
retrieved_asset = catalog.get_asset("customers_table")
print(f"자산명: {retrieved_asset.name}")
print(f"컬럼 수: {len(retrieved_asset.columns)}")

# 태그로 검색
customer_assets = catalog.search_assets_by_tag("customer")
print(f"고객 관련 자산 수: {len(customer_assets)}")
```

### RBAC (역할 기반 접근 제어)

```python
from src.bridge.governance.rbac import RBACManager, Role, Permission, User

# RBAC 관리자 생성
rbac_manager = RBACManager()

# 역할 생성
analyst_role = Role(
    id="analyst",
    name="데이터 분석가",
    description="데이터 분석 및 조회 권한",
    permissions=[
        Permission(
            resource_type="table",
            resource_id="customers",
            actions=["read", "query"]
        ),
        Permission(
            resource_type="table",
            resource_id="sales",
            actions=["read", "query", "analyze"]
        )
    ]
)

# 사용자 생성
user = User(
    id="user123",
    username="john_doe",
    email="john@example.com",
    roles=["analyst"]
)

# 역할 및 사용자 등록
rbac_manager.create_role(analyst_role)
rbac_manager.create_user(user)

# 접근 권한 확인
has_access = rbac_manager.check_permission(
    user_id="user123",
    resource_type="table",
    resource_id="customers",
    action="read"
)
print(f"접근 권한: {has_access}")

# 사용자 역할 조회
user_roles = rbac_manager.get_user_roles("user123")
print(f"사용자 역할: {[role.name for role in user_roles]}")
```

### 감사 로그

```python
from src.bridge.governance.audit import AuditLogManager, AuditEvent

# 감사 로그 관리자 생성
audit_manager = AuditLogManager()

# 감사 이벤트 생성
event = AuditEvent(
    event_type="data_access",
    user_id="user123",
    resource_type="table",
    resource_id="customers",
    action="query",
    details={
        "query": "SELECT * FROM customers WHERE region = 'North'",
        "rows_returned": 150
    }
)

# 이벤트 로깅
audit_manager.log_event(event)

# 감사 로그 조회
recent_events = audit_manager.get_events(
    user_id="user123",
    hours=24
)
print(f"최근 24시간 이벤트 수: {len(recent_events)}")

# 컴플라이언스 리포트 생성
report = audit_manager.generate_compliance_report(
    start_date="2024-01-01",
    end_date="2024-01-31"
)
print(f"1월 감사 리포트: {report['summary']}")
```

## 🤖 자동화 파이프라인 사용법

### 데이터 품질 모니터링

```python
from src.bridge.automation.quality_monitor import QualityMonitor, QualityThreshold, AlertSeverity

# 품질 모니터 생성
monitor = QualityMonitor(check_interval=300)  # 5분 간격

# 임계값 설정
threshold = QualityThreshold(
    id="cpu_threshold",
    name="CPU 사용률 임계값",
    metric_type="cpu_percent",
    threshold_value=80.0,
    operator="gt",
    severity=AlertSeverity.WARNING
)

# 모니터링 작업 추가
monitor.add_threshold(threshold)
monitor.add_monitoring_task(
    task_id="system_monitoring",
    data_source="postgres://analytics_db",
    table_name="system_metrics"
)

# 모니터링 시작
monitor.start_monitoring()

# 알림 콜백 설정
def alert_callback(alert):
    print(f"알림 발생: {alert.message}")

monitor.add_alert_callback(alert_callback)

# 모니터링 상태 조회
status = monitor.get_monitoring_status()
print(f"모니터링 상태: {status['status']}")
print(f"임계값 수: {status['thresholds_count']}")
```

### 자동 리포트 생성

```python
from src.bridge.automation.report_automation import ReportAutomation, ReportTemplate, ReportSchedule, ScheduleType

# 리포트 자동화 생성
automation = ReportAutomation()

# 리포트 템플릿 생성
template = ReportTemplate(
    id="daily_sales_report",
    name="일일 매출 리포트",
    description="매일 자동 생성되는 매출 분석 리포트",
    data_source="postgres://analytics_db",
    query="SELECT * FROM daily_sales WHERE date = CURRENT_DATE",
    output_format="html"
)

# 스케줄 생성
schedule = ReportSchedule(
    id="daily_schedule",
    template_id="daily_sales_report",
    schedule_type=ScheduleType.DAILY,
    start_time=datetime.now().replace(hour=9, minute=0, second=0)
)

# 템플릿 및 스케줄 등록
automation.create_template(template)
automation.create_schedule(schedule)

# 스케줄러 시작
automation.start_scheduler()

# 수동 리포트 실행
job_id = automation.execute_template("daily_sales_report")
print(f"리포트 작업 ID: {job_id}")

# 작업 상태 조회
job = automation.get_job_status(job_id)
print(f"작업 상태: {job.status}")
```

### 알림 시스템

```python
from src.bridge.automation.notification_system import NotificationSystem, AlertRule, AlertPriority

# 알림 시스템 생성
notification = NotificationSystem()

# 이메일 채널 설정
email_config = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "username": "your_email@gmail.com",
    "password": "your_password",
    "from_email": "your_email@gmail.com",
    "to_emails": ["admin@example.com"]
}

notification.add_channel("email", email_config)

# 알림 규칙 생성
rule = AlertRule(
    id="high_cpu_alert",
    name="높은 CPU 사용률 알림",
    event_type="quality_alert",
    conditions={"severity": "warning"},
    channels=["email"],
    priority=AlertPriority.HIGH
)

# 규칙 등록
notification.add_rule(rule)

# 알림 발송
data = {
    "severity": "warning",
    "message": "CPU 사용률이 80%를 초과했습니다",
    "value": 85.5,
    "threshold": 80.0
}

sent_messages = notification.send_notification("quality_alert", data)
print(f"발송된 알림 수: {len(sent_messages)}")
```

### 작업 스케줄러

```python
from src.bridge.automation.scheduler import TaskScheduler, ScheduledTask, TaskType

# 스케줄러 생성
scheduler = TaskScheduler()

# 작업 함수 정의
def data_cleanup_job():
    print("데이터 정리 작업 실행")
    # 실제 정리 로직
    return {"status": "completed", "cleaned_rows": 1000}

# 스케줄된 작업 생성
task = ScheduledTask(
    id="daily_cleanup",
    name="일일 데이터 정리",
    description="매일 실행되는 데이터 정리 작업",
    task_type=TaskType.CLEANUP,
    function=data_cleanup_job,
    cron_expression="0 2 * * *",  # 매일 오전 2시
    enabled=True
)

# 작업 등록
scheduler.add_task(task)

# 스케줄러 시작
scheduler.start_scheduler()

# 작업 상태 조회
status = scheduler.get_task_status("daily_cleanup")
print(f"작업 상태: {status['enabled']}")
print(f"다음 실행: {status['next_run']}")

# 즉시 실행
execution_id = scheduler.execute_task_now("daily_cleanup")
print(f"실행 ID: {execution_id}")
```

## 📊 대시보드 사용법

### 대시보드 관리

```python
from src.bridge.dashboard.dashboard_manager import DashboardManager, DashboardConfig, DashboardWidget, WidgetType

# 대시보드 관리자 생성
dashboard_manager = DashboardManager()

# 대시보드 생성
config = DashboardConfig(
    id="sales_dashboard",
    name="매출 분석 대시보드",
    description="매출 관련 지표 및 차트를 보여주는 대시보드",
    layout_type="grid",
    grid_columns=4,
    grid_rows=3
)

dashboard_manager.create_dashboard(config)

# 위젯 추가
widget = DashboardWidget(
    id="sales_chart",
    widget_type=WidgetType.CHART,
    title="월별 매출 추이",
    position={"x": 0, "y": 0, "width": 2, "height": 1},
    config={
        "chart_type": "line",
        "data_source": "monthly_sales"
    }
)

dashboard_manager.add_widget(widget, "sales_dashboard")

# 대시보드 데이터 조회
dashboard_data = dashboard_manager.get_dashboard_data("sales_dashboard")
print(f"대시보드명: {dashboard_data['name']}")
print(f"위젯 수: {len(dashboard_data['widgets'])}")
```

### 실시간 모니터링

```python
from src.bridge.dashboard.real_time_monitor import RealTimeMonitor

# 실시간 모니터 생성
monitor = RealTimeMonitor()

# 시스템 메트릭 수집기 등록
monitor.register_system_collector()

# 애플리케이션 메트릭 수집기 등록
monitor.register_application_collector()

# 모니터링 시작
monitor.start_monitoring(collection_interval=5)

# 클라이언트 연결
client_id = "dashboard_client"
monitor.add_connection(client_id)

# 최신 데이터 조회
data = monitor.get_latest_data(client_id)
if data:
    print(f"메트릭 수: {len(data['metrics'])}")
    print(f"알림 수: {len(data['alerts'])}")

# 모니터링 상태 조회
status = monitor.get_monitoring_status()
print(f"활성 연결 수: {status['active_connections']}")
print(f"수집기 수: {status['collectors_count']}")
```

### 시각화 엔진

```python
from src.bridge.dashboard.visualization_engine import VisualizationEngine, ChartConfig, ChartType

# 시각화 엔진 생성
engine = VisualizationEngine()

# 샘플 데이터 생성
cpu_data = engine.create_sample_data("cpu", 100)
memory_data = engine.create_sample_data("memory", 100)

# 대시보드 설정
dashboard_config = {
    "id": "system_dashboard",
    "name": "시스템 모니터링 대시보드",
    "layout_type": "grid",
    "grid_columns": 2,
    "grid_rows": 2,
    "widgets": [
        {
            "id": "cpu_chart",
            "type": "chart",
            "title": "CPU 사용률",
            "config": {
                "chart_type": "line",
                "width": 400,
                "height": 300
            }
        },
        {
            "id": "memory_metric",
            "type": "metric",
            "title": "메모리 사용률",
            "config": {
                "width": 200,
                "height": 100
            }
        }
    ]
}

# 위젯 데이터
widgets_data = {
    "cpu_chart": cpu_data,
    "memory_metric": [{"value": 65, "unit": "%"}]
}

# 대시보드 렌더링
rendered_dashboard = engine.render_dashboard(dashboard_config, widgets_data)
print(f"렌더링 상태: {rendered_dashboard['status']}")
print(f"위젯 수: {len(rendered_dashboard['layout']['widgets'])}")
```

## 🔧 고급 사용법

### 커스텀 커넥터 개발

```python
from src.bridge.connectors.base import BaseConnector
from typing import Dict, Any, AsyncGenerator

class CustomConnector(BaseConnector):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    async def test_connection(self) -> bool:
        # 연결 테스트 로직
        try:
            # 실제 연결 테스트
            return True
        except Exception:
            return False
    
    async def get_metadata(self) -> Dict[str, Any]:
        # 메타데이터 수집 로직
        return {
            "tables": ["table1", "table2"],
            "version": "1.0"
        }
    
    async def run_query(self, query: str, params: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        # 쿼리 실행 로직
        # 실제 데이터베이스 쿼리 실행
        yield {"result": "data"}
```

### 커스텀 분석 도구 개발

```python
from src.bridge.analytics.core import UnifiedDataFrame
from typing import Dict, Any

class CustomAnalyzer:
    def __init__(self):
        self.name = "Custom Analyzer"
    
    def analyze(self, data: UnifiedDataFrame) -> Dict[str, Any]:
        # 커스텀 분석 로직
        pandas_df = data.to_pandas()
        
        # 분석 수행
        result = {
            "total_rows": len(pandas_df),
            "custom_metric": self._calculate_custom_metric(pandas_df)
        }
        
        return result
    
    def _calculate_custom_metric(self, df) -> float:
        # 커스텀 메트릭 계산
        return df.mean().mean()
```

## 🚨 문제 해결

### 일반적인 문제

1. **연결 오류**
   ```python
   # 연결 테스트
   connector = PostgresConnector(...)
   is_connected = await connector.test_connection()
   if not is_connected:
       print("데이터베이스 연결을 확인하세요")
   ```

2. **메모리 부족**
   ```python
   # 배치 처리로 메모리 사용량 제한
   async for batch in connector.run_query(query, batch_size=1000):
       # 배치별 처리
       process_batch(batch)
   ```

3. **권한 오류**
   ```python
   # RBAC 권한 확인
   has_permission = rbac_manager.check_permission(
       user_id="user123",
       resource_type="table",
       resource_id="customers",
       action="read"
   )
   ```

### 로그 확인

```bash
# 애플리케이션 로그
tail -f logs/bridge.log

# 감사 로그
tail -f logs/audit/audit.log

# 에러 로그
tail -f logs/bridge_error.log
```

### 성능 모니터링

```python
# 성능 메트릭 수집
from src.bridge.dashboard.monitoring_dashboard import MonitoringDashboard

monitor = MonitoringDashboard()
monitor.start_monitoring()

# 현재 메트릭 조회
metrics = monitor.get_current_metrics()
print(f"CPU 사용률: {metrics['system']['cpu_percent']:.1f}%")
print(f"메모리 사용률: {metrics['system']['memory_percent']:.1f}%")
```

## 📚 추가 자료

- [API 참조 문서](api-reference.md)
- [개발자 가이드](developer-guide.md)
- [아키텍처 문서](architecture.md)
- [보안 가이드](security-guide.md)
- [성능 튜닝](performance-tuning.md)
