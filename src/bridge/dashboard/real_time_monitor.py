"""실시간 모니터링

WebSocket 기반 실시간 데이터 스트리밍 및 모니터링 기능을 제공합니다.
"""

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """메트릭 타입"""

    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    CUSTOM = "custom"


class ConnectionStatus(Enum):
    """연결 상태"""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class MetricData:
    """메트릭 데이터"""

    id: str
    metric_type: MetricType
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metric_type": self.metric_type.value,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class AlertRule:
    """알림 규칙"""

    id: str
    name: str
    metric_type: MetricType
    metric_name: str
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    severity: str  # info, warning, error, critical
    enabled: bool = True
    cooldown_seconds: int = 60

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "metric_type": self.metric_type.value,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity,
            "enabled": self.enabled,
            "cooldown_seconds": self.cooldown_seconds,
        }


@dataclass
class Alert:
    """알림"""

    id: str
    rule_id: str
    metric_name: str
    value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "threshold": self.threshold,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
        }


class MetricCollector:
    """메트릭 수집기"""

    def __init__(self):
        self.collectors: Dict[str, Callable[[], List[MetricData]]] = {}
        self.metrics_queue: queue.Queue = queue.Queue(maxsize=10000)
        self.is_collecting = False
        self.collect_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)

    def register_collector(self, name: str, collector_func: Callable[[], List[MetricData]]):
        """수집기 등록"""
        self.collectors[name] = collector_func
        self.logger.info(f"메트릭 수집기 등록: {name}")

    def start_collection(self, interval: int = 5):
        """수집 시작"""
        if self.is_collecting:
            return

        self.stop_event.clear()
        self.collect_thread = threading.Thread(
            target=self._collection_loop, args=(interval,), daemon=True
        )
        self.collect_thread.start()
        self.is_collecting = True
        self.logger.info("메트릭 수집 시작")

    def stop_collection(self):
        """수집 중지"""
        self.stop_event.set()
        if self.collect_thread and self.collect_thread.is_alive():
            self.collect_thread.join(timeout=5)
        self.is_collecting = False
        self.logger.info("메트릭 수집 중지")

    def _collection_loop(self, interval: int):
        """수집 루프"""
        while not self.stop_event.is_set():
            try:
                for name, collector in self.collectors.items():
                    try:
                        metrics = collector()
                        for metric in metrics:
                            if not self.metrics_queue.full():
                                self.metrics_queue.put(metric)
                    except Exception as e:
                        self.logger.error(f"수집기 {name} 실행 실패: {e}")

                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"수집 루프 오류: {e}")
                break

    def get_metrics(self, max_count: int = 100) -> List[MetricData]:
        """메트릭 조회"""
        metrics = []
        count = 0

        while count < max_count and not self.metrics_queue.empty():
            try:
                metric = self.metrics_queue.get_nowait()
                metrics.append(metric)
                count += 1
            except queue.Empty:
                break

        return metrics

    def get_metrics_by_type(
        self, metric_type: MetricType, max_count: int = 100
    ) -> List[MetricData]:
        """타입별 메트릭 조회"""
        all_metrics = self.get_metrics(max_count * 2)  # 더 많이 가져와서 필터링
        return [m for m in all_metrics if m.metric_type == metric_type][:max_count]


class AlertManager:
    """알림 관리자"""

    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alerts: List[Alert] = []
        self.last_alert_time: Dict[str, datetime] = {}
        self.logger = logging.getLogger(__name__)

    def add_rule(self, rule: AlertRule) -> bool:
        """알림 규칙 추가"""
        try:
            self.rules[rule.id] = rule
            self.logger.info(f"알림 규칙 추가: {rule.id}")
            return True
        except Exception as e:
            self.logger.error(f"알림 규칙 추가 실패: {e}")
            return False

    def remove_rule(self, rule_id: str) -> bool:
        """알림 규칙 제거"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"알림 규칙 제거: {rule_id}")
            return True
        return False

    def check_metrics(self, metrics: List[MetricData]) -> List[Alert]:
        """메트릭 확인 및 알림 생성"""
        new_alerts = []

        for metric in metrics:
            for rule in self.rules.values():
                if not rule.enabled:
                    continue

                if rule.metric_type == metric.metric_type and rule.metric_name == metric.name:

                    # 쿨다운 확인
                    rule_key = f"{rule.id}_{metric.name}"
                    if rule_key in self.last_alert_time:
                        time_since_last = datetime.now() - self.last_alert_time[rule_key]
                        if time_since_last.total_seconds() < rule.cooldown_seconds:
                            continue

                    # 조건 확인
                    if self._check_condition(metric.value, rule.condition, rule.threshold):
                        alert = self._create_alert(rule, metric)
                        if alert:
                            new_alerts.append(alert)
                            self.last_alert_time[rule_key] = datetime.now()

        self.alerts.extend(new_alerts)
        return new_alerts

    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """조건 확인"""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        else:
            return False

    def _create_alert(self, rule: AlertRule, metric: MetricData) -> Optional[Alert]:
        """알림 생성"""
        import uuid

        alert = Alert(
            id=str(uuid.uuid4()),
            rule_id=rule.id,
            metric_name=metric.name,
            value=metric.value,
            threshold=rule.threshold,
            severity=rule.severity,
            message=f"{metric.name}이 임계값을 초과했습니다: {metric.value} {metric.unit}",
            timestamp=datetime.now(),
        )

        return alert

    def get_alerts(
        self, severity: Optional[str] = None, acknowledged: Optional[bool] = None, hours: int = 24
    ) -> List[Dict[str, Any]]:
        """알림 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]

        if severity is not None:
            filtered_alerts = [alert for alert in filtered_alerts if alert.severity == severity]

        if acknowledged is not None:
            filtered_alerts = [
                alert for alert in filtered_alerts if alert.acknowledged == acknowledged
            ]

        return [
            alert.to_dict()
            for alert in sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
        ]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """알림 확인 처리"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"알림 확인 처리: {alert_id}")
                return True
        return False


class RealTimeMonitor:
    """실시간 모니터링"""

    def __init__(self):
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.connections: Set[str] = set()  # 연결된 클라이언트 ID
        self.connection_status: Dict[str, ConnectionStatus] = {}
        self.data_queue: queue.Queue = queue.Queue(maxsize=1000)
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)

        # 콜백 함수들
        self.data_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []

    def start_monitoring(self, collection_interval: int = 5):
        """모니터링 시작"""
        if self.is_running:
            return

        # 메트릭 수집 시작
        self.metric_collector.start_collection(collection_interval)

        # 모니터링 루프 시작
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.is_running = True
        self.logger.info("실시간 모니터링 시작")

    def stop_monitoring(self):
        """모니터링 중지"""
        self.stop_event.set()

        # 메트릭 수집 중지
        self.metric_collector.stop_collection()

        # 모니터링 루프 중지
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        self.is_running = False
        self.logger.info("실시간 모니터링 중지")

    def _monitoring_loop(self):
        """모니터링 루프"""
        while not self.stop_event.is_set():
            try:
                # 메트릭 수집
                metrics = self.metric_collector.get_metrics(100)

                if metrics:
                    # 알림 확인
                    new_alerts = self.alert_manager.check_metrics(metrics)

                    # 데이터 패키징
                    data = {
                        "type": "metrics",
                        "timestamp": datetime.now().isoformat(),
                        "metrics": [metric.to_dict() for metric in metrics],
                        "alerts": [alert.to_dict() for alert in new_alerts],
                    }

                    # 데이터 큐에 추가
                    if not self.data_queue.full():
                        self.data_queue.put(data)

                    # 콜백 함수 호출
                    for callback in self.data_callbacks:
                        try:
                            callback(data)
                        except Exception as e:
                            self.logger.error(f"데이터 콜백 실행 실패: {e}")

                    # 알림 콜백 호출
                    for alert in new_alerts:
                        for callback in self.alert_callbacks:
                            try:
                                callback(alert)
                            except Exception as e:
                                self.logger.error(f"알림 콜백 실행 실패: {e}")

                time.sleep(1)  # 1초마다 확인

            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                break

    def add_connection(self, client_id: str):
        """연결 추가"""
        self.connections.add(client_id)
        self.connection_status[client_id] = ConnectionStatus.CONNECTED
        self.logger.info(f"클라이언트 연결: {client_id}")

    def remove_connection(self, client_id: str):
        """연결 제거"""
        self.connections.discard(client_id)
        self.connection_status.pop(client_id, None)
        self.logger.info(f"클라이언트 연결 해제: {client_id}")

    def get_connection_status(self, client_id: str) -> ConnectionStatus:
        """연결 상태 조회"""
        return self.connection_status.get(client_id, ConnectionStatus.DISCONNECTED)

    def get_latest_data(self, client_id: str) -> Optional[Dict[str, Any]]:
        """최신 데이터 조회"""
        if client_id not in self.connections:
            return None

        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def add_alert_rule(self, rule: AlertRule) -> bool:
        """알림 규칙 추가"""
        return self.alert_manager.add_rule(rule)

    def remove_alert_rule(self, rule_id: str) -> bool:
        """알림 규칙 제거"""
        return self.alert_manager.remove_rule(rule_id)

    def get_alerts(self, **kwargs) -> List[Dict[str, Any]]:
        """알림 조회"""
        return self.alert_manager.get_alerts(**kwargs)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """알림 확인 처리"""
        return self.alert_manager.acknowledge_alert(alert_id)

    def add_data_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """데이터 콜백 추가"""
        self.data_callbacks.append(callback)

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """모니터링 상태 조회"""
        return {
            "is_running": self.is_running,
            "active_connections": len(self.connections),
            "connection_status": {
                client_id: status.value for client_id, status in self.connection_status.items()
            },
            "queue_size": self.data_queue.qsize(),
            "collectors_count": len(self.metric_collector.collectors),
            "alert_rules_count": len(self.alert_manager.rules),
            "total_alerts": len(self.alert_manager.alerts),
        }

    def register_system_collector(self):
        """시스템 메트릭 수집기 등록"""

        def collect_system_metrics() -> List[MetricData]:
            import uuid

            import psutil

            metrics = []
            timestamp = datetime.now()

            try:
                # CPU 사용률
                cpu_percent = psutil.cpu_percent(interval=1)
                metrics.append(
                    MetricData(
                        id=str(uuid.uuid4()),
                        metric_type=MetricType.SYSTEM,
                        name="cpu_percent",
                        value=cpu_percent,
                        unit="%",
                        timestamp=timestamp,
                        tags={"host": "localhost"},
                        metadata={"description": "CPU 사용률"},
                    )
                )

                # 메모리 사용률
                memory = psutil.virtual_memory()
                metrics.append(
                    MetricData(
                        id=str(uuid.uuid4()),
                        metric_type=MetricType.SYSTEM,
                        name="memory_percent",
                        value=memory.percent,
                        unit="%",
                        timestamp=timestamp,
                        tags={"host": "localhost"},
                        metadata={"description": "메모리 사용률"},
                    )
                )

                # 디스크 사용률
                disk = psutil.disk_usage("/")
                disk_percent = (disk.used / disk.total) * 100
                metrics.append(
                    MetricData(
                        id=str(uuid.uuid4()),
                        metric_type=MetricType.SYSTEM,
                        name="disk_percent",
                        value=disk_percent,
                        unit="%",
                        timestamp=timestamp,
                        tags={"host": "localhost"},
                        metadata={"description": "디스크 사용률"},
                    )
                )

            except Exception as e:
                self.logger.error(f"시스템 메트릭 수집 실패: {e}")

            return metrics

        self.metric_collector.register_collector("system", collect_system_metrics)

    def register_application_collector(self):
        """애플리케이션 메트릭 수집기 등록"""

        def collect_application_metrics() -> List[MetricData]:
            import random
            import uuid

            metrics = []
            timestamp = datetime.now()

            # 모의 애플리케이션 메트릭
            metrics.append(
                MetricData(
                    id=str(uuid.uuid4()),
                    metric_type=MetricType.APPLICATION,
                    name="response_time",
                    value=random.uniform(100, 500),
                    unit="ms",
                    timestamp=timestamp,
                    tags={"service": "bridge-api"},
                    metadata={"description": "API 응답시간"},
                )
            )

            metrics.append(
                MetricData(
                    id=str(uuid.uuid4()),
                    metric_type=MetricType.APPLICATION,
                    name="throughput",
                    value=random.uniform(50, 200),
                    unit="req/s",
                    timestamp=timestamp,
                    tags={"service": "bridge-api"},
                    metadata={"description": "처리량"},
                )
            )

            metrics.append(
                MetricData(
                    id=str(uuid.uuid4()),
                    metric_type=MetricType.APPLICATION,
                    name="error_rate",
                    value=random.uniform(0, 5),
                    unit="%",
                    timestamp=timestamp,
                    tags={"service": "bridge-api"},
                    metadata={"description": "에러율"},
                )
            )

            return metrics

        self.metric_collector.register_collector("application", collect_application_metrics)
