"""모니터링 대시보드

시스템 상태, 성능 메트릭, 실시간 모니터링 기능을 제공합니다.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import psutil

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """메트릭 타입"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    DATABASE = "database"
    APPLICATION = "application"
    CUSTOM = "custom"


class AlertLevel(Enum):
    """알림 레벨"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """시스템 메트릭"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int  # MB
    disk_usage_percent: float
    disk_free: int  # MB
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    load_average: List[float]  # 1분, 5분, 15분 평균

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available": self.memory_available,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_free": self.disk_free,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "process_count": self.process_count,
            "load_average": self.load_average
        }


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    timestamp: datetime
    response_time: float  # ms
    throughput: float  # requests/second
    error_rate: float  # percentage
    active_connections: int
    queue_length: int
    cache_hit_rate: float  # percentage
    database_connections: int
    custom_metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "response_time": self.response_time,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "active_connections": self.active_connections,
            "queue_length": self.queue_length,
            "cache_hit_rate": self.cache_hit_rate,
            "database_connections": self.database_connections,
            "custom_metrics": self.custom_metrics
        }


@dataclass
class Alert:
    """알림"""
    id: str
    metric_type: MetricType
    alert_level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "metric_type": self.metric_type.value,
            "alert_level": self.alert_level.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }


class MonitoringDashboard:
    """모니터링 대시보드"""
    
    def __init__(self, refresh_interval: int = 5):
        self.refresh_interval = refresh_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # 메트릭 저장소
        self.system_metrics: List[SystemMetrics] = []
        self.performance_metrics: List[PerformanceMetrics] = []
        self.alerts: List[Alert] = []
        
        # 임계값 설정
        self.thresholds = {
            MetricType.CPU: {"warning": 70.0, "critical": 90.0},
            MetricType.MEMORY: {"warning": 80.0, "critical": 95.0},
            MetricType.DISK: {"warning": 85.0, "critical": 95.0},
            MetricType.APPLICATION: {"warning": 5.0, "critical": 10.0}  # 응답시간(초)
        }
        
        # 콜백 함수들
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.metric_callbacks: List[Callable[[SystemMetrics], None]] = []
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self) -> bool:
        """모니터링 시작"""
        if self.is_monitoring:
            return True
        
        try:
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.is_monitoring = True
            self.logger.info("모니터링 시작")
            return True
        except Exception as e:
            self.logger.error(f"모니터링 시작 실패: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """모니터링 중지"""
        try:
            self.stop_event.set()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            self.is_monitoring = False
            self.logger.info("모니터링 중지")
            return True
        except Exception as e:
            self.logger.error(f"모니터링 중지 실패: {e}")
            return False
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while not self.stop_event.is_set():
            try:
                # 시스템 메트릭 수집
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # 성능 메트릭 수집
                performance_metrics = self._collect_performance_metrics()
                self.performance_metrics.append(performance_metrics)
                
                # 임계값 확인 및 알림 생성
                self._check_thresholds(system_metrics, performance_metrics)
                
                # 오래된 메트릭 정리 (1시간 이상)
                self._cleanup_old_metrics()
                
                # 콜백 함수 호출
                for callback in self.metric_callbacks:
                    try:
                        callback(system_metrics)
                    except Exception as e:
                        self.logger.error(f"메트릭 콜백 실행 실패: {e}")
                
                time.sleep(self.refresh_interval)
                
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                break
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """시스템 메트릭 수집"""
        try:
            # CPU 사용률
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 메모리 사용률
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available // (1024 * 1024)  # MB
            
            # 디스크 사용률
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free = disk.free // (1024 * 1024)  # MB
            
            # 네트워크 통계
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # 프로세스 수
            process_count = len(psutil.pids())
            
            # 로드 평균 (Unix 계열에서만 사용 가능)
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                load_average = [0.0, 0.0, 0.0]  # Windows에서는 지원하지 않음
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                disk_usage_percent=disk_usage_percent,
                disk_free=disk_free,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                process_count=process_count,
                load_average=load_average
            )
            
        except Exception as e:
            self.logger.error(f"시스템 메트릭 수집 실패: {e}")
            # 기본값 반환
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available=0,
                disk_usage_percent=0.0,
                disk_free=0,
                network_bytes_sent=0,
                network_bytes_recv=0,
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """성능 메트릭 수집"""
        try:
            # 실제 구현에서는 애플리케이션별 메트릭을 수집
            # 여기서는 모의 데이터 생성
            import random
            
            return PerformanceMetrics(
                timestamp=datetime.now(),
                response_time=random.uniform(100, 500),  # 100-500ms
                throughput=random.uniform(50, 200),  # 50-200 req/s
                error_rate=random.uniform(0, 5),  # 0-5%
                active_connections=random.randint(10, 100),
                queue_length=random.randint(0, 50),
                cache_hit_rate=random.uniform(80, 99),  # 80-99%
                database_connections=random.randint(5, 20),
                custom_metrics={
                    "api_calls_per_minute": random.randint(100, 1000),
                    "cache_size_mb": random.randint(100, 1000),
                    "active_users": random.randint(50, 500)
                }
            )
            
        except Exception as e:
            self.logger.error(f"성능 메트릭 수집 실패: {e}")
            return PerformanceMetrics(
                timestamp=datetime.now(),
                response_time=0.0,
                throughput=0.0,
                error_rate=0.0,
                active_connections=0,
                queue_length=0,
                cache_hit_rate=0.0,
                database_connections=0,
                custom_metrics={}
            )
    
    def _check_thresholds(self, system_metrics: SystemMetrics, performance_metrics: PerformanceMetrics):
        """임계값 확인 및 알림 생성"""
        # CPU 임계값 확인
        self._check_metric_threshold(
            MetricType.CPU, system_metrics.cpu_percent, "CPU 사용률"
        )
        
        # 메모리 임계값 확인
        self._check_metric_threshold(
            MetricType.MEMORY, system_metrics.memory_percent, "메모리 사용률"
        )
        
        # 디스크 임계값 확인
        self._check_metric_threshold(
            MetricType.DISK, system_metrics.disk_usage_percent, "디스크 사용률"
        )
        
        # 응답시간 임계값 확인
        response_time_seconds = performance_metrics.response_time / 1000
        self._check_metric_threshold(
            MetricType.APPLICATION, response_time_seconds, "응답시간"
        )
    
    def _check_metric_threshold(self, metric_type: MetricType, value: float, metric_name: str):
        """개별 메트릭 임계값 확인"""
        if metric_type not in self.thresholds:
            return
        
        thresholds = self.thresholds[metric_type]
        
        # Critical 임계값 확인
        if value >= thresholds.get("critical", float('inf')):
            self._create_alert(
                metric_type, AlertLevel.CRITICAL, 
                f"{metric_name}이 임계값을 초과했습니다: {value:.2f}%",
                value, thresholds["critical"]
            )
        
        # Warning 임계값 확인
        elif value >= thresholds.get("warning", float('inf')):
            self._create_alert(
                metric_type, AlertLevel.WARNING,
                f"{metric_name}이 경고 임계값을 초과했습니다: {value:.2f}%",
                value, thresholds["warning"]
            )
    
    def _create_alert(self, metric_type: MetricType, alert_level: AlertLevel, 
                     message: str, value: float, threshold: float):
        """알림 생성"""
        import uuid
        
        # 중복 알림 방지 (같은 메트릭 타입과 레벨의 최근 알림 확인)
        recent_alerts = [
            alert for alert in self.alerts
            if (alert.metric_type == metric_type and 
                alert.alert_level == alert_level and
                (datetime.now() - alert.timestamp).seconds < 300)  # 5분 이내
        ]
        
        if recent_alerts:
            return  # 중복 알림 방지
        
        alert = Alert(
            id=str(uuid.uuid4()),
            metric_type=metric_type,
            alert_level=alert_level,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=datetime.now()
        )
        
        self.alerts.append(alert)
        
        # 콜백 함수 호출
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"알림 콜백 실행 실패: {e}")
    
    def _cleanup_old_metrics(self):
        """오래된 메트릭 정리"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        # 시스템 메트릭 정리
        self.system_metrics = [
            metric for metric in self.system_metrics
            if metric.timestamp >= cutoff_time
        ]
        
        # 성능 메트릭 정리
        self.performance_metrics = [
            metric for metric in self.performance_metrics
            if metric.timestamp >= cutoff_time
        ]
        
        # 알림 정리 (24시간 이상)
        alert_cutoff = datetime.now() - timedelta(hours=24)
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= alert_cutoff
        ]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 조회"""
        if not self.system_metrics:
            return {}
        
        latest_system = self.system_metrics[-1]
        latest_performance = self.performance_metrics[-1] if self.performance_metrics else None
        
        return {
            "system": latest_system.to_dict(),
            "performance": latest_performance.to_dict() if latest_performance else None,
            "is_monitoring": self.is_monitoring,
            "total_alerts": len(self.alerts),
            "unacknowledged_alerts": len([a for a in self.alerts if not a.acknowledged])
        }
    
    def get_metrics_history(self, hours: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """메트릭 히스토리 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        system_history = [
            metric.to_dict() for metric in self.system_metrics
            if metric.timestamp >= cutoff_time
        ]
        
        performance_history = [
            metric.to_dict() for metric in self.performance_metrics
            if metric.timestamp >= cutoff_time
        ]
        
        return {
            "system": system_history,
            "performance": performance_history
        }
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                   acknowledged: Optional[bool] = None,
                   hours: int = 24) -> List[Dict[str, Any]]:
        """알림 조회"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        if level is not None:
            filtered_alerts = [alert for alert in filtered_alerts if alert.alert_level == level]
        
        if acknowledged is not None:
            filtered_alerts = [alert for alert in filtered_alerts if alert.acknowledged == acknowledged]
        
        return [alert.to_dict() for alert in sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)]
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """알림 확인 처리"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = user_id
                alert.acknowledged_at = datetime.now()
                self.logger.info(f"알림 확인 처리 완료: {alert_id}")
                return True
        return False
    
    def set_threshold(self, metric_type: MetricType, warning: float, critical: float) -> bool:
        """임계값 설정"""
        try:
            self.thresholds[metric_type] = {
                "warning": warning,
                "critical": critical
            }
            self.logger.info(f"임계값 설정 완료: {metric_type.value}")
            return True
        except Exception as e:
            self.logger.error(f"임계값 설정 실패: {e}")
            return False
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)
    
    def add_metric_callback(self, callback: Callable[[SystemMetrics], None]) -> None:
        """메트릭 콜백 추가"""
        self.metric_callbacks.append(callback)
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """대시보드 요약 정보"""
        current_metrics = self.get_current_metrics()
        
        return {
            "status": "monitoring" if self.is_monitoring else "stopped",
            "refresh_interval": self.refresh_interval,
            "metrics_count": {
                "system": len(self.system_metrics),
                "performance": len(self.performance_metrics),
                "alerts": len(self.alerts)
            },
            "current_status": {
                "cpu_percent": current_metrics.get("system", {}).get("cpu_percent", 0),
                "memory_percent": current_metrics.get("system", {}).get("memory_percent", 0),
                "disk_percent": current_metrics.get("system", {}).get("disk_usage_percent", 0),
                "response_time": current_metrics.get("performance", {}).get("response_time", 0) if current_metrics.get("performance") else 0
            },
            "alerts_summary": {
                "total": len(self.alerts),
                "unacknowledged": len([a for a in self.alerts if not a.acknowledged]),
                "by_level": {
                    level.value: len([a for a in self.alerts if a.alert_level == level])
                    for level in AlertLevel
                }
            }
        }
