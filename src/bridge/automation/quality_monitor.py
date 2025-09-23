"""데이터 품질 모니터링 시스템

정기적인 데이터 품질 검사를 수행하고 임계값을 모니터링합니다.
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from bridge.analytics.core import QualityChecker, StatisticsAnalyzer, UnifiedDataFrame
from bridge.governance.contracts import ContractValidator, DataContract

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """알림 심각도"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitorStatus(Enum):
    """모니터 상태"""

    RUNNING = "running"
    STOPPED = "stopped"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class QualityThreshold:
    """품질 임계값 정의"""

    id: str
    name: str
    metric_type: str  # missing_ratio, outlier_ratio, consistency_score, etc.
    threshold_value: float
    operator: str  # gt, lt, eq, gte, lte
    severity: AlertSeverity = AlertSeverity.WARNING
    description: Optional[str] = None
    enabled: bool = True
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def check_threshold(self, value: float) -> bool:
        """임계값 확인"""
        if not self.enabled:
            return False

        if self.operator == "gt":
            return value > self.threshold_value
        elif self.operator == "lt":
            return value < self.threshold_value
        elif self.operator == "eq":
            return value == self.threshold_value
        elif self.operator == "gte":
            return value >= self.threshold_value
        elif self.operator == "lte":
            return value <= self.threshold_value
        else:
            return False


@dataclass
class QualityAlert:
    """품질 알림"""

    id: str
    threshold_id: str
    metric_type: str
    current_value: float
    threshold_value: float
    severity: AlertSeverity
    message: str
    data_source: Optional[str] = None
    table_name: Optional[str] = None
    column_name: Optional[str] = None
    timestamp: datetime = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def acknowledge(self, user_id: str) -> None:
        """알림 확인 처리"""
        self.acknowledged = True
        self.acknowledged_by = user_id
        self.acknowledged_at = datetime.now()


class QualityMonitor:
    """데이터 품질 모니터"""

    def __init__(self, check_interval: int = 300):  # 5분 기본
        self.check_interval = check_interval
        self.status = MonitorStatus.STOPPED
        self.thresholds: Dict[str, QualityThreshold] = {}
        self.alerts: List[QualityAlert] = []
        self.monitoring_tasks: Dict[str, Dict[str, Any]] = {}
        self.quality_checker = QualityChecker()
        self.statistics_analyzer = StatisticsAnalyzer()
        self.contract_validator = ContractValidator()
        self.alert_callbacks: List[Callable[[QualityAlert], None]] = []
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)

    def add_threshold(self, threshold: QualityThreshold) -> bool:
        """임계값 추가"""
        try:
            self.thresholds[threshold.id] = threshold
            self.logger.info(f"품질 임계값 추가 완료: {threshold.id}")
            return True
        except Exception as e:
            self.logger.error(f"품질 임계값 추가 실패: {e}")
            return False

    def remove_threshold(self, threshold_id: str) -> bool:
        """임계값 제거"""
        if threshold_id in self.thresholds:
            del self.thresholds[threshold_id]
            self.logger.info(f"품질 임계값 제거 완료: {threshold_id}")
            return True
        return False

    def add_monitoring_task(
        self,
        task_id: str,
        data_source: str,
        table_name: str,
        contract_id: Optional[str] = None,
        check_columns: Optional[List[str]] = None,
    ) -> bool:
        """모니터링 작업 추가"""
        try:
            self.monitoring_tasks[task_id] = {
                "data_source": data_source,
                "table_name": table_name,
                "contract_id": contract_id,
                "check_columns": check_columns or [],
                "last_check": None,
                "enabled": True,
            }
            self.logger.info(f"모니터링 작업 추가 완료: {task_id}")
            return True
        except Exception as e:
            self.logger.error(f"모니터링 작업 추가 실패: {e}")
            return False

    def remove_monitoring_task(self, task_id: str) -> bool:
        """모니터링 작업 제거"""
        if task_id in self.monitoring_tasks:
            del self.monitoring_tasks[task_id]
            self.logger.info(f"모니터링 작업 제거 완료: {task_id}")
            return True
        return False

    def add_alert_callback(self, callback: Callable[[QualityAlert], None]) -> None:
        """알림 콜백 추가"""
        self.alert_callbacks.append(callback)

    def start_monitoring(self) -> bool:
        """모니터링 시작"""
        if self.status == MonitorStatus.RUNNING:
            return True

        try:
            self.stop_event.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.status = MonitorStatus.RUNNING
            self.logger.info("품질 모니터링 시작")
            return True
        except Exception as e:
            self.logger.error(f"모니터링 시작 실패: {e}")
            self.status = MonitorStatus.ERROR
            return False

    def stop_monitoring(self) -> bool:
        """모니터링 중지"""
        try:
            self.stop_event.set()
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5)
            self.status = MonitorStatus.STOPPED
            self.logger.info("품질 모니터링 중지")
            return True
        except Exception as e:
            self.logger.error(f"모니터링 중지 실패: {e}")
            return False

    def pause_monitoring(self) -> bool:
        """모니터링 일시 중지"""
        self.status = MonitorStatus.PAUSED
        self.logger.info("품질 모니터링 일시 중지")
        return True

    def resume_monitoring(self) -> bool:
        """모니터링 재개"""
        if self.status == MonitorStatus.PAUSED:
            self.status = MonitorStatus.RUNNING
            self.logger.info("품질 모니터링 재개")
            return True
        return False

    def _monitor_loop(self):
        """모니터링 루프"""
        while not self.stop_event.is_set():
            try:
                if self.status == MonitorStatus.RUNNING:
                    self._check_all_tasks()

                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"모니터링 루프 오류: {e}")
                self.status = MonitorStatus.ERROR
                break

    def _check_all_tasks(self):
        """모든 모니터링 작업 확인"""
        for task_id, task_config in self.monitoring_tasks.items():
            if not task_config.get("enabled", True):
                continue

            try:
                self._check_single_task(task_id, task_config)
            except Exception as e:
                self.logger.error(f"모니터링 작업 확인 실패 {task_id}: {e}")

    def _check_single_task(self, task_id: str, task_config: Dict[str, Any]):
        """단일 모니터링 작업 확인"""
        # 실제 구현에서는 데이터 소스에서 데이터를 가져와야 함
        # 여기서는 모의 데이터로 품질 검사 수행
        mock_data = self._generate_mock_data(task_config)

        if mock_data is None:
            return

        # 품질 검사 수행
        quality_report = self.quality_checker.generate_quality_report(mock_data)

        # 임계값 확인
        self._check_thresholds(task_id, task_config, quality_report)

        # 작업 상태 업데이트
        task_config["last_check"] = datetime.now()

    def _generate_mock_data(self, task_config: Dict[str, Any]) -> Optional[UnifiedDataFrame]:
        """모의 데이터 생성 (실제 구현에서는 데이터 소스에서 가져옴)"""
        import numpy as np
        import pandas as pd

        # 간단한 모의 데이터 생성
        np.random.seed(42)
        data = {
            "id": list(range(100)),
            "value1": np.random.normal(100, 15, 100).tolist(),
            "value2": np.random.normal(50, 10, 100).tolist(),
            "category": (["A", "B", "C"] * 33 + ["A"])[:100],
            "missing_col": ([1, 2, 3, None, 5] * 20)[:100],
        }

        try:
            return UnifiedDataFrame(data)
        except Exception as e:
            self.logger.error(f"모의 데이터 생성 실패: {e}")
            return None

    def _check_thresholds(self, task_id: str, task_config: Dict[str, Any], quality_report: Any):
        """임계값 확인"""
        for threshold in self.thresholds.values():
            if not threshold.enabled:
                continue

            current_value = self._get_metric_value(quality_report, threshold.metric_type)
            if current_value is None:
                continue

            if threshold.check_threshold(current_value):
                alert = self._create_alert(threshold, current_value, task_config)
                self._handle_alert(alert)

    def _get_metric_value(self, quality_report: Any, metric_type: str) -> Optional[float]:
        """품질 리포트에서 메트릭 값 추출"""
        try:
            if metric_type == "missing_ratio":
                return quality_report.missing_value_score / 100.0
            elif metric_type == "outlier_ratio":
                return (100 - quality_report.outlier_score) / 100.0
            elif metric_type == "consistency_score":
                return quality_report.consistency_score / 100.0
            elif metric_type == "overall_score":
                return quality_report.overall_score / 100.0
            else:
                return None
        except Exception as e:
            self.logger.error(f"메트릭 값 추출 실패: {e}")
            return None

    def _create_alert(
        self, threshold: QualityThreshold, current_value: float, task_config: Dict[str, Any]
    ) -> QualityAlert:
        """알림 생성"""
        import uuid

        alert = QualityAlert(
            id=str(uuid.uuid4()),
            threshold_id=threshold.id,
            metric_type=threshold.metric_type,
            current_value=current_value,
            threshold_value=threshold.threshold_value,
            severity=threshold.severity,
            message=f"{threshold.name}: {current_value:.3f} {threshold.operator} {threshold.threshold_value}",
            data_source=task_config.get("data_source"),
            table_name=task_config.get("table_name"),
        )

        self.alerts.append(alert)
        return alert

    def _handle_alert(self, alert: QualityAlert):
        """알림 처리"""
        self.logger.warning(f"품질 알림 발생: {alert.message}")

        # 콜백 함수 호출
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"알림 콜백 실행 실패: {e}")

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        acknowledged: Optional[bool] = None,
        hours: int = 24,
    ) -> List[QualityAlert]:
        """알림 조회"""
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]

        if severity is not None:
            filtered_alerts = [alert for alert in filtered_alerts if alert.severity == severity]

        if acknowledged is not None:
            filtered_alerts = [
                alert for alert in filtered_alerts if alert.acknowledged == acknowledged
            ]

        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """알림 확인 처리"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledge(user_id)
                self.logger.info(f"알림 확인 처리 완료: {alert_id}")
                return True
        return False

    def get_monitoring_status(self) -> Dict[str, Any]:
        """모니터링 상태 조회"""
        last_checks = [
            task.get("last_check")
            for task in self.monitoring_tasks.values()
            if task.get("last_check")
        ]
        last_check = max(last_checks) if last_checks else None

        return {
            "status": self.status.value,
            "check_interval": self.check_interval,
            "thresholds_count": len(self.thresholds),
            "monitoring_tasks_count": len(self.monitoring_tasks),
            "alerts_count": len(self.alerts),
            "unacknowledged_alerts": len([a for a in self.alerts if not a.acknowledged]),
            "last_check": last_check.isoformat() if last_check else None,
        }

    def get_quality_metrics(self, task_id: str) -> Optional[Dict[str, Any]]:
        """품질 메트릭 조회"""
        if task_id not in self.monitoring_tasks:
            return None

        task_config = self.monitoring_tasks[task_id]
        mock_data = self._generate_mock_data(task_config)

        if mock_data is None:
            return None

        quality_report = self.quality_checker.generate_quality_report(mock_data)

        return {
            "overall_score": quality_report.overall_score,
            "missing_value_score": quality_report.missing_value_score,
            "outlier_score": quality_report.outlier_score,
            "consistency_score": quality_report.consistency_score,
            "recommendations": quality_report.recommendations,
            "critical_issues": quality_report.critical_issues,
            "last_check": task_config.get("last_check"),
        }

    def export_alerts(self, file_path: str, hours: int = 24) -> bool:
        """알림 내보내기"""
        try:
            alerts = self.get_alerts(hours=hours)

            data = {
                "alerts": [asdict(alert) for alert in alerts],
                "exported_at": datetime.now().isoformat(),
                "total_alerts": len(alerts),
            }

            import json

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"알림 내보내기 완료: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"알림 내보내기 실패: {e}")
            return False
