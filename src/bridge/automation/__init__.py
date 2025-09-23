"""Bridge 자동화 모듈

데이터 품질 모니터링, 자동 리포트 생성, 알림 시스템 등의 자동화 기능을 제공합니다.
"""

from .notification_system import AlertRule, NotificationChannel, NotificationSystem
from .quality_monitor import QualityAlert, QualityMonitor, QualityThreshold
from .report_automation import ReportAutomation, ReportSchedule, ReportTemplate
from .scheduler import ScheduledTask, TaskScheduler, TaskStatus

__all__ = [
    # 품질 모니터링
    "QualityMonitor",
    "QualityAlert",
    "QualityThreshold",
    # 리포트 자동화
    "ReportAutomation",
    "ReportSchedule",
    "ReportTemplate",
    # 알림 시스템
    "NotificationSystem",
    "NotificationChannel",
    "AlertRule",
    # 스케줄러
    "TaskScheduler",
    "ScheduledTask",
    "TaskStatus",
]
