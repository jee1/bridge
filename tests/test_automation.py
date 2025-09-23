"""자동화 모듈 테스트"""

import time
import unittest
from datetime import datetime, timedelta

from src.bridge.automation import (
    AlertRule,
    NotificationChannel,
    NotificationSystem,
    QualityAlert,
    QualityMonitor,
    QualityThreshold,
    ReportAutomation,
    ReportSchedule,
    ReportTemplate,
    ScheduledTask,
    TaskScheduler,
    TaskStatus,
)
from src.bridge.automation.notification_system import AlertPriority
from src.bridge.automation.quality_monitor import AlertSeverity, MonitorStatus
from src.bridge.automation.report_automation import ReportStatus, ScheduleType
from src.bridge.automation.scheduler import TaskType


class TestQualityMonitor(unittest.TestCase):
    """품질 모니터 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.monitor = QualityMonitor(check_interval=1)  # 1초 간격으로 설정

        # 테스트 임계값
        self.threshold = QualityThreshold(
            id="test_threshold_1",
            name="Missing Value Threshold",
            metric_type="missing_ratio",
            threshold_value=0.1,
            operator="gt",
            severity=AlertSeverity.WARNING,
        )

    def test_add_threshold(self):
        """임계값 추가 테스트"""
        result = self.monitor.add_threshold(self.threshold)
        self.assertTrue(result)
        self.assertIn("test_threshold_1", self.monitor.thresholds)

    def test_add_monitoring_task(self):
        """모니터링 작업 추가 테스트"""
        result = self.monitor.add_monitoring_task("task_1", "test_db", "users")
        self.assertTrue(result)
        self.assertIn("task_1", self.monitor.monitoring_tasks)

    def test_start_stop_monitoring(self):
        """모니터링 시작/중지 테스트"""
        # 시작
        result = self.monitor.start_monitoring()
        self.assertTrue(result)
        self.assertEqual(self.monitor.status, MonitorStatus.RUNNING)

        # 중지
        result = self.monitor.stop_monitoring()
        self.assertTrue(result)
        self.assertEqual(self.monitor.status, MonitorStatus.STOPPED)

    def test_get_monitoring_status(self):
        """모니터링 상태 조회 테스트"""
        self.monitor.add_threshold(self.threshold)
        self.monitor.add_monitoring_task("task_1", "test_db", "users")

        status = self.monitor.get_monitoring_status()
        self.assertIn("status", status)
        self.assertIn("thresholds_count", status)
        self.assertIn("monitoring_tasks_count", status)
        self.assertEqual(status["thresholds_count"], 1)
        self.assertEqual(status["monitoring_tasks_count"], 1)


class TestReportAutomation(unittest.TestCase):
    """리포트 자동화 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.automation = ReportAutomation()

        # 테스트 템플릿
        self.template = ReportTemplate(
            id="test_template_1",
            name="Test Report",
            description="Test report template",
            data_source="test_db",
            output_format="html",
        )

        # 테스트 스케줄
        self.schedule = ReportSchedule(
            id="test_schedule_1",
            template_id="test_template_1",
            schedule_type=ScheduleType.DAILY,
            start_time=datetime.now() + timedelta(seconds=1),
        )

    def test_create_template(self):
        """템플릿 생성 테스트"""
        result = self.automation.create_template(self.template)
        self.assertTrue(result)
        self.assertIn("test_template_1", self.automation.templates)

    def test_create_schedule(self):
        """스케줄 생성 테스트"""
        self.automation.create_template(self.template)
        result = self.automation.create_schedule(self.schedule)
        self.assertTrue(result)
        self.assertIn("test_schedule_1", self.automation.schedules)

    def test_execute_template(self):
        """템플릿 즉시 실행 테스트"""
        self.automation.create_template(self.template)

        job_id = self.automation.execute_template("test_template_1")
        self.assertIsNotNone(job_id)

        # 작업 상태 확인
        time.sleep(2)  # 작업 완료 대기
        job = self.automation.get_job_status(job_id)
        self.assertIsNotNone(job)
        self.assertIn(job.status, [ReportStatus.COMPLETED, ReportStatus.FAILED])

    def test_get_scheduler_status(self):
        """스케줄러 상태 조회 테스트"""
        self.automation.create_template(self.template)
        self.automation.create_schedule(self.schedule)

        status = self.automation.get_scheduler_status()
        self.assertIn("status", status)
        self.assertIn("templates_count", status)
        self.assertIn("schedules_count", status)
        self.assertEqual(status["templates_count"], 1)
        self.assertEqual(status["schedules_count"], 1)


class TestNotificationSystem(unittest.TestCase):
    """알림 시스템 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.notification = NotificationSystem()

        # 테스트 규칙
        self.rule = AlertRule(
            id="test_rule_1",
            name="Test Alert Rule",
            description="Test alert rule",
            event_type="quality_alert",
            conditions={"severity": "warning"},
            channels=["console"],
            priority=AlertPriority.MEDIUM,
        )

    def test_add_rule(self):
        """알림 규칙 추가 테스트"""
        result = self.notification.add_rule(self.rule)
        self.assertTrue(result)
        self.assertIn("test_rule_1", self.notification.rules)

    def test_send_notification(self):
        """알림 발송 테스트"""
        self.notification.add_rule(self.rule)

        data = {"severity": "warning", "message": "Test alert message"}

        sent_messages = self.notification.send_notification("quality_alert", data)
        self.assertGreater(len(sent_messages), 0)

    def test_test_notification(self):
        """알림 테스트"""
        result = self.notification.test_notification("console", "Test notification")
        self.assertTrue(result)

    def test_get_notification_stats(self):
        """알림 통계 조회 테스트"""
        self.notification.add_rule(self.rule)

        # 테스트 알림 발송
        data = {"severity": "warning", "message": "Test"}
        self.notification.send_notification("quality_alert", data)

        stats = self.notification.get_notification_stats()
        self.assertIn("total_messages", stats)
        self.assertIn("sent_messages", stats)
        self.assertIn("messages_by_priority", stats)


class TestTaskScheduler(unittest.TestCase):
    """작업 스케줄러 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.scheduler = TaskScheduler()

        # 테스트 작업
        def test_function(message="Hello"):
            return f"Test: {message}"

        self.task = ScheduledTask(
            id="test_task_1",
            name="Test Task",
            description="Test scheduled task",
            task_type=TaskType.CUSTOM,
            function=test_function,
            parameters={"message": "World"},
            cron_expression="* * * * *",  # 매분 실행
            enabled=True,
        )

    def test_add_task(self):
        """작업 추가 테스트"""
        result = self.scheduler.add_task(self.task)
        self.assertTrue(result)
        self.assertIn("test_task_1", self.scheduler.tasks)

    def test_execute_task_now(self):
        """작업 즉시 실행 테스트"""
        self.scheduler.add_task(self.task)

        execution_id = self.scheduler.execute_task_now("test_task_1")
        self.assertIsNotNone(execution_id)

        # 실행 기록 확인
        executions = self.scheduler.get_executions(task_id="test_task_1")
        self.assertGreater(len(executions), 0)

    def test_get_task_status(self):
        """작업 상태 조회 테스트"""
        self.scheduler.add_task(self.task)

        status = self.scheduler.get_task_status("test_task_1")
        self.assertIsNotNone(status)
        self.assertEqual(status["task_id"], "test_task_1")
        self.assertEqual(status["name"], "Test Task")
        self.assertTrue(status["enabled"])

    def test_get_scheduler_status(self):
        """스케줄러 상태 조회 테스트"""
        self.scheduler.add_task(self.task)

        status = self.scheduler.get_scheduler_status()
        self.assertIn("status", status)
        self.assertIn("total_tasks", status)
        self.assertIn("enabled_tasks", status)
        self.assertEqual(status["total_tasks"], 1)
        self.assertEqual(status["enabled_tasks"], 1)


class TestIntegration(unittest.TestCase):
    """통합 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.quality_monitor = QualityMonitor()
        self.report_automation = ReportAutomation()
        self.notification = NotificationSystem()
        self.scheduler = TaskScheduler()

    def test_end_to_end_automation(self):
        """전체 자동화 워크플로우 테스트"""
        # 1. 품질 모니터링 설정
        threshold = QualityThreshold(
            id="integration_threshold",
            name="Integration Test Threshold",
            metric_type="missing_ratio",
            threshold_value=0.1,
            operator="gt",
            severity=AlertSeverity.WARNING,
        )
        self.quality_monitor.add_threshold(threshold)
        self.quality_monitor.add_monitoring_task("integration_task", "test_db", "users")

        # 2. 알림 규칙 설정
        rule = AlertRule(
            id="integration_rule",
            name="Integration Alert Rule",
            event_type="quality_alert",
            conditions={"severity": "warning"},
            channels=["console"],
            priority=AlertPriority.MEDIUM,
        )
        self.notification.add_rule(rule)

        # 3. 리포트 템플릿 설정
        template = ReportTemplate(
            id="integration_template",
            name="Integration Report",
            data_source="test_db",
            output_format="html",
        )
        self.report_automation.create_template(template)

        # 4. 스케줄된 작업 설정
        def quality_check_job():
            # 품질 검사 수행
            return {"status": "completed"}

        task = ScheduledTask(
            id="integration_task",
            name="Integration Quality Check",
            task_type=TaskType.DATA_QUALITY_CHECK,
            function=quality_check_job,
            cron_expression="0 */6 * * *",  # 6시간마다 실행
            enabled=True,
        )
        self.scheduler.add_task(task)

        # 5. 검증
        self.assertEqual(len(self.quality_monitor.thresholds), 1)
        self.assertEqual(len(self.notification.rules), 1)
        self.assertEqual(len(self.report_automation.templates), 1)
        self.assertEqual(len(self.scheduler.tasks), 1)

        # 6. 작업 즉시 실행
        execution_id = self.scheduler.execute_task_now("integration_task")
        self.assertIsNotNone(execution_id)

        # 7. 상태 확인
        quality_status = self.quality_monitor.get_monitoring_status()
        notification_stats = self.notification.get_notification_stats()
        report_status = self.report_automation.get_scheduler_status()
        scheduler_status = self.scheduler.get_scheduler_status()

        self.assertIn("thresholds_count", quality_status)
        self.assertIn("total_messages", notification_stats)
        self.assertIn("templates_count", report_status)
        self.assertIn("total_tasks", scheduler_status)


if __name__ == "__main__":
    unittest.main()
