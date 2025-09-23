"""대시보드 모듈 테스트"""

import time
import unittest
from datetime import datetime, timedelta

from src.bridge.dashboard import (
    AlertManager,
    ChartRenderer,
    DashboardConfig,
    DashboardManager,
    DashboardWidget,
    LayoutManager,
    MetricCollector,
    MonitoringDashboard,
    PerformanceMetrics,
    RealTimeMonitor,
    SystemMetrics,
    VisualizationEngine,
)
from src.bridge.dashboard.dashboard_manager import LayoutType, WidgetType
from src.bridge.dashboard.monitoring_dashboard import AlertLevel, MetricType
from src.bridge.dashboard.real_time_monitor import ConnectionStatus
from src.bridge.dashboard.visualization_engine import (
    ChartConfig,
    ChartType,
    LayoutConfig,
    LayoutType,
)


class TestDashboardManager(unittest.TestCase):
    """대시보드 관리자 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.manager = DashboardManager()

        # 테스트 대시보드 설정
        self.dashboard_config = DashboardConfig(
            id="test_dashboard",
            name="Test Dashboard",
            description="Test dashboard for unit testing",
            layout_type=LayoutType.GRID,
            grid_columns=4,
            grid_rows=3,
        )

    def test_create_dashboard(self):
        """대시보드 생성 테스트"""
        result = self.manager.create_dashboard(self.dashboard_config)
        self.assertTrue(result)

        dashboard = self.manager.get_dashboard("test_dashboard")
        self.assertIsNotNone(dashboard)
        self.assertEqual(dashboard["name"], "Test Dashboard")

    def test_add_widget(self):
        """위젯 추가 테스트"""
        self.manager.create_dashboard(self.dashboard_config)

        widget = DashboardWidget(
            id="test_widget",
            widget_type=WidgetType.CHART,
            title="Test Chart",
            position={"x": 0, "y": 0, "width": 2, "height": 1},
            config={"chart_type": "line", "data_source": "cpu_metrics"},
        )

        result = self.manager.add_widget(widget, "test_dashboard")
        self.assertTrue(result)

        widgets = self.manager.get_dashboard_widgets("test_dashboard")
        self.assertEqual(len(widgets), 1)
        self.assertEqual(widgets[0].title, "Test Chart")

    def test_get_dashboard_data(self):
        """대시보드 데이터 조회 테스트"""
        self.manager.create_dashboard(self.dashboard_config)

        widget = DashboardWidget(
            id="test_widget",
            widget_type=WidgetType.METRIC,
            title="Test Metric",
            position={"x": 0, "y": 0, "width": 1, "height": 1},
            config={"value": 75, "unit": "%"},
        )

        self.manager.add_widget(widget, "test_dashboard")

        dashboard_data = self.manager.get_dashboard_data("test_dashboard")
        self.assertIsNotNone(dashboard_data)
        self.assertIn("widgets", dashboard_data)
        self.assertEqual(len(dashboard_data["widgets"]), 1)

    def test_list_dashboards(self):
        """대시보드 목록 조회 테스트"""
        dashboards = self.manager.list_dashboards()
        self.assertGreater(len(dashboards), 0)

        # 기본 대시보드들이 있는지 확인
        dashboard_ids = [d["id"] for d in dashboards]
        self.assertIn("system_monitoring", dashboard_ids)
        self.assertIn("data_quality", dashboard_ids)
        self.assertIn("analytics_reports", dashboard_ids)


class TestMonitoringDashboard(unittest.TestCase):
    """모니터링 대시보드 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.dashboard = MonitoringDashboard(refresh_interval=1)

    def test_start_stop_monitoring(self):
        """모니터링 시작/중지 테스트"""
        # 시작
        result = self.dashboard.start_monitoring()
        self.assertTrue(result)
        self.assertTrue(self.dashboard.is_monitoring)

        # 잠시 대기
        time.sleep(2)

        # 중지
        result = self.dashboard.stop_monitoring()
        self.assertTrue(result)
        self.assertFalse(self.dashboard.is_monitoring)

    def test_get_current_metrics(self):
        """현재 메트릭 조회 테스트"""
        self.dashboard.start_monitoring()
        time.sleep(2)  # 메트릭 수집 대기

        metrics = self.dashboard.get_current_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn("system", metrics)
        self.assertIn("is_monitoring", metrics)

        self.dashboard.stop_monitoring()

    def test_set_threshold(self):
        """임계값 설정 테스트"""
        result = self.dashboard.set_threshold(MetricType.CPU, 70.0, 90.0)
        self.assertTrue(result)

        # 임계값 확인
        thresholds = self.dashboard.thresholds[MetricType.CPU]
        self.assertEqual(thresholds["warning"], 70.0)
        self.assertEqual(thresholds["critical"], 90.0)

    def test_get_dashboard_summary(self):
        """대시보드 요약 정보 테스트"""
        summary = self.dashboard.get_dashboard_summary()
        self.assertIn("status", summary)
        self.assertIn("refresh_interval", summary)
        self.assertIn("metrics_count", summary)
        self.assertIn("current_status", summary)
        self.assertIn("alerts_summary", summary)


class TestRealTimeMonitor(unittest.TestCase):
    """실시간 모니터링 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.monitor = RealTimeMonitor()

    def test_add_remove_connection(self):
        """연결 추가/제거 테스트"""
        client_id = "test_client"

        # 연결 추가
        self.monitor.add_connection(client_id)
        self.assertIn(client_id, self.monitor.connections)
        self.assertEqual(self.monitor.get_connection_status(client_id), ConnectionStatus.CONNECTED)

        # 연결 제거
        self.monitor.remove_connection(client_id)
        self.assertNotIn(client_id, self.monitor.connections)
        self.assertEqual(
            self.monitor.get_connection_status(client_id), ConnectionStatus.DISCONNECTED
        )

    def test_register_collectors(self):
        """수집기 등록 테스트"""
        self.monitor.register_system_collector()
        self.monitor.register_application_collector()

        self.assertEqual(len(self.monitor.metric_collector.collectors), 2)
        self.assertIn("system", self.monitor.metric_collector.collectors)
        self.assertIn("application", self.monitor.metric_collector.collectors)

    def test_start_stop_monitoring(self):
        """모니터링 시작/중지 테스트"""
        self.monitor.register_system_collector()

        # 시작
        self.monitor.start_monitoring(collection_interval=1)
        self.assertTrue(self.monitor.is_running)

        # 잠시 대기
        time.sleep(2)

        # 중지
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.is_running)

    def test_get_monitoring_status(self):
        """모니터링 상태 조회 테스트"""
        status = self.monitor.get_monitoring_status()
        self.assertIn("is_running", status)
        self.assertIn("active_connections", status)
        self.assertIn("connection_status", status)
        self.assertIn("queue_size", status)


class TestVisualizationEngine(unittest.TestCase):
    """시각화 엔진 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.engine = VisualizationEngine()
        self.chart_renderer = ChartRenderer()
        self.layout_manager = LayoutManager()

    def test_render_line_chart(self):
        """선 차트 렌더링 테스트"""
        data = [
            {"timestamp": "2024-01-01T00:00:00", "value": 50, "unit": "%"},
            {"timestamp": "2024-01-01T01:00:00", "value": 60, "unit": "%"},
            {"timestamp": "2024-01-01T02:00:00", "value": 55, "unit": "%"},
        ]

        config = ChartConfig(
            chart_type=ChartType.LINE, title="CPU Usage", x_axis="timestamp", y_axis="value"
        )

        result = self.chart_renderer.render_chart(data, config)
        self.assertIn("type", result)
        self.assertEqual(result["type"], "chart")
        self.assertEqual(result["chart_type"], "line")
        self.assertIn("image", result)

    def test_render_metric(self):
        """메트릭 카드 렌더링 테스트"""
        data = [{"value": 75, "unit": "%"}]

        config = ChartConfig(
            chart_type=ChartType.METRIC,
            title="CPU Usage",
            custom_config={"threshold_warning": 70, "threshold_critical": 90},
        )

        result = self.chart_renderer.render_chart(data, config)
        self.assertIn("type", result)
        self.assertEqual(result["type"], "metric")
        self.assertEqual(result["value"], 75)
        self.assertEqual(result["unit"], "%")

    def test_render_table(self):
        """테이블 렌더링 테스트"""
        data = [
            {"name": "CPU", "value": 75, "unit": "%"},
            {"name": "Memory", "value": 60, "unit": "%"},
            {"name": "Disk", "value": 45, "unit": "%"},
        ]

        config = ChartConfig(chart_type=ChartType.TABLE, title="System Metrics")

        result = self.chart_renderer.render_chart(data, config)
        self.assertIn("type", result)
        self.assertEqual(result["type"], "table")
        self.assertIn("html", result)
        self.assertIn("data", result)

    def test_create_grid_layout(self):
        """그리드 레이아웃 생성 테스트"""
        widgets = [
            {"id": "widget1", "type": "chart", "title": "Chart 1"},
            {"id": "widget2", "type": "metric", "title": "Metric 1"},
            {"id": "widget3", "type": "table", "title": "Table 1"},
        ]

        config = LayoutConfig(layout_type=LayoutType.GRID, columns=2, rows=2)

        layout = self.layout_manager.create_layout(widgets, config)
        self.assertEqual(layout["type"], "grid")
        self.assertEqual(layout["columns"], 2)
        self.assertEqual(layout["rows"], 2)
        self.assertEqual(len(layout["widgets"]), 3)

    def test_create_sample_data(self):
        """샘플 데이터 생성 테스트"""
        data = self.engine.create_sample_data("cpu", 10)
        self.assertEqual(len(data), 10)

        for item in data:
            self.assertIn("timestamp", item)
            self.assertIn("value", item)
            self.assertIn("unit", item)
            self.assertEqual(item["unit"], "%")

    def test_render_dashboard(self):
        """대시보드 렌더링 테스트"""
        dashboard_config = {
            "id": "test_dashboard",
            "name": "Test Dashboard",
            "layout_type": "grid",
            "grid_columns": 2,
            "grid_rows": 2,
            "widgets": [
                {
                    "id": "cpu_chart",
                    "type": "chart",
                    "title": "CPU Usage",
                    "config": {"chart_type": "line", "width": 400, "height": 300},
                },
                {
                    "id": "memory_metric",
                    "type": "metric",
                    "title": "Memory Usage",
                    "config": {"width": 200, "height": 100},
                },
            ],
        }

        widgets_data = {
            "cpu_chart": self.engine.create_sample_data("cpu", 20),
            "memory_metric": [{"value": 65, "unit": "%"}],
        }

        result = self.engine.render_dashboard(dashboard_config, widgets_data)
        self.assertEqual(result["status"], "success")
        self.assertIn("layout", result)
        self.assertIn("rendered_at", result)


class TestIntegration(unittest.TestCase):
    """통합 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.dashboard_manager = DashboardManager()
        self.monitoring_dashboard = MonitoringDashboard()
        self.real_time_monitor = RealTimeMonitor()
        self.visualization_engine = VisualizationEngine()

    def test_end_to_end_dashboard(self):
        """전체 대시보드 워크플로우 테스트"""
        # 1. 대시보드 생성
        dashboard_config = DashboardConfig(
            id="integration_dashboard",
            name="Integration Dashboard",
            description="Integration test dashboard",
            layout_type=LayoutType.GRID,
            grid_columns=3,
            grid_rows=2,
        )

        result = self.dashboard_manager.create_dashboard(dashboard_config)
        self.assertTrue(result)

        # 2. 위젯 추가
        cpu_widget = DashboardWidget(
            id="cpu_widget",
            widget_type=WidgetType.CHART,
            title="CPU Usage",
            position={"x": 0, "y": 0, "width": 1, "height": 1},
            config={"chart_type": "line", "data_source": "cpu_metrics"},
        )

        memory_widget = DashboardWidget(
            id="memory_widget",
            widget_type=WidgetType.METRIC,
            title="Memory Usage",
            position={"x": 1, "y": 0, "width": 1, "height": 1},
            config={"value": 65, "unit": "%"},
        )

        self.dashboard_manager.add_widget(cpu_widget, "integration_dashboard")
        self.dashboard_manager.add_widget(memory_widget, "integration_dashboard")

        # 3. 모니터링 시작
        self.monitoring_dashboard.start_monitoring()
        time.sleep(2)  # 메트릭 수집 대기

        # 4. 실시간 모니터링 설정
        self.real_time_monitor.register_system_collector()
        self.real_time_monitor.start_monitoring()
        time.sleep(2)

        # 5. 대시보드 데이터 조회
        dashboard_data = self.dashboard_manager.get_dashboard_data("integration_dashboard")
        self.assertIsNotNone(dashboard_data)
        self.assertEqual(len(dashboard_data["widgets"]), 2)

        # 6. 모니터링 상태 확인
        monitoring_status = self.monitoring_dashboard.get_dashboard_summary()
        self.assertIn("status", monitoring_status)

        real_time_status = self.real_time_monitor.get_monitoring_status()
        self.assertIn("is_running", real_time_status)

        # 7. 정리
        self.monitoring_dashboard.stop_monitoring()
        self.real_time_monitor.stop_monitoring()


if __name__ == "__main__":
    unittest.main()
