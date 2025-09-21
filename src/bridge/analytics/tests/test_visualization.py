"""데이터 시각화 모듈 테스트"""

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bridge.analytics.core.data_integration import UnifiedDataFrame
from bridge.analytics.core.visualization import (
    ChartConfig,
    ChartGenerator,
    DashboardConfig,
    DashboardGenerator,
    ReportConfig,
    ReportGenerator,
)


class TestChartGenerator(unittest.TestCase):
    """ChartGenerator 테스트 클래스"""

    def setUp(self):
        """테스트 설정"""
        self.generator = ChartGenerator()

        # 테스트 데이터 생성
        np.random.seed(42)
        self.test_data = {
            "id": list(range(100)),
            "value1": np.random.normal(100, 15, 100).tolist(),
            "value2": np.random.normal(50, 10, 100).tolist(),
            "value3": np.random.normal(200, 30, 100).tolist(),
            "category": (["A", "B", "C"] * 33 + ["A"])[:100],
            "score": np.random.uniform(0, 100, 100).tolist(),
        }

        self.test_df = UnifiedDataFrame(self.test_data)

    def test_create_bar_chart_count(self):
        """카운트 막대 차트 생성 테스트"""
        fig = self.generator.create_bar_chart(self.test_df, "category")

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Bar Chart: category")

        plt.close(fig)

    def test_create_bar_chart_grouped(self):
        """그룹별 막대 차트 생성 테스트"""
        fig = self.generator.create_bar_chart(
            self.test_df, "category", "value1", ChartConfig(title="Test Bar Chart")
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Test Bar Chart")

        plt.close(fig)

    def test_create_line_chart(self):
        """선 차트 생성 테스트"""
        fig = self.generator.create_line_chart(
            self.test_df, "id", "value1", ChartConfig(title="Test Line Chart")
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Test Line Chart")

        plt.close(fig)

    def test_create_scatter_plot(self):
        """산점도 생성 테스트"""
        fig = self.generator.create_scatter_plot(
            self.test_df, "value1", "value2", config=ChartConfig(title="Test Scatter Plot")
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Test Scatter Plot")

        plt.close(fig)

    def test_create_scatter_plot_with_hue(self):
        """색상 구분 산점도 생성 테스트"""
        fig = self.generator.create_scatter_plot(
            self.test_df,
            "value1",
            "value2",
            hue_column="category",
            config=ChartConfig(title="Test Scatter Plot with Hue"),
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Test Scatter Plot with Hue")

        plt.close(fig)

    def test_create_histogram(self):
        """히스토그램 생성 테스트"""
        fig = self.generator.create_histogram(
            self.test_df, "value1", config=ChartConfig(title="Test Histogram")
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Test Histogram")

        plt.close(fig)

    def test_create_box_plot_single(self):
        """단일 박스 플롯 생성 테스트"""
        fig = self.generator.create_box_plot(
            self.test_df, None, "value1", ChartConfig(title="Test Box Plot")
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Test Box Plot")

        plt.close(fig)

    def test_create_box_plot_grouped(self):
        """그룹별 박스 플롯 생성 테스트"""
        fig = self.generator.create_box_plot(
            self.test_df, "category", "value1", ChartConfig(title="Test Grouped Box Plot")
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Test Grouped Box Plot")

        plt.close(fig)

    def test_create_heatmap(self):
        """상관관계 히트맵 생성 테스트"""
        fig = self.generator.create_heatmap(self.test_df, config=ChartConfig(title="Test Heatmap"))

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Test Heatmap")

        plt.close(fig)

    def test_create_heatmap_with_columns(self):
        """특정 컬럼 히트맵 생성 테스트"""
        fig = self.generator.create_heatmap(
            self.test_df,
            config=ChartConfig(title="Test Heatmap with Columns"),
            columns=["value1", "value2", "value3"],
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig.get_axes()[0].get_title(), "Test Heatmap with Columns")

        plt.close(fig)


class TestDashboardGenerator(unittest.TestCase):
    """DashboardGenerator 테스트 클래스"""

    def setUp(self):
        """테스트 설정"""
        self.generator = DashboardGenerator()

        # 테스트 데이터 생성
        np.random.seed(42)
        self.test_data = {
            "id": list(range(100)),
            "value1": np.random.normal(100, 15, 100).tolist(),
            "value2": np.random.normal(50, 10, 100).tolist(),
            "category": (["A", "B", "C"] * 33 + ["A"])[:100],
            "score": np.random.uniform(0, 100, 100).tolist(),
        }

        self.test_df = UnifiedDataFrame(self.test_data)

    def test_create_analytics_dashboard(self):
        """분석 대시보드 생성 테스트"""
        fig = self.generator.create_analytics_dashboard(
            self.test_df, DashboardConfig(title="Test Dashboard")
        )

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig._suptitle.get_text(), "Test Dashboard")

        plt.close(fig)

    def test_create_analytics_dashboard_default_config(self):
        """기본 설정으로 대시보드 생성 테스트"""
        fig = self.generator.create_analytics_dashboard(self.test_df)

        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(fig._suptitle.get_text(), "Analytics Dashboard")

        plt.close(fig)


class TestReportGenerator(unittest.TestCase):
    """ReportGenerator 테스트 클래스"""

    def setUp(self):
        """테스트 설정"""
        self.generator = ReportGenerator()

        # 테스트 데이터 생성
        np.random.seed(42)
        self.test_data = {
            "id": list(range(100)),
            "value1": np.random.normal(100, 15, 100).tolist(),
            "value2": np.random.normal(50, 10, 100).tolist(),
            "category": (["A", "B", "C"] * 33 + ["A"])[:100],
            "score": np.random.uniform(0, 100, 100).tolist(),
        }

        self.test_df = UnifiedDataFrame(self.test_data)

    def test_generate_analytics_report(self):
        """분석 리포트 생성 테스트"""
        report = self.generator.generate_analytics_report(
            self.test_df, ReportConfig(title="Test Report", author="Test Author")
        )

        # 리포트 구조 검증
        self.assertIsInstance(report, dict)
        self.assertIn("title", report)
        self.assertIn("author", report)
        self.assertIn("basic_stats", report)
        self.assertIn("column_stats", report)
        self.assertIn("charts", report)
        self.assertIn("dashboard", report)

        # 제목 검증
        self.assertEqual(report["title"], "Test Report")
        self.assertEqual(report["author"], "Test Author")

        # 기본 통계 검증
        basic_stats = report["basic_stats"]
        self.assertEqual(basic_stats["total_rows"], 100)
        self.assertEqual(basic_stats["total_columns"], 5)
        self.assertGreaterEqual(basic_stats["numeric_columns"], 3)
        self.assertGreaterEqual(basic_stats["categorical_columns"], 1)

        # 컬럼 통계 검증
        column_stats = report["column_stats"]
        self.assertIn("value1", column_stats)
        self.assertIn("category", column_stats)

        # 차트 검증
        charts = report["charts"]
        self.assertIsInstance(charts, dict)

        # 대시보드 검증
        dashboard = report["dashboard"]
        self.assertIsInstance(dashboard, plt.Figure)

        plt.close(dashboard)
        for chart in charts.values():
            if isinstance(chart, plt.Figure):
                plt.close(chart)

    def test_generate_analytics_report_default_config(self):
        """기본 설정으로 리포트 생성 테스트"""
        report = self.generator.generate_analytics_report(self.test_df)

        self.assertIsInstance(report, dict)
        self.assertEqual(report["title"], "Analytics Report")
        self.assertIsNone(report["author"])
        self.assertIsNone(report["date"])

        # 차트 정리
        dashboard = report["dashboard"]
        plt.close(dashboard)
        for chart in report["charts"].values():
            if isinstance(chart, plt.Figure):
                plt.close(chart)


class TestConfigClasses(unittest.TestCase):
    """설정 클래스 테스트"""

    def test_chart_config(self):
        """ChartConfig 테스트"""
        config = ChartConfig(
            title="Test Chart",
            x_label="X Axis",
            y_label="Y Axis",
            figsize=(12, 8),
            style="darkgrid",
            color_palette="viridis",
            dpi=150,
        )

        self.assertEqual(config.title, "Test Chart")
        self.assertEqual(config.x_label, "X Axis")
        self.assertEqual(config.y_label, "Y Axis")
        self.assertEqual(config.figsize, (12, 8))
        self.assertEqual(config.style, "darkgrid")
        self.assertEqual(config.color_palette, "viridis")
        self.assertEqual(config.dpi, 150)

    def test_dashboard_config(self):
        """DashboardConfig 테스트"""
        config = DashboardConfig(
            title="Test Dashboard",
            layout=(3, 2),
            figsize=(20, 15),
            style="white",
            color_palette="Set3",
            dpi=120,
        )

        self.assertEqual(config.title, "Test Dashboard")
        self.assertEqual(config.layout, (3, 2))
        self.assertEqual(config.figsize, (20, 15))
        self.assertEqual(config.style, "white")
        self.assertEqual(config.color_palette, "Set3")
        self.assertEqual(config.dpi, 120)

    def test_report_config(self):
        """ReportConfig 테스트"""
        config = ReportConfig(
            title="Test Report",
            author="Test Author",
            date="2024-01-01",
            figsize=(16, 12),
            style="ticks",
            color_palette="husl",
            dpi=200,
        )

        self.assertEqual(config.title, "Test Report")
        self.assertEqual(config.author, "Test Author")
        self.assertEqual(config.date, "2024-01-01")
        self.assertEqual(config.figsize, (16, 12))
        self.assertEqual(config.style, "ticks")
        self.assertEqual(config.color_palette, "husl")
        self.assertEqual(config.dpi, 200)


if __name__ == "__main__":
    unittest.main()
