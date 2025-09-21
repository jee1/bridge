"""MCP Analytics 도구 테스트"""

import unittest
import asyncio
import json
import numpy as np
from unittest.mock import AsyncMock, patch
from src.bridge.mcp_server_unified import UnifiedBridgeMCPServer


class TestMCPAnalyticsTools(unittest.TestCase):
    """MCP Analytics 도구 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.server = UnifiedBridgeMCPServer()
        self.server.mode = "development"  # 개발 모드로 설정
        
        # 테스트 데이터
        self.test_data = {
            'id': list(range(100)),
            'value1': np.random.normal(100, 15, 100).tolist(),
            'value2': np.random.normal(50, 10, 100).tolist(),
            'category': (['A', 'B', 'C'] * 33 + ['A'])[:100],
            'score': np.random.uniform(0, 100, 100).tolist()
        }
    
    def test_statistics_analyzer_descriptive(self):
        """통계 분석 도구 - 기술 통계 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "analysis_type": "descriptive"
            }
            result = await self.server._statistics_analyzer(args)
            
            self.assertTrue(result["success"])
            self.assertEqual(result["analysis_type"], "descriptive")
            self.assertIn("result", result)
            self.assertIn("data_shape", result)
        
        asyncio.run(run_test())
    
    def test_statistics_analyzer_correlation(self):
        """통계 분석 도구 - 상관관계 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "analysis_type": "correlation"
            }
            result = await self.server._statistics_analyzer(args)
            
            self.assertTrue(result["success"])
            self.assertEqual(result["analysis_type"], "correlation")
            self.assertIn("result", result)
        
        asyncio.run(run_test())
    
    def test_data_profiler(self):
        """데이터 프로파일링 도구 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "include_stats": True,
                "include_quality": True
            }
            result = await self.server._data_profiler(args)
            
            self.assertTrue(result["success"])
            self.assertIn("profile", result)
            self.assertIn("basic_info", result["profile"])
            self.assertIn("statistics", result["profile"])
            self.assertIn("quality", result["profile"])
        
        asyncio.run(run_test())
    
    def test_outlier_detector_iqr(self):
        """이상치 탐지 도구 - IQR 방법 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "method": "iqr",
                "columns": ["value1", "value2"]
            }
            result = await self.server._outlier_detector(args)
            
            self.assertTrue(result["success"])
            self.assertEqual(result["method"], "iqr")
            self.assertIn("outliers", result)
            self.assertIn("total_columns_analyzed", result)
        
        asyncio.run(run_test())
    
    def test_outlier_detector_zscore(self):
        """이상치 탐지 도구 - Z-score 방법 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "method": "zscore",
                "threshold": 2.5
            }
            result = await self.server._outlier_detector(args)
            
            self.assertTrue(result["success"])
            self.assertEqual(result["method"], "zscore")
            self.assertEqual(result["threshold"], 2.5)
            self.assertIn("outliers", result)
        
        asyncio.run(run_test())
    
    def test_chart_generator_bar(self):
        """차트 생성 도구 - 막대 차트 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "chart_type": "bar",
                "x_column": "category",
                "title": "Test Bar Chart"
            }
            result = await self.server._chart_generator(args)
            
            self.assertTrue(result["success"])
            self.assertEqual(result["chart_type"], "bar")
            self.assertEqual(result["title"], "Test Bar Chart")
            self.assertIn("chart_data", result)
        
        asyncio.run(run_test())
    
    def test_chart_generator_scatter(self):
        """차트 생성 도구 - 산점도 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "chart_type": "scatter",
                "x_column": "value1",
                "y_column": "value2",
                "hue_column": "category",
                "title": "Test Scatter Plot"
            }
            result = await self.server._chart_generator(args)
            
            self.assertTrue(result["success"])
            self.assertEqual(result["chart_type"], "scatter")
            self.assertEqual(result["title"], "Test Scatter Plot")
            self.assertIn("chart_data", result)
        
        asyncio.run(run_test())
    
    def test_chart_generator_histogram(self):
        """차트 생성 도구 - 히스토그램 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "chart_type": "histogram",
                "x_column": "value1",
                "title": "Test Histogram"
            }
            result = await self.server._chart_generator(args)
            
            self.assertTrue(result["success"])
            self.assertEqual(result["chart_type"], "histogram")
            self.assertEqual(result["title"], "Test Histogram")
            self.assertIn("chart_data", result)
        
        asyncio.run(run_test())
    
    def test_quality_checker(self):
        """데이터 품질 검사 도구 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "check_missing": True,
                "check_outliers": True,
                "check_consistency": True,
                "outlier_method": "iqr"
            }
            result = await self.server._quality_checker(args)
            
            self.assertTrue(result["success"])
            self.assertIn("quality_report", result)
            self.assertIn("overall_score", result["quality_report"])
            self.assertIn("missing_values", result["quality_report"])
            self.assertIn("outliers", result["quality_report"])
            self.assertIn("consistency", result["quality_report"])
        
        asyncio.run(run_test())
    
    def test_report_builder(self):
        """리포트 생성 도구 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "title": "Test Report",
                "author": "Test Author",
                "include_charts": True,
                "include_dashboard": True,
                "include_quality": True
            }
            result = await self.server._report_builder(args)
            
            self.assertTrue(result["success"])
            self.assertIn("report", result)
            self.assertEqual(result["report"]["title"], "Test Report")
            self.assertEqual(result["report"]["author"], "Test Author")
            self.assertIn("basic_stats", result["report"])
            self.assertIn("charts", result["report"])
            self.assertIn("dashboard", result["report"])
            self.assertIn("quality", result["report"])
        
        asyncio.run(run_test())
    
    def test_invalid_analysis_type(self):
        """잘못된 분석 유형 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "analysis_type": "invalid_type"
            }
            result = await self.server._statistics_analyzer(args)
            
            self.assertFalse(result["success"])
            self.assertIn("error", result)
        
        asyncio.run(run_test())
    
    def test_missing_required_columns(self):
        """필수 컬럼 누락 테스트"""
        async def run_test():
            args = {
                "data": self.test_data,
                "chart_type": "line"
                # x_column, y_column 누락
            }
            result = await self.server._chart_generator(args)
            
            self.assertFalse(result["success"])
            self.assertIn("error", result)
        
        asyncio.run(run_test())
    
    def test_empty_data(self):
        """빈 데이터 테스트"""
        async def run_test():
            args = {
                "data": {},
                "analysis_type": "descriptive"
            }
            result = await self.server._statistics_analyzer(args)
            
            # 빈 데이터도 처리되어야 함
            self.assertTrue(result["success"])
        
        asyncio.run(run_test())


class TestMCPToolDefinitions(unittest.TestCase):
    """MCP 도구 정의 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.server = UnifiedBridgeMCPServer()
        self.tools = self.server._define_tools()
    
    def test_tool_definitions(self):
        """도구 정의 검증"""
        tool_names = [tool["name"] for tool in self.tools]
        
        # 기존 도구들
        self.assertIn("query_database", tool_names)
        self.assertIn("get_schema", tool_names)
        self.assertIn("analyze_data", tool_names)
        self.assertIn("list_connectors", tool_names)
        
        # Analytics 도구들
        self.assertIn("statistics_analyzer", tool_names)
        self.assertIn("data_profiler", tool_names)
        self.assertIn("outlier_detector", tool_names)
        self.assertIn("chart_generator", tool_names)
        self.assertIn("quality_checker", tool_names)
        self.assertIn("report_builder", tool_names)
    
    def test_statistics_analyzer_schema(self):
        """statistics_analyzer 스키마 검증"""
        tool = next(tool for tool in self.tools if tool["name"] == "statistics_analyzer")
        
        self.assertEqual(tool["name"], "statistics_analyzer")
        self.assertIn("inputSchema", tool)
        
        schema = tool["inputSchema"]
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        
        # 필수 필드 검증
        self.assertIn("data", schema["required"])
        self.assertIn("analysis_type", schema["required"])
        
        # 속성 검증
        properties = schema["properties"]
        self.assertIn("data", properties)
        self.assertIn("columns", properties)
        self.assertIn("analysis_type", properties)
        
        # analysis_type enum 검증
        self.assertEqual(properties["analysis_type"]["enum"], 
                        ["descriptive", "distribution", "correlation", "summary"])
    
    def test_chart_generator_schema(self):
        """chart_generator 스키마 검증"""
        tool = next(tool for tool in self.tools if tool["name"] == "chart_generator")
        
        self.assertEqual(tool["name"], "chart_generator")
        
        schema = tool["inputSchema"]
        properties = schema["properties"]
        
        # 필수 필드 검증
        self.assertIn("data", schema["required"])
        self.assertIn("chart_type", schema["required"])
        
        # chart_type enum 검증
        self.assertEqual(properties["chart_type"]["enum"], 
                        ["bar", "line", "scatter", "histogram", "box", "heatmap"])
    
    def test_quality_checker_schema(self):
        """quality_checker 스키마 검증"""
        tool = next(tool for tool in self.tools if tool["name"] == "quality_checker")
        
        self.assertEqual(tool["name"], "quality_checker")
        
        schema = tool["inputSchema"]
        properties = schema["properties"]
        
        # 필수 필드 검증
        self.assertIn("data", schema["required"])
        
        # 기본값 검증
        self.assertTrue(properties["check_missing"]["default"])
        self.assertTrue(properties["check_outliers"]["default"])
        self.assertTrue(properties["check_consistency"]["default"])
        self.assertEqual(properties["outlier_method"]["default"], "iqr")


if __name__ == '__main__':
    unittest.main()
