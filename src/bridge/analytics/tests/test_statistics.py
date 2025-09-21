"""통계 분석 모듈 테스트"""

import unittest
import numpy as np
import pandas as pd
from bridge.analytics.core.statistics import StatisticsAnalyzer, DescriptiveStats, CorrelationResult
from bridge.analytics.core.data_integration import UnifiedDataFrame


class TestStatisticsAnalyzer(unittest.TestCase):
    """StatisticsAnalyzer 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.analyzer = StatisticsAnalyzer()
        
        # 테스트 데이터 생성
        np.random.seed(42)
        self.test_data = {
            'id': range(100),
            'value1': np.random.normal(100, 15, 100),
            'value2': np.random.normal(50, 10, 100),
            'value3': np.random.normal(200, 30, 100),
            'category': ['A', 'B', 'C'] * 33 + ['A'],
            'missing_col': [1, 2, 3, None, 5] * 20
        }
        
        self.test_df = UnifiedDataFrame(self.test_data)
    
    def test_calculate_descriptive_stats(self):
        """기술 통계 계산 테스트"""
        stats = self.analyzer.calculate_descriptive_stats(self.test_df)
        
        # 결과 검증
        self.assertIsInstance(stats, dict)
        self.assertIn('value1', stats)
        self.assertIn('value2', stats)
        self.assertIn('value3', stats)
        self.assertIn('missing_col', stats)
        
        # value1 통계 검증
        value1_stats = stats['value1']
        self.assertIsInstance(value1_stats, DescriptiveStats)
        self.assertEqual(value1_stats.count, 100)
        self.assertGreater(value1_stats.mean, 0)
        self.assertGreater(value1_stats.std, 0)
        self.assertLess(value1_stats.min, value1_stats.max)
        
        # missing_col 통계 검증 (결측값 포함)
        missing_stats = stats['missing_col']
        self.assertEqual(missing_stats.count, 80)  # 100 - 20 (None 개수)
        self.assertEqual(missing_stats.missing_count, 20)
        self.assertEqual(missing_stats.missing_ratio, 0.2)
    
    def test_calculate_descriptive_stats_with_columns(self):
        """특정 컬럼만 기술 통계 계산 테스트"""
        stats = self.analyzer.calculate_descriptive_stats(
            self.test_df, 
            columns=['value1', 'value2']
        )
        
        # 결과 검증
        self.assertEqual(len(stats), 2)
        self.assertIn('value1', stats)
        self.assertIn('value2', stats)
        self.assertNotIn('value3', stats)
    
    def test_calculate_correlation(self):
        """상관관계 분석 테스트"""
        corr_result = self.analyzer.calculate_correlation(self.test_df)
        
        # 결과 검증
        self.assertIsInstance(corr_result, CorrelationResult)
        self.assertIsInstance(corr_result.correlation_matrix, pd.DataFrame)
        self.assertIsInstance(corr_result.strong_correlations, list)
        self.assertIsInstance(corr_result.moderate_correlations, list)
        
        # 상관계수 매트릭스 크기 검증 (id 컬럼도 포함됨)
        expected_size = 5  # id, value1, value2, value3, missing_col
        self.assertEqual(corr_result.correlation_matrix.shape, (expected_size, expected_size))
    
    def test_calculate_correlation_with_columns(self):
        """특정 컬럼만 상관관계 분석 테스트"""
        corr_result = self.analyzer.calculate_correlation(
            self.test_df,
            columns=['value1', 'value2']
        )
        
        # 결과 검증
        expected_size = 2
        self.assertEqual(corr_result.correlation_matrix.shape, (expected_size, expected_size))
    
    def test_calculate_distribution_stats(self):
        """분포 통계 계산 테스트"""
        dist_stats = self.analyzer.calculate_distribution_stats(self.test_df)
        
        # 결과 검증
        self.assertIsInstance(dist_stats, dict)
        self.assertIn('value1', dist_stats)
        
        # value1 분포 통계 검증
        value1_dist = dist_stats['value1']
        self.assertIn('skewness', value1_dist)
        self.assertIn('kurtosis', value1_dist)
        self.assertIn('variance', value1_dist)
        self.assertIn('range', value1_dist)
        self.assertIn('iqr', value1_dist)
        
        # 값 범위 검증
        self.assertGreaterEqual(value1_dist['range'], 0)
        self.assertGreaterEqual(value1_dist['iqr'], 0)
    
    def test_generate_summary_report(self):
        """종합 통계 요약 리포트 생성 테스트"""
        report = self.analyzer.generate_summary_report(self.test_df)
        
        # 결과 검증
        self.assertIsInstance(report, dict)
        self.assertIn('data_overview', report)
        self.assertIn('descriptive_stats', report)
        self.assertIn('correlation_analysis', report)
        self.assertIn('distribution_stats', report)
        
        # 데이터 개요 검증
        overview = report['data_overview']
        self.assertEqual(overview['total_rows'], 100)
        self.assertGreater(overview['numeric_columns'], 0)
        
        # 기술 통계 검증
        desc_stats = report['descriptive_stats']
        self.assertIsInstance(desc_stats, dict)
        self.assertIn('value1', desc_stats)
        
        # 상관관계 분석 검증
        corr_analysis = report['correlation_analysis']
        self.assertIn('strong_correlations', corr_analysis)
        self.assertIn('moderate_correlations', corr_analysis)
        self.assertIn('correlation_matrix', corr_analysis)
    
    def test_empty_dataframe(self):
        """빈 데이터프레임 처리 테스트"""
        empty_df = UnifiedDataFrame([])
        
        # 기술 통계
        stats = self.analyzer.calculate_descriptive_stats(empty_df)
        self.assertEqual(len(stats), 0)
        
        # 상관관계 분석
        corr_result = self.analyzer.calculate_correlation(empty_df)
        self.assertEqual(len(corr_result.correlation_matrix), 0)
        
        # 분포 통계
        dist_stats = self.analyzer.calculate_distribution_stats(empty_df)
        self.assertEqual(len(dist_stats), 0)
    
    def test_single_column_dataframe(self):
        """단일 컬럼 데이터프레임 처리 테스트"""
        single_col_df = UnifiedDataFrame({'value': [1, 2, 3, 4, 5]})
        
        # 기술 통계
        stats = self.analyzer.calculate_descriptive_stats(single_col_df)
        self.assertEqual(len(stats), 1)
        self.assertIn('value', stats)
        
        # 상관관계 분석 (단일 컬럼이므로 빈 결과)
        corr_result = self.analyzer.calculate_correlation(single_col_df)
        self.assertEqual(len(corr_result.correlation_matrix), 0)
    
    def test_all_missing_values(self):
        """모든 값이 결측값인 컬럼 처리 테스트"""
        missing_df = UnifiedDataFrame({'missing': [None, None, None, None, None]})
        
        stats = self.analyzer.calculate_descriptive_stats(missing_df)
        # 모든 값이 결측값인 경우 숫자형 컬럼으로 인식되지 않을 수 있음
        if 'missing' in stats:
            missing_stats = stats['missing']
            self.assertEqual(missing_stats.count, 0)
            self.assertEqual(missing_stats.missing_count, 5)
            self.assertEqual(missing_stats.missing_ratio, 1.0)
        else:
            # 모든 값이 결측값인 경우 빈 결과가 반환될 수 있음
            self.assertEqual(len(stats), 0)


if __name__ == '__main__':
    unittest.main()
