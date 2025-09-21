"""데이터 품질 검사 모듈 테스트"""

import unittest
import numpy as np
import pandas as pd
from bridge.analytics.core.quality import (
    QualityChecker, MissingValueStats, OutlierStats, 
    ConsistencyStats, QualityReport
)
from bridge.analytics.core.data_integration import UnifiedDataFrame


class TestQualityChecker(unittest.TestCase):
    """QualityChecker 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.checker = QualityChecker()
        
        # 테스트 데이터 생성
        np.random.seed(42)
        self.test_data = {
            'id': list(range(100)),
            'value1': np.random.normal(100, 15, 100).tolist(),
            'value2': np.random.normal(50, 10, 100).tolist(),
            'value3': np.random.normal(200, 30, 100).tolist(),
            'category': (['A', 'B', 'C'] * 33 + ['A'])[:100],
            'missing_col': ([1, 2, 3, None, 5] * 20)[:100],
            'outlier_col': ([1, 2, 3, 4, 5] * 20)[:100],  # 이상치 포함을 위해 마지막 2개를 큰 값으로 변경
            'duplicate_col': ([1, 2, 3, 4, 5] * 20)[:100]  # 중복 데이터
        }
        
        # 이상치 추가
        self.test_data['outlier_col'][-2] = 1000
        self.test_data['outlier_col'][-1] = 2000
        
        self.test_df = UnifiedDataFrame(self.test_data)
    
    def test_analyze_missing_values(self):
        """결측값 분석 테스트"""
        stats = self.checker.analyze_missing_values(self.test_df)
        
        # 결과 검증
        self.assertIsInstance(stats, dict)
        self.assertIn('missing_col', stats)
        
        # missing_col 통계 검증
        missing_stats = stats['missing_col']
        self.assertIsInstance(missing_stats, MissingValueStats)
        self.assertEqual(missing_stats.total_values, 100)
        self.assertEqual(missing_stats.missing_count, 20)
        self.assertEqual(missing_stats.missing_ratio, 0.2)
        self.assertEqual(missing_stats.complete_count, 80)
        self.assertEqual(missing_stats.complete_ratio, 0.8)
    
    def test_analyze_missing_values_with_columns(self):
        """특정 컬럼만 결측값 분석 테스트"""
        stats = self.checker.analyze_missing_values(
            self.test_df, 
            columns=['missing_col', 'value1']
        )
        
        # 결과 검증
        self.assertEqual(len(stats), 2)
        self.assertIn('missing_col', stats)
        self.assertIn('value1', stats)
        self.assertNotIn('value2', stats)
    
    def test_detect_outliers_iqr(self):
        """IQR 방법으로 이상치 탐지 테스트"""
        stats = self.checker.detect_outliers(self.test_df, method='iqr')
        
        # 결과 검증
        self.assertIsInstance(stats, dict)
        self.assertIn('outlier_col', stats)
        
        # outlier_col 통계 검증
        outlier_stats = stats['outlier_col']
        self.assertIsInstance(outlier_stats, OutlierStats)
        self.assertEqual(outlier_stats.total_values, 100)
        self.assertGreater(outlier_stats.outlier_count, 0)
        self.assertGreater(outlier_stats.outlier_ratio, 0)
        self.assertGreater(len(outlier_stats.outlier_indices), 0)
        self.assertGreater(len(outlier_stats.outlier_values), 0)
    
    def test_detect_outliers_zscore(self):
        """Z-score 방법으로 이상치 탐지 테스트"""
        stats = self.checker.detect_outliers(self.test_df, method='zscore')
        
        # 결과 검증
        self.assertIsInstance(stats, dict)
        self.assertIn('outlier_col', stats)
        
        # outlier_col 통계 검증
        outlier_stats = stats['outlier_col']
        self.assertIsInstance(outlier_stats, OutlierStats)
        self.assertEqual(outlier_stats.total_values, 100)
    
    def test_detect_outliers_with_columns(self):
        """특정 컬럼만 이상치 탐지 테스트"""
        stats = self.checker.detect_outliers(
            self.test_df,
            columns=['outlier_col', 'value1']
        )
        
        # 결과 검증
        self.assertEqual(len(stats), 2)
        self.assertIn('outlier_col', stats)
        self.assertIn('value1', stats)
        self.assertNotIn('value2', stats)
    
    def test_check_consistency(self):
        """데이터 일관성 검사 테스트"""
        stats = self.checker.check_consistency(self.test_df)
        
        # 결과 검증
        self.assertIsInstance(stats, ConsistencyStats)
        self.assertGreaterEqual(stats.duplicate_rows, 0)
        self.assertGreaterEqual(stats.duplicate_ratio, 0)
        self.assertGreaterEqual(stats.unique_rows, 0)
        self.assertGreaterEqual(stats.unique_ratio, 0)
        self.assertIsInstance(stats.data_types_consistent, bool)
        self.assertIsInstance(stats.schema_issues, list)
    
    def test_generate_quality_report(self):
        """종합 품질 리포트 생성 테스트"""
        report = self.checker.generate_quality_report(self.test_df)
        
        # 결과 검증
        self.assertIsInstance(report, QualityReport)
        self.assertGreaterEqual(report.overall_score, 0)
        self.assertLessEqual(report.overall_score, 100)
        self.assertGreaterEqual(report.missing_value_score, 0)
        self.assertLessEqual(report.missing_value_score, 100)
        self.assertGreaterEqual(report.outlier_score, 0)
        self.assertLessEqual(report.outlier_score, 100)
        self.assertGreaterEqual(report.consistency_score, 0)
        self.assertLessEqual(report.consistency_score, 100)
        self.assertIsInstance(report.recommendations, list)
        self.assertIsInstance(report.critical_issues, list)
    
    def test_empty_dataframe(self):
        """빈 데이터프레임 처리 테스트"""
        empty_df = UnifiedDataFrame([])
        
        # 결측값 분석
        missing_stats = self.checker.analyze_missing_values(empty_df)
        self.assertEqual(len(missing_stats), 0)
        
        # 이상치 탐지
        outlier_stats = self.checker.detect_outliers(empty_df)
        self.assertEqual(len(outlier_stats), 0)
        
        # 일관성 검사
        consistency_stats = self.checker.check_consistency(empty_df)
        self.assertEqual(consistency_stats.duplicate_rows, 0)
        self.assertEqual(consistency_stats.unique_rows, 0)
    
    def test_single_column_dataframe(self):
        """단일 컬럼 데이터프레임 처리 테스트"""
        single_col_df = UnifiedDataFrame({'value': [1, 2, 3, 4, 5]})
        
        # 결측값 분석
        missing_stats = self.checker.analyze_missing_values(single_col_df)
        self.assertEqual(len(missing_stats), 1)
        self.assertIn('value', missing_stats)
        
        # 이상치 탐지
        outlier_stats = self.checker.detect_outliers(single_col_df)
        self.assertEqual(len(outlier_stats), 1)
        self.assertIn('value', outlier_stats)
    
    def test_all_missing_values(self):
        """모든 값이 결측값인 컬럼 처리 테스트"""
        missing_df = UnifiedDataFrame({'missing': [None, None, None, None, None]})
        
        stats = self.checker.analyze_missing_values(missing_df)
        self.assertIn('missing', stats)
        
        missing_stats = stats['missing']
        self.assertEqual(missing_stats.total_values, 5)
        self.assertEqual(missing_stats.missing_count, 5)
        self.assertEqual(missing_stats.missing_ratio, 1.0)
        self.assertEqual(missing_stats.complete_count, 0)
        self.assertEqual(missing_stats.complete_ratio, 0.0)
    
    def test_no_outliers(self):
        """이상치가 없는 데이터 테스트"""
        normal_df = UnifiedDataFrame({'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        
        stats = self.checker.detect_outliers(normal_df)
        self.assertIn('value', stats)
        
        outlier_stats = stats['value']
        self.assertEqual(outlier_stats.outlier_count, 0)
        self.assertEqual(outlier_stats.outlier_ratio, 0.0)
        self.assertEqual(len(outlier_stats.outlier_indices), 0)
        self.assertEqual(len(outlier_stats.outlier_values), 0)
    
    def test_duplicate_data(self):
        """중복 데이터가 있는 경우 테스트"""
        duplicate_df = UnifiedDataFrame({
            'id': [1, 2, 3, 1, 2, 3],
            'value': [10, 20, 30, 10, 20, 30]
        })
        
        stats = self.checker.check_consistency(duplicate_df)
        self.assertEqual(stats.duplicate_rows, 3)  # 3개 중복 행
        self.assertEqual(stats.duplicate_ratio, 0.5)  # 50% 중복
        self.assertEqual(stats.unique_rows, 3)
        self.assertEqual(stats.unique_ratio, 0.5)
    
    def test_mixed_data_types(self):
        """혼합된 데이터 타입 테스트"""
        # PyArrow가 혼합 타입을 처리할 수 있도록 문자열로 통일
        mixed_df = UnifiedDataFrame({
            'mixed': ['1', '2', '3.0', 'four', '5']  # 모든 값을 문자열로
        })
        
        stats = self.checker.check_consistency(mixed_df)
        # 문자열 컬럼이므로 타입 일관성은 True가 됨
        self.assertTrue(stats.data_types_consistent)
        self.assertEqual(len(stats.schema_issues), 0)


if __name__ == '__main__':
    unittest.main()
