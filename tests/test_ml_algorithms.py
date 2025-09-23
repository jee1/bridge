"""고급 분석 알고리즘 테스트"""

import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bridge.analytics.core.data_integration import UnifiedDataFrame
from bridge.ml.algorithms import (
    AnomalyDetector,
    ClusteringAnalyzer,
    DimensionalityReducer,
    TimeSeriesAnalyzer,
)


class TestTimeSeriesAnalyzer(unittest.TestCase):
    """시계열 분석기 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.analyzer = TimeSeriesAnalyzer()

        # 시계열 데이터 생성
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        values = np.sin(np.arange(100) * 0.1) + np.random.normal(0, 0.1, 100)

        self.ts_data = pd.DataFrame({"date": dates, "value": values})
        self.ts_df = UnifiedDataFrame(self.ts_data)

    def test_analyze_stationarity(self):
        """정상성 분석 테스트"""
        result = self.analyzer.analyze_stationarity(self.ts_df, "date", "value")

        self.assertIn("is_stationary", result)
        self.assertIn("adf_statistic", result)
        self.assertIn("adf_pvalue", result)
        self.assertIsInstance(result["is_stationary"], bool)

    def test_decompose_time_series(self):
        """시계열 분해 테스트"""
        result = self.analyzer.decompose_time_series(self.ts_df, "date", "value")

        self.assertIn("trend", result)
        self.assertIn("seasonal", result)
        self.assertIn("residual", result)
        self.assertIn("observed", result)

    def test_detect_trend(self):
        """트렌드 탐지 테스트"""
        result = self.analyzer.detect_trend(self.ts_df, "date", "value")

        self.assertIn("trend_direction", result)
        self.assertIn("slope", result)
        self.assertIn("r_squared", result)
        self.assertIn("is_significant", result)

    def test_detect_seasonality(self):
        """계절성 탐지 테스트"""
        result = self.analyzer.detect_seasonality(self.ts_df, "date", "value")

        self.assertIn("has_seasonality", result)
        self.assertIn("dominant_frequency", result)
        self.assertIn("period", result)


class TestAnomalyDetector(unittest.TestCase):
    """이상 탐지기 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.detector = AnomalyDetector()

        # 정상 데이터 생성
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 3))

        # 이상치 추가
        anomaly_data = np.random.normal(5, 0.5, (10, 3))

        # 데이터 결합
        all_data = np.vstack([normal_data, anomaly_data])
        np.random.shuffle(all_data)

        self.anomaly_df = UnifiedDataFrame(pd.DataFrame(all_data, columns=["x1", "x2", "x3"]))

    def test_detect_with_isolation_forest(self):
        """Isolation Forest 이상 탐지 테스트"""
        result = self.detector.detect_with_isolation_forest(self.anomaly_df, contamination=0.1)

        self.assertEqual(result.model_type, "IsolationForest")
        self.assertEqual(len(result.anomaly_scores), 110)
        self.assertEqual(len(result.is_anomaly), 110)
        self.assertGreater(len(result.anomaly_indices), 0)

    def test_detect_with_one_class_svm(self):
        """One-Class SVM 이상 탐지 테스트"""
        result = self.detector.detect_with_one_class_svm(self.anomaly_df, nu=0.1)

        self.assertEqual(result.model_type, "OneClassSVM")
        self.assertEqual(len(result.anomaly_scores), 110)
        self.assertEqual(len(result.is_anomaly), 110)

    def test_detect_with_zscore(self):
        """Z-Score 이상 탐지 테스트"""
        result = self.detector.detect_with_zscore(self.anomaly_df, threshold=3.0)

        self.assertEqual(result.model_type, "ZScore")
        self.assertEqual(len(result.anomaly_scores), 110)
        self.assertEqual(len(result.is_anomaly), 110)

    def test_get_anomaly_stats(self):
        """이상 탐지 통계 테스트"""
        result = self.detector.detect_with_isolation_forest(self.anomaly_df, contamination=0.1)

        stats = self.detector.get_anomaly_stats(result)

        self.assertEqual(stats.total_points, 110)
        self.assertGreaterEqual(stats.anomaly_count, 0)
        self.assertGreaterEqual(stats.normal_count, 0)
        self.assertAlmostEqual(stats.anomaly_ratio + stats.normal_ratio, 1.0, places=5)


class TestClusteringAnalyzer(unittest.TestCase):
    """클러스터링 분석기 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.analyzer = ClusteringAnalyzer()

        # 클러스터링용 데이터 생성
        np.random.seed(42)

        # 3개의 클러스터 생성
        cluster1 = np.random.normal([2, 2], 0.5, (30, 2))
        cluster2 = np.random.normal([-2, -2], 0.5, (30, 2))
        cluster3 = np.random.normal([2, -2], 0.5, (30, 2))

        all_data = np.vstack([cluster1, cluster2, cluster3])
        np.random.shuffle(all_data)

        self.cluster_df = UnifiedDataFrame(pd.DataFrame(all_data, columns=["x", "y"]))

    def test_kmeans_clustering(self):
        """K-means 클러스터링 테스트"""
        result = self.analyzer.kmeans_clustering(self.cluster_df, n_clusters=3)

        self.assertEqual(result.model_type, "KMeans")
        self.assertEqual(result.n_clusters, 3)
        self.assertEqual(len(result.labels), 90)
        self.assertIsNotNone(result.cluster_centers)
        self.assertEqual(result.cluster_centers.shape, (3, 2))

    def test_dbscan_clustering(self):
        """DBSCAN 클러스터링 테스트"""
        result = self.analyzer.dbscan_clustering(self.cluster_df, eps=0.5, min_samples=5)

        self.assertEqual(result.model_type, "DBSCAN")
        self.assertEqual(len(result.labels), 90)
        self.assertGreaterEqual(result.n_clusters, 0)

    def test_hierarchical_clustering(self):
        """Hierarchical 클러스터링 테스트"""
        result = self.analyzer.hierarchical_clustering(self.cluster_df, n_clusters=3)

        self.assertEqual(result.model_type, "Hierarchical")
        self.assertEqual(result.n_clusters, 3)
        self.assertEqual(len(result.labels), 90)

    def test_find_optimal_clusters(self):
        """최적 클러스터 수 찾기 테스트"""
        result = self.analyzer.find_optimal_clusters(
            self.cluster_df, max_clusters=5, method="kmeans"
        )

        self.assertIn("optimal_clusters", result)
        self.assertIn("silhouette_scores", result)
        self.assertGreater(result["optimal_clusters"], 1)
        self.assertLessEqual(result["optimal_clusters"], 5)

    def test_get_cluster_stats(self):
        """클러스터 통계 테스트"""
        clustering_result = self.analyzer.kmeans_clustering(self.cluster_df, n_clusters=3)

        stats = self.analyzer.get_cluster_stats(clustering_result, self.cluster_df)

        self.assertEqual(len(stats), 3)
        for stat in stats:
            self.assertGreater(stat.size, 0)
            self.assertGreater(stat.percentage, 0)


class TestDimensionalityReducer(unittest.TestCase):
    """차원 축소기 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.reducer = DimensionalityReducer()

        # 고차원 데이터 생성
        np.random.seed(42)
        high_dim_data = np.random.randn(100, 10)

        self.high_dim_df = UnifiedDataFrame(
            pd.DataFrame(high_dim_data, columns=[f"feature_{i}" for i in range(10)])
        )

    def test_pca_reduction(self):
        """PCA 차원 축소 테스트"""
        result = self.reducer.pca_reduction(self.high_dim_df, n_components=3)

        self.assertEqual(result.model_type, "PCA")
        self.assertEqual(result.n_components, 3)
        self.assertEqual(result.transformed_data.shape, (100, 3))
        self.assertIsNotNone(result.explained_variance_ratio)
        self.assertEqual(len(result.explained_variance_ratio), 3)

    def test_tsne_reduction(self):
        """t-SNE 차원 축소 테스트"""
        result = self.reducer.tsne_reduction(self.high_dim_df, n_components=2)

        self.assertEqual(result.model_type, "t-SNE")
        self.assertEqual(result.n_components, 2)
        self.assertEqual(result.transformed_data.shape, (100, 2))

    def test_ica_reduction(self):
        """ICA 차원 축소 테스트"""
        result = self.reducer.ica_reduction(self.high_dim_df, n_components=3)

        self.assertEqual(result.model_type, "ICA")
        self.assertEqual(result.n_components, 3)
        self.assertEqual(result.transformed_data.shape, (100, 3))

    def test_find_optimal_components(self):
        """최적 컴포넌트 수 찾기 테스트"""
        result = self.reducer.find_optimal_components(
            self.high_dim_df, method="pca", max_components=5
        )

        self.assertIn("optimal_components", result)
        self.assertIn("explained_variance_ratio", result)
        self.assertGreater(result["optimal_components"], 0)
        self.assertLessEqual(result["optimal_components"], 5)

    def test_get_component_info(self):
        """주성분 정보 테스트"""
        pca_result = self.reducer.pca_reduction(self.high_dim_df, n_components=3)

        components = self.reducer.get_component_info(pca_result)

        self.assertEqual(len(components), 3)
        for i, component in enumerate(components):
            self.assertEqual(component.component_id, i)
            self.assertGreater(component.explained_variance_ratio, 0)


if __name__ == "__main__":
    unittest.main()
