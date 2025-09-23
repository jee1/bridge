"""클러스터링 알고리즘 모듈

K-means, DBSCAN, Hierarchical Clustering 등을 활용한 클러스터링을 제공합니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bridge.analytics.core.data_integration import UnifiedDataFrame


@dataclass
class ClusteringResult:
    """클러스터링 결과를 담는 데이터 클래스"""

    labels: np.ndarray
    n_clusters: int
    cluster_centers: Optional[np.ndarray] = None
    model_type: str = ""
    model_metrics: Optional[Dict[str, float]] = None
    silhouette_score: Optional[float] = None
    inertia: Optional[float] = None


@dataclass
class ClusterStats:
    """클러스터 통계를 담는 데이터 클래스"""

    cluster_id: int
    size: int
    percentage: float
    center: Optional[np.ndarray] = None
    avg_distance_to_center: Optional[float] = None
    max_distance_to_center: Optional[float] = None
    min_distance_to_center: Optional[float] = None


class ClusteringAnalyzer:
    """클러스터링을 수행하는 클래스"""

    def __init__(self):
        """클러스터링 분석기 초기화"""
        self.logger = __import__("logging").getLogger(__name__)

    def kmeans_clustering(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        n_clusters: int = 3,
        random_state: int = 42,
        max_iter: int = 300,
    ) -> ClusteringResult:
        """K-means 클러스터링

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_clusters: 클러스터 수
            random_state: 랜덤 시드
            max_iter: 최대 반복 횟수

        Returns:
            클러스터링 결과
        """
        try:
            pandas_df = df.to_pandas()

            # 분석할 컬럼 선택
            if columns is None:
                numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in pandas_df.columns]

            if not numeric_columns:
                raise ValueError("분석할 수 있는 숫자형 컬럼이 없습니다")

            # 데이터 준비 및 정규화
            data = pandas_df[numeric_columns].fillna(0)

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # K-means 모델
            from sklearn.cluster import KMeans

            model = KMeans(
                n_clusters=n_clusters, random_state=random_state, max_iter=max_iter, n_init=10
            )

            # 모델 피팅
            labels = model.fit_predict(data_scaled)

            # 실루엣 점수 계산
            from sklearn.metrics import silhouette_score

            silhouette_avg = silhouette_score(data_scaled, labels)

            # 모델 메트릭
            metrics = {
                "inertia": model.inertia_,
                "silhouette_score": silhouette_avg,
                "n_iter": model.n_iter_,
                "n_clusters": n_clusters,
            }

            return ClusteringResult(
                labels=labels,
                n_clusters=n_clusters,
                cluster_centers=model.cluster_centers_,
                model_type="KMeans",
                model_metrics=metrics,
                silhouette_score=silhouette_avg,
                inertia=model.inertia_,
            )

        except Exception as e:
            self.logger.error(f"K-means 클러스터링 중 오류 발생: {e}")
            return ClusteringResult(
                labels=np.array([]),
                n_clusters=0,
                model_type="KMeans",
                model_metrics={"error": str(e)},
            )

    def dbscan_clustering(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> ClusteringResult:
        """DBSCAN 클러스터링

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            eps: 이웃 반경
            min_samples: 최소 샘플 수

        Returns:
            클러스터링 결과
        """
        try:
            pandas_df = df.to_pandas()

            # 분석할 컬럼 선택
            if columns is None:
                numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in pandas_df.columns]

            if not numeric_columns:
                raise ValueError("분석할 수 있는 숫자형 컬럼이 없습니다")

            # 데이터 준비 및 정규화
            data = pandas_df[numeric_columns].fillna(0)

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # DBSCAN 모델
            from sklearn.cluster import DBSCAN

            model = DBSCAN(eps=eps, min_samples=min_samples)

            # 모델 피팅
            labels = model.fit_predict(data_scaled)

            # 클러스터 수 계산 (노이즈 제외)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # 실루엣 점수 계산 (노이즈가 아닌 점들만)
            if n_clusters > 1:
                from sklearn.metrics import silhouette_score

                mask = labels != -1
                if np.sum(mask) > 1:
                    silhouette_avg = silhouette_score(data_scaled[mask], labels[mask])
                else:
                    silhouette_avg = -1
            else:
                silhouette_avg = -1

            # 모델 메트릭
            n_noise = list(labels).count(-1)
            metrics = {
                "eps": eps,
                "min_samples": min_samples,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_ratio": n_noise / len(labels) if len(labels) > 0 else 0,
                "silhouette_score": silhouette_avg,
            }

            return ClusteringResult(
                labels=labels,
                n_clusters=n_clusters,
                model_type="DBSCAN",
                model_metrics=metrics,
                silhouette_score=silhouette_avg,
            )

        except Exception as e:
            self.logger.error(f"DBSCAN 클러스터링 중 오류 발생: {e}")
            return ClusteringResult(
                labels=np.array([]),
                n_clusters=0,
                model_type="DBSCAN",
                model_metrics={"error": str(e)},
            )

    def hierarchical_clustering(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        n_clusters: int = 3,
        linkage: str = "ward",
    ) -> ClusteringResult:
        """Hierarchical Clustering

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_clusters: 클러스터 수
            linkage: 연결 방법 ('ward', 'complete', 'average', 'single')

        Returns:
            클러스터링 결과
        """
        try:
            pandas_df = df.to_pandas()

            # 분석할 컬럼 선택
            if columns is None:
                numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in pandas_df.columns]

            if not numeric_columns:
                raise ValueError("분석할 수 있는 숫자형 컬럼이 없습니다")

            # 데이터 준비 및 정규화
            data = pandas_df[numeric_columns].fillna(0)

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Hierarchical Clustering 모델
            from sklearn.cluster import AgglomerativeClustering

            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

            # 모델 피팅
            labels = model.fit_predict(data_scaled)

            # 실루엣 점수 계산
            from sklearn.metrics import silhouette_score

            silhouette_avg = silhouette_score(data_scaled, labels)

            # 모델 메트릭
            metrics = {
                "n_clusters": n_clusters,
                "linkage": linkage,
                "silhouette_score": silhouette_avg,
            }

            return ClusteringResult(
                labels=labels,
                n_clusters=n_clusters,
                model_type="Hierarchical",
                model_metrics=metrics,
                silhouette_score=silhouette_avg,
            )

        except Exception as e:
            self.logger.error(f"Hierarchical 클러스터링 중 오류 발생: {e}")
            return ClusteringResult(
                labels=np.array([]),
                n_clusters=0,
                model_type="Hierarchical",
                model_metrics={"error": str(e)},
            )

    def gaussian_mixture_clustering(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        n_components: int = 3,
        random_state: int = 42,
    ) -> ClusteringResult:
        """Gaussian Mixture Model 클러스터링

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_components: 컴포넌트 수
            random_state: 랜덤 시드

        Returns:
            클러스터링 결과
        """
        try:
            pandas_df = df.to_pandas()

            # 분석할 컬럼 선택
            if columns is None:
                numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in pandas_df.columns]

            if not numeric_columns:
                raise ValueError("분석할 수 있는 숫자형 컬럼이 없습니다")

            # 데이터 준비 및 정규화
            data = pandas_df[numeric_columns].fillna(0)

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # Gaussian Mixture 모델
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(n_components=n_components, random_state=random_state)

            # 모델 피팅
            labels = model.fit_predict(data_scaled)

            # 실루엣 점수 계산
            from sklearn.metrics import silhouette_score

            silhouette_avg = silhouette_score(data_scaled, labels)

            # 모델 메트릭
            metrics = {
                "n_components": n_components,
                "aic": model.aic(data_scaled),
                "bic": model.bic(data_scaled),
                "converged": model.converged_,
                "n_iter": model.n_iter_,
                "silhouette_score": silhouette_avg,
            }

            return ClusteringResult(
                labels=labels,
                n_clusters=n_components,
                cluster_centers=model.means_,
                model_type="GaussianMixture",
                model_metrics=metrics,
                silhouette_score=silhouette_avg,
            )

        except Exception as e:
            self.logger.error(f"Gaussian Mixture 클러스터링 중 오류 발생: {e}")
            return ClusteringResult(
                labels=np.array([]),
                n_clusters=0,
                model_type="GaussianMixture",
                model_metrics={"error": str(e)},
            )

    def find_optimal_clusters(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        max_clusters: int = 10,
        method: str = "kmeans",
    ) -> Dict[str, Any]:
        """최적 클러스터 수 찾기

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            max_clusters: 최대 클러스터 수
            method: 클러스터링 방법 ('kmeans', 'hierarchical', 'gaussian_mixture')

        Returns:
            최적 클러스터 수와 메트릭
        """
        try:
            pandas_df = df.to_pandas()

            # 분석할 컬럼 선택
            if columns is None:
                numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in pandas_df.columns]

            if not numeric_columns:
                raise ValueError("분석할 수 있는 숫자형 컬럼이 없습니다")

            # 데이터 준비 및 정규화
            data = pandas_df[numeric_columns].fillna(0)

            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)

            # 엘보우 방법과 실루엣 점수 계산
            inertias = []
            silhouette_scores = []
            k_range = range(2, min(max_clusters + 1, len(data)))

            for k in k_range:
                if method == "kmeans":
                    from sklearn.cluster import KMeans

                    model = KMeans(n_clusters=k, random_state=42)
                    labels = model.fit_predict(data_scaled)
                    inertias.append(model.inertia_)
                elif method == "hierarchical":
                    from sklearn.cluster import AgglomerativeClustering

                    model = AgglomerativeClustering(n_clusters=k)
                    labels = model.fit_predict(data_scaled)
                    inertias.append(0)  # Hierarchical은 inertia가 없음
                elif method == "gaussian_mixture":
                    from sklearn.mixture import GaussianMixture

                    model = GaussianMixture(n_components=k, random_state=42)
                    labels = model.fit_predict(data_scaled)
                    inertias.append(model.aic(data_scaled))

                # 실루엣 점수 계산
                from sklearn.metrics import silhouette_score

                silhouette_avg = silhouette_score(data_scaled, labels)
                silhouette_scores.append(silhouette_avg)

            # 최적 클러스터 수 찾기 (실루엣 점수 기준)
            optimal_k = k_range[np.argmax(silhouette_scores)]

            return {
                "optimal_clusters": optimal_k,
                "k_range": list(k_range),
                "inertias": inertias,
                "silhouette_scores": silhouette_scores,
                "max_silhouette_score": max(silhouette_scores),
                "method": method,
            }

        except Exception as e:
            self.logger.error(f"최적 클러스터 수 찾기 중 오류 발생: {e}")
            return {"error": str(e)}

    def get_cluster_stats(
        self, result: ClusteringResult, df: UnifiedDataFrame, columns: Optional[List[str]] = None
    ) -> List[ClusterStats]:
        """클러스터 통계 계산

        Args:
            result: 클러스터링 결과
            df: 원본 데이터
            columns: 분석한 컬럼 목록

        Returns:
            각 클러스터별 통계
        """
        try:
            pandas_df = df.to_pandas()

            # 분석한 컬럼 선택
            if columns is None:
                numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in pandas_df.columns]

            data = pandas_df[numeric_columns].fillna(0)

            stats = []
            total_points = len(result.labels)

            for cluster_id in range(result.n_clusters):
                mask = result.labels == cluster_id
                cluster_data = data[mask]
                cluster_size = np.sum(mask)

                if cluster_size > 0:
                    # 클러스터 중심까지의 거리 계산
                    if result.cluster_centers is not None:
                        center = result.cluster_centers[cluster_id]
                        distances = np.linalg.norm(cluster_data - center, axis=1)
                        avg_distance = np.mean(distances)
                        max_distance = np.max(distances)
                        min_distance = np.min(distances)
                    else:
                        center = None
                        avg_distance = None
                        max_distance = None
                        min_distance = None

                    stats.append(
                        ClusterStats(
                            cluster_id=cluster_id,
                            size=cluster_size,
                            percentage=cluster_size / total_points * 100,
                            center=center,
                            avg_distance_to_center=avg_distance,
                            max_distance_to_center=max_distance,
                            min_distance_to_center=min_distance,
                        )
                    )

            return stats

        except Exception as e:
            self.logger.error(f"클러스터 통계 계산 중 오류 발생: {e}")
            return []

    def compare_methods(
        self, df: UnifiedDataFrame, columns: Optional[List[str]] = None, n_clusters: int = 3
    ) -> Dict[str, ClusteringResult]:
        """여러 클러스터링 방법 비교

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_clusters: 클러스터 수

        Returns:
            각 방법별 클러스터링 결과
        """
        results = {}

        # K-means
        results["kmeans"] = self.kmeans_clustering(df, columns, n_clusters)

        # DBSCAN
        results["dbscan"] = self.dbscan_clustering(df, columns)

        # Hierarchical
        results["hierarchical"] = self.hierarchical_clustering(df, columns, n_clusters)

        # Gaussian Mixture
        results["gaussian_mixture"] = self.gaussian_mixture_clustering(df, columns, n_clusters)

        return results
