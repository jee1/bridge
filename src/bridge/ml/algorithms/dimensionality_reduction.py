"""차원 축소 알고리즘 모듈

PCA, t-SNE, UMAP 등을 활용한 차원 축소를 제공합니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bridge.analytics.core.data_integration import UnifiedDataFrame


@dataclass
class DimensionalityReductionResult:
    """차원 축소 결과를 담는 데이터 클래스"""

    transformed_data: np.ndarray
    n_components: int
    explained_variance_ratio: Optional[np.ndarray] = None
    cumulative_variance_ratio: Optional[np.ndarray] = None
    model_type: str = ""
    model_metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[np.ndarray] = None


@dataclass
class ComponentInfo:
    """주성분 정보를 담는 데이터 클래스"""

    component_id: int
    explained_variance_ratio: float
    cumulative_variance_ratio: float
    feature_importance: Optional[np.ndarray] = None


class DimensionalityReducer:
    """차원 축소를 수행하는 클래스"""

    def __init__(self):
        """차원 축소기 초기화"""
        self.logger = __import__("logging").getLogger(__name__)

    def pca_reduction(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        n_components: Optional[int] = None,
        explained_variance_threshold: float = 0.95,
    ) -> DimensionalityReductionResult:
        """PCA 차원 축소

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_components: 주성분 수 (None이면 자동 결정)
            explained_variance_threshold: 설명 분산 비율 임계값

        Returns:
            PCA 차원 축소 결과
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

            # PCA 모델
            from sklearn.decomposition import PCA

            if n_components is None:
                # 설명 분산 비율 기준으로 자동 결정
                pca = PCA()
                pca.fit(data_scaled)
                cumsum = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= explained_variance_threshold) + 1

            pca = PCA(n_components=n_components)
            transformed_data = pca.fit_transform(data_scaled)

            # 모델 메트릭
            metrics = {
                "n_components": n_components,
                "total_variance_explained": np.sum(pca.explained_variance_ratio_),
                "original_dimensions": data_scaled.shape[1],
                "reduction_ratio": n_components / data_scaled.shape[1],
            }

            return DimensionalityReductionResult(
                transformed_data=transformed_data,
                n_components=n_components,
                explained_variance_ratio=pca.explained_variance_ratio_,
                cumulative_variance_ratio=np.cumsum(pca.explained_variance_ratio_),
                model_type="PCA",
                model_metrics=metrics,
                feature_importance=pca.components_,
            )

        except Exception as e:
            self.logger.error(f"PCA 차원 축소 중 오류 발생: {e}")
            return DimensionalityReductionResult(
                transformed_data=np.array([]),
                n_components=0,
                model_type="PCA",
                model_metrics={"error": str(e)},
            )

    def tsne_reduction(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: int = 42,
    ) -> DimensionalityReductionResult:
        """t-SNE 차원 축소

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_components: 축소할 차원 수
            perplexity: t-SNE perplexity 파라미터
            learning_rate: 학습률
            n_iter: 반복 횟수
            random_state: 랜덤 시드

        Returns:
            t-SNE 차원 축소 결과
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

            # t-SNE 모델
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                learning_rate=learning_rate,
                n_iter=n_iter,
                random_state=random_state,
            )

            transformed_data = tsne.fit_transform(data_scaled)

            # 모델 메트릭
            metrics = {
                "n_components": n_components,
                "perplexity": perplexity,
                "learning_rate": learning_rate,
                "n_iter": n_iter,
                "original_dimensions": data_scaled.shape[1],
                "reduction_ratio": n_components / data_scaled.shape[1],
            }

            return DimensionalityReductionResult(
                transformed_data=transformed_data,
                n_components=n_components,
                model_type="t-SNE",
                model_metrics=metrics,
            )

        except Exception as e:
            self.logger.error(f"t-SNE 차원 축소 중 오류 발생: {e}")
            return DimensionalityReductionResult(
                transformed_data=np.array([]),
                n_components=0,
                model_type="t-SNE",
                model_metrics={"error": str(e)},
            )

    def umap_reduction(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        random_state: int = 42,
    ) -> DimensionalityReductionResult:
        """UMAP 차원 축소

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_components: 축소할 차원 수
            n_neighbors: 이웃 수
            min_dist: 최소 거리
            metric: 거리 메트릭
            random_state: 랜덤 시드

        Returns:
            UMAP 차원 축소 결과
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

            # UMAP 모델
            try:
                import umap

                umap_model = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    random_state=random_state,
                )

                transformed_data = umap_model.fit_transform(data_scaled)

                # 모델 메트릭
                metrics = {
                    "n_components": n_components,
                    "n_neighbors": n_neighbors,
                    "min_dist": min_dist,
                    "metric": metric,
                    "original_dimensions": data_scaled.shape[1],
                    "reduction_ratio": n_components / data_scaled.shape[1],
                }

                return DimensionalityReductionResult(
                    transformed_data=transformed_data,
                    n_components=n_components,
                    model_type="UMAP",
                    model_metrics=metrics,
                )

            except ImportError:
                self.logger.warning("UMAP 라이브러리가 설치되지 않았습니다. PCA로 대체합니다.")
                return self.pca_reduction(df, columns, n_components)

        except Exception as e:
            self.logger.error(f"UMAP 차원 축소 중 오류 발생: {e}")
            return DimensionalityReductionResult(
                transformed_data=np.array([]),
                n_components=0,
                model_type="UMAP",
                model_metrics={"error": str(e)},
            )

    def ica_reduction(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        n_components: Optional[int] = None,
        random_state: int = 42,
    ) -> DimensionalityReductionResult:
        """Independent Component Analysis (ICA) 차원 축소

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_components: 독립 성분 수
            random_state: 랜덤 시드

        Returns:
            ICA 차원 축소 결과
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

            # ICA 모델
            from sklearn.decomposition import FastICA

            if n_components is None:
                n_components = min(data_scaled.shape[1], data_scaled.shape[0])

            ica = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)

            transformed_data = ica.fit_transform(data_scaled)

            # 모델 메트릭
            metrics = {
                "n_components": n_components,
                "original_dimensions": data_scaled.shape[1],
                "reduction_ratio": n_components / data_scaled.shape[1],
                "convergence": ica.n_iter_ < 1000,
            }

            return DimensionalityReductionResult(
                transformed_data=transformed_data,
                n_components=n_components,
                model_type="ICA",
                model_metrics=metrics,
                feature_importance=ica.components_,
            )

        except Exception as e:
            self.logger.error(f"ICA 차원 축소 중 오류 발생: {e}")
            return DimensionalityReductionResult(
                transformed_data=np.array([]),
                n_components=0,
                model_type="ICA",
                model_metrics={"error": str(e)},
            )

    def lda_reduction(
        self,
        df: UnifiedDataFrame,
        target_column: str,
        columns: Optional[List[str]] = None,
        n_components: Optional[int] = None,
    ) -> DimensionalityReductionResult:
        """Linear Discriminant Analysis (LDA) 차원 축소

        Args:
            df: 입력 데이터
            target_column: 타겟 컬럼명
            columns: 분석할 컬럼 목록
            n_components: 축소할 차원 수

        Returns:
            LDA 차원 축소 결과
        """
        try:
            pandas_df = df.to_pandas()

            # 분석할 컬럼 선택
            if columns is None:
                numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numeric_columns = [col for col in columns if col in pandas_df.columns]

            # 타겟 컬럼 제외
            if target_column in numeric_columns:
                numeric_columns.remove(target_column)

            if not numeric_columns:
                raise ValueError("분석할 수 있는 숫자형 컬럼이 없습니다")

            # 데이터 준비
            X = pandas_df[numeric_columns].fillna(0)
            y = pandas_df[target_column]

            # LDA 모델
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

            if n_components is None:
                n_components = min(len(numeric_columns), len(np.unique(y)) - 1)

            lda = LinearDiscriminantAnalysis(n_components=n_components)
            transformed_data = lda.fit_transform(X, y)

            # 모델 메트릭
            metrics = {
                "n_components": n_components,
                "original_dimensions": X.shape[1],
                "reduction_ratio": n_components / X.shape[1],
                "explained_variance_ratio": lda.explained_variance_ratio_.sum(),
            }

            return DimensionalityReductionResult(
                transformed_data=transformed_data,
                n_components=n_components,
                explained_variance_ratio=lda.explained_variance_ratio_,
                model_type="LDA",
                model_metrics=metrics,
                feature_importance=lda.coef_,
            )

        except Exception as e:
            self.logger.error(f"LDA 차원 축소 중 오류 발생: {e}")
            return DimensionalityReductionResult(
                transformed_data=np.array([]),
                n_components=0,
                model_type="LDA",
                model_metrics={"error": str(e)},
            )

    def get_component_info(self, result: DimensionalityReductionResult) -> List[ComponentInfo]:
        """주성분 정보 추출

        Args:
            result: 차원 축소 결과

        Returns:
            주성분 정보 목록
        """
        try:
            components = []

            if result.explained_variance_ratio is not None:
                for i in range(result.n_components):
                    component = ComponentInfo(
                        component_id=i,
                        explained_variance_ratio=result.explained_variance_ratio[i],
                        cumulative_variance_ratio=(
                            result.cumulative_variance_ratio[i]
                            if result.cumulative_variance_ratio is not None
                            else None
                        ),
                        feature_importance=(
                            result.feature_importance[i]
                            if result.feature_importance is not None
                            else None
                        ),
                    )
                    components.append(component)

            return components

        except Exception as e:
            self.logger.error(f"주성분 정보 추출 중 오류 발생: {e}")
            return []

    def find_optimal_components(
        self,
        df: UnifiedDataFrame,
        columns: Optional[List[str]] = None,
        method: str = "pca",
        max_components: int = 10,
        explained_variance_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """최적 컴포넌트 수 찾기

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            method: 차원 축소 방법
            max_components: 최대 컴포넌트 수
            explained_variance_threshold: 설명 분산 비율 임계값

        Returns:
            최적 컴포넌트 수와 메트릭
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

            if method == "pca":
                from sklearn.decomposition import PCA

                # 모든 컴포넌트로 PCA 수행
                pca = PCA()
                pca.fit(data_scaled)

                # 설명 분산 비율 계산
                explained_variance_ratio = pca.explained_variance_ratio_
                cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

                # 임계값을 만족하는 최소 컴포넌트 수
                optimal_components = (
                    np.argmax(cumulative_variance_ratio >= explained_variance_threshold) + 1
                )

                return {
                    "optimal_components": optimal_components,
                    "explained_variance_ratio": explained_variance_ratio.tolist(),
                    "cumulative_variance_ratio": cumulative_variance_ratio.tolist(),
                    "total_variance_explained": cumulative_variance_ratio[optimal_components - 1],
                    "method": method,
                }

            else:
                # 다른 방법들은 컴포넌트 수를 직접 지정해야 함
                return {
                    "optimal_components": min(max_components, data_scaled.shape[1]),
                    "method": method,
                    "note": f"{method}는 최적 컴포넌트 수를 자동으로 찾을 수 없습니다",
                }

        except Exception as e:
            self.logger.error(f"최적 컴포넌트 수 찾기 중 오류 발생: {e}")
            return {"error": str(e)}

    def compare_methods(
        self, df: UnifiedDataFrame, columns: Optional[List[str]] = None, n_components: int = 2
    ) -> Dict[str, DimensionalityReductionResult]:
        """여러 차원 축소 방법 비교

        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_components: 축소할 차원 수

        Returns:
            각 방법별 차원 축소 결과
        """
        results = {}

        # PCA
        results["pca"] = self.pca_reduction(df, columns, n_components)

        # t-SNE
        results["tsne"] = self.tsne_reduction(df, columns, n_components)

        # UMAP
        results["umap"] = self.umap_reduction(df, columns, n_components)

        # ICA
        results["ica"] = self.ica_reduction(df, columns, n_components)

        return results
