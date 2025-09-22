"""이상 탐지 알고리즘 모듈

Isolation Forest, One-Class SVM, LSTM Autoencoder 등을 활용한 이상 탐지를 제공합니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from bridge.analytics.core.data_integration import UnifiedDataFrame


@dataclass
class AnomalyResult:
    """이상 탐지 결과를 담는 데이터 클래스"""
    
    anomaly_scores: np.ndarray
    is_anomaly: np.ndarray
    anomaly_indices: List[int]
    normal_indices: List[int]
    model_type: str
    threshold: float
    contamination: float
    model_metrics: Optional[Dict[str, float]] = None


@dataclass
class AnomalyStats:
    """이상 탐지 통계를 담는 데이터 클래스"""
    
    total_points: int
    anomaly_count: int
    anomaly_ratio: float
    normal_count: int
    normal_ratio: float
    avg_anomaly_score: float
    max_anomaly_score: float
    min_anomaly_score: float


class AnomalyDetector:
    """이상 탐지를 수행하는 클래스"""
    
    def __init__(self):
        """이상 탐지기 초기화"""
        self.logger = __import__('logging').getLogger(__name__)
    
    def detect_with_isolation_forest(
        self, 
        df: UnifiedDataFrame, 
        columns: Optional[List[str]] = None,
        contamination: float = 0.1,
        random_state: int = 42
    ) -> AnomalyResult:
        """Isolation Forest를 사용한 이상 탐지
        
        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            contamination: 이상치 비율 (0.0 ~ 0.5)
            random_state: 랜덤 시드
            
        Returns:
            이상 탐지 결과
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
            
            # 데이터 준비
            data = pandas_df[numeric_columns].fillna(0)
            
            # Isolation Forest 모델
            from sklearn.ensemble import IsolationForest
            
            model = IsolationForest(
                contamination=contamination,
                random_state=random_state,
                n_estimators=100
            )
            
            # 모델 피팅 및 예측
            anomaly_scores = model.decision_function(data)
            is_anomaly = model.predict(data)
            
            # 이상치 인덱스
            anomaly_indices = np.where(is_anomaly == -1)[0].tolist()
            normal_indices = np.where(is_anomaly == 1)[0].tolist()
            
            # 임계값 계산
            threshold = np.percentile(anomaly_scores, contamination * 100)
            
            # 모델 메트릭
            metrics = {
                'contamination': contamination,
                'n_estimators': 100,
                'anomaly_ratio': len(anomaly_indices) / len(data)
            }
            
            return AnomalyResult(
                anomaly_scores=anomaly_scores,
                is_anomaly=is_anomaly,
                anomaly_indices=anomaly_indices,
                normal_indices=normal_indices,
                model_type='IsolationForest',
                threshold=threshold,
                contamination=contamination,
                model_metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Isolation Forest 이상 탐지 중 오류 발생: {e}")
            return AnomalyResult(
                anomaly_scores=np.array([]),
                is_anomaly=np.array([]),
                anomaly_indices=[],
                normal_indices=[],
                model_type='IsolationForest',
                threshold=0.0,
                contamination=contamination,
                model_metrics={'error': str(e)}
            )
    
    def detect_with_one_class_svm(
        self, 
        df: UnifiedDataFrame, 
        columns: Optional[List[str]] = None,
        nu: float = 0.1,
        kernel: str = 'rbf',
        gamma: str = 'scale'
    ) -> AnomalyResult:
        """One-Class SVM을 사용한 이상 탐지
        
        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            nu: 이상치 비율 (0.0 ~ 1.0)
            kernel: SVM 커널
            gamma: RBF 커널의 gamma 파라미터
            
        Returns:
            이상 탐지 결과
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
            
            # One-Class SVM 모델
            from sklearn.svm import OneClassSVM
            
            model = OneClassSVM(
                nu=nu,
                kernel=kernel,
                gamma=gamma
            )
            
            # 모델 피팅 및 예측
            is_anomaly = model.fit_predict(data_scaled)
            anomaly_scores = model.decision_function(data_scaled)
            
            # 이상치 인덱스
            anomaly_indices = np.where(is_anomaly == -1)[0].tolist()
            normal_indices = np.where(is_anomaly == 1)[0].tolist()
            
            # 임계값 계산
            threshold = np.percentile(anomaly_scores, nu * 100)
            
            # 모델 메트릭
            metrics = {
                'nu': nu,
                'kernel': kernel,
                'gamma': gamma,
                'anomaly_ratio': len(anomaly_indices) / len(data)
            }
            
            return AnomalyResult(
                anomaly_scores=anomaly_scores,
                is_anomaly=is_anomaly,
                anomaly_indices=anomaly_indices,
                normal_indices=normal_indices,
                model_type='OneClassSVM',
                threshold=threshold,
                contamination=nu,
                model_metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"One-Class SVM 이상 탐지 중 오류 발생: {e}")
            return AnomalyResult(
                anomaly_scores=np.array([]),
                is_anomaly=np.array([]),
                anomaly_indices=[],
                normal_indices=[],
                model_type='OneClassSVM',
                threshold=0.0,
                contamination=nu,
                model_metrics={'error': str(e)}
            )
    
    def detect_with_local_outlier_factor(
        self, 
        df: UnifiedDataFrame, 
        columns: Optional[List[str]] = None,
        n_neighbors: int = 20,
        contamination: float = 0.1
    ) -> AnomalyResult:
        """Local Outlier Factor를 사용한 이상 탐지
        
        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            n_neighbors: 이웃 수
            contamination: 이상치 비율
            
        Returns:
            이상 탐지 결과
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
            
            # 데이터 준비
            data = pandas_df[numeric_columns].fillna(0)
            
            # Local Outlier Factor 모델
            from sklearn.neighbors import LocalOutlierFactor
            
            model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=False
            )
            
            # 모델 피팅 및 예측
            is_anomaly = model.fit_predict(data)
            anomaly_scores = model.negative_outlier_factor_
            
            # 이상치 인덱스
            anomaly_indices = np.where(is_anomaly == -1)[0].tolist()
            normal_indices = np.where(is_anomaly == 1)[0].tolist()
            
            # 임계값 계산
            threshold = np.percentile(anomaly_scores, contamination * 100)
            
            # 모델 메트릭
            metrics = {
                'n_neighbors': n_neighbors,
                'contamination': contamination,
                'anomaly_ratio': len(anomaly_indices) / len(data)
            }
            
            return AnomalyResult(
                anomaly_scores=anomaly_scores,
                is_anomaly=is_anomaly,
                anomaly_indices=anomaly_indices,
                normal_indices=normal_indices,
                model_type='LocalOutlierFactor',
                threshold=threshold,
                contamination=contamination,
                model_metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Local Outlier Factor 이상 탐지 중 오류 발생: {e}")
            return AnomalyResult(
                anomaly_scores=np.array([]),
                is_anomaly=np.array([]),
                anomaly_indices=[],
                normal_indices=[],
                model_type='LocalOutlierFactor',
                threshold=0.0,
                contamination=contamination,
                model_metrics={'error': str(e)}
            )
    
    def detect_with_elliptic_envelope(
        self, 
        df: UnifiedDataFrame, 
        columns: Optional[List[str]] = None,
        contamination: float = 0.1,
        random_state: int = 42
    ) -> AnomalyResult:
        """Elliptic Envelope를 사용한 이상 탐지
        
        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            contamination: 이상치 비율
            random_state: 랜덤 시드
            
        Returns:
            이상 탐지 결과
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
            
            # 데이터 준비
            data = pandas_df[numeric_columns].fillna(0)
            
            # Elliptic Envelope 모델
            from sklearn.covariance import EllipticEnvelope
            
            model = EllipticEnvelope(
                contamination=contamination,
                random_state=random_state
            )
            
            # 모델 피팅 및 예측
            is_anomaly = model.fit_predict(data)
            anomaly_scores = model.decision_function(data)
            
            # 이상치 인덱스
            anomaly_indices = np.where(is_anomaly == -1)[0].tolist()
            normal_indices = np.where(is_anomaly == 1)[0].tolist()
            
            # 임계값 계산
            threshold = np.percentile(anomaly_scores, contamination * 100)
            
            # 모델 메트릭
            metrics = {
                'contamination': contamination,
                'anomaly_ratio': len(anomaly_indices) / len(data)
            }
            
            return AnomalyResult(
                anomaly_scores=anomaly_scores,
                is_anomaly=is_anomaly,
                anomaly_indices=anomaly_indices,
                normal_indices=normal_indices,
                model_type='EllipticEnvelope',
                threshold=threshold,
                contamination=contamination,
                model_metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Elliptic Envelope 이상 탐지 중 오류 발생: {e}")
            return AnomalyResult(
                anomaly_scores=np.array([]),
                is_anomaly=np.array([]),
                anomaly_indices=[],
                normal_indices=[],
                model_type='EllipticEnvelope',
                threshold=0.0,
                contamination=contamination,
                model_metrics={'error': str(e)}
            )
    
    def detect_with_zscore(
        self, 
        df: UnifiedDataFrame, 
        columns: Optional[List[str]] = None,
        threshold: float = 3.0
    ) -> AnomalyResult:
        """Z-Score를 사용한 이상 탐지
        
        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            threshold: Z-Score 임계값
            
        Returns:
            이상 탐지 결과
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
            
            # 데이터 준비
            data = pandas_df[numeric_columns].fillna(0)
            
            # Z-Score 계산
            z_scores = np.abs((data - data.mean()) / data.std())
            max_z_scores = z_scores.max(axis=1)
            
            # 이상치 탐지
            is_anomaly = (max_z_scores > threshold).astype(int)
            is_anomaly[is_anomaly == 1] = -1  # 이상치를 -1로 표시
            is_anomaly[is_anomaly == 0] = 1   # 정상치를 1로 표시
            
            # 이상치 인덱스
            anomaly_indices = np.where(is_anomaly == -1)[0].tolist()
            normal_indices = np.where(is_anomaly == 1)[0].tolist()
            
            # 모델 메트릭
            metrics = {
                'threshold': threshold,
                'anomaly_ratio': len(anomaly_indices) / len(data),
                'max_z_score': max_z_scores.max(),
                'mean_z_score': max_z_scores.mean()
            }
            
            return AnomalyResult(
                anomaly_scores=max_z_scores,
                is_anomaly=is_anomaly,
                anomaly_indices=anomaly_indices,
                normal_indices=normal_indices,
                model_type='ZScore',
                threshold=threshold,
                contamination=len(anomaly_indices) / len(data),
                model_metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Z-Score 이상 탐지 중 오류 발생: {e}")
            return AnomalyResult(
                anomaly_scores=np.array([]),
                is_anomaly=np.array([]),
                anomaly_indices=[],
                normal_indices=[],
                model_type='ZScore',
                threshold=threshold,
                contamination=0.0,
                model_metrics={'error': str(e)}
            )
    
    def get_anomaly_stats(self, result: AnomalyResult) -> AnomalyStats:
        """이상 탐지 통계 계산
        
        Args:
            result: 이상 탐지 결과
            
        Returns:
            이상 탐지 통계
        """
        total_points = len(result.anomaly_scores)
        anomaly_count = len(result.anomaly_indices)
        normal_count = len(result.normal_indices)
        
        return AnomalyStats(
            total_points=total_points,
            anomaly_count=anomaly_count,
            anomaly_ratio=anomaly_count / total_points if total_points > 0 else 0,
            normal_count=normal_count,
            normal_ratio=normal_count / total_points if total_points > 0 else 0,
            avg_anomaly_score=np.mean(result.anomaly_scores) if len(result.anomaly_scores) > 0 else 0,
            max_anomaly_score=np.max(result.anomaly_scores) if len(result.anomaly_scores) > 0 else 0,
            min_anomaly_score=np.min(result.anomaly_scores) if len(result.anomaly_scores) > 0 else 0
        )
    
    def compare_methods(
        self, 
        df: UnifiedDataFrame, 
        columns: Optional[List[str]] = None,
        contamination: float = 0.1
    ) -> Dict[str, AnomalyResult]:
        """여러 이상 탐지 방법 비교
        
        Args:
            df: 입력 데이터
            columns: 분석할 컬럼 목록
            contamination: 이상치 비율
            
        Returns:
            각 방법별 이상 탐지 결과
        """
        results = {}
        
        # Isolation Forest
        results['isolation_forest'] = self.detect_with_isolation_forest(
            df, columns, contamination
        )
        
        # One-Class SVM
        results['one_class_svm'] = self.detect_with_one_class_svm(
            df, columns, contamination
        )
        
        # Local Outlier Factor
        results['local_outlier_factor'] = self.detect_with_local_outlier_factor(
            df, columns, contamination
        )
        
        # Elliptic Envelope
        results['elliptic_envelope'] = self.detect_with_elliptic_envelope(
            df, columns, contamination
        )
        
        # Z-Score
        results['z_score'] = self.detect_with_zscore(
            df, columns, threshold=3.0
        )
        
        return results
