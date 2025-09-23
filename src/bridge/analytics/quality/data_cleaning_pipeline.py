"""데이터 정제 파이프라인.

CA 마일스톤 3.3: 데이터 품질 관리 시스템
- 자동화된 데이터 정제 및 변환 파이프라인
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from bridge.analytics.core.data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)


@dataclass
class CleaningStep:
    """데이터 정제 단계를 정의하는 클래스"""

    name: str
    function: Callable
    parameters: Dict[str, Any]
    enabled: bool = True


@dataclass
class CleaningResult:
    """데이터 정제 결과를 담는 데이터 클래스"""

    cleaned_data: UnifiedDataFrame
    cleaning_steps: List[Dict[str, Any]]
    quality_improvement: Dict[str, float]
    removed_rows: int
    removed_columns: int
    transformed_columns: List[str]


class DataCleaningPipeline:
    """데이터 정제 파이프라인 클래스"""

    def __init__(self):
        """데이터 정제 파이프라인 초기화"""
        self.logger = logging.getLogger(__name__)
        self.cleaning_steps: List[CleaningStep] = []
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def add_step(
        self, name: str, function: Callable, parameters: Dict[str, Any] = None, enabled: bool = True
    ):
        """정제 단계 추가

        Args:
            name: 단계 이름
            function: 정제 함수
            parameters: 함수 매개변수
            enabled: 활성화 여부
        """
        if parameters is None:
            parameters = {}

        step = CleaningStep(name=name, function=function, parameters=parameters, enabled=enabled)
        self.cleaning_steps.append(step)

    def remove_step(self, name: str):
        """정제 단계 제거

        Args:
            name: 제거할 단계 이름
        """
        self.cleaning_steps = [step for step in self.cleaning_steps if step.name != name]

    def enable_step(self, name: str):
        """정제 단계 활성화

        Args:
            name: 활성화할 단계 이름
        """
        for step in self.cleaning_steps:
            if step.name == name:
                step.enabled = True
                break

    def disable_step(self, name: str):
        """정제 단계 비활성화

        Args:
            name: 비활성화할 단계 이름
        """
        for step in self.cleaning_steps:
            if step.name == name:
                step.enabled = False
                break

    def clean_data(self, data: UnifiedDataFrame) -> CleaningResult:
        """데이터 정제 실행

        Args:
            data: 정제할 데이터

        Returns:
            CleaningResult: 정제 결과
        """
        try:
            current_data = data
            cleaning_steps = []
            original_quality = self._calculate_quality_score(data)

            original_rows = len(current_data.to_pandas())
            original_columns = len(current_data.to_pandas().columns)

            for step in self.cleaning_steps:
                if not step.enabled:
                    continue

                try:
                    # 정제 단계 실행
                    step_result = step.function(current_data, **step.parameters)

                    if isinstance(step_result, UnifiedDataFrame):
                        current_data = step_result
                    elif isinstance(step_result, dict) and "data" in step_result:
                        current_data = step_result["data"]
                    else:
                        self.logger.warning(f"단계 {step.name}의 결과가 올바르지 않습니다.")
                        continue

                    # 단계 결과 기록
                    step_info = {
                        "name": step.name,
                        "parameters": step.parameters,
                        "success": True,
                        "rows_before": original_rows,
                        "rows_after": len(current_data.to_pandas()),
                        "columns_before": original_columns,
                        "columns_after": len(current_data.to_pandas().columns),
                    }
                    cleaning_steps.append(step_info)

                    original_rows = len(current_data.to_pandas())
                    original_columns = len(current_data.to_pandas().columns)

                except Exception as e:
                    self.logger.error(f"정제 단계 {step.name} 실행 실패: {e}")
                    step_info = {
                        "name": step.name,
                        "parameters": step.parameters,
                        "success": False,
                        "error": str(e),
                    }
                    cleaning_steps.append(step_info)

            # 품질 개선 계산
            final_quality = self._calculate_quality_score(current_data)
            quality_improvement = {
                "original_quality": original_quality,
                "final_quality": final_quality,
                "improvement": final_quality - original_quality,
            }

            # 제거된 행/열 수 계산
            removed_rows = original_rows - len(current_data.to_pandas())
            removed_columns = original_columns - len(current_data.to_pandas().columns)

            return CleaningResult(
                cleaned_data=current_data,
                cleaning_steps=cleaning_steps,
                quality_improvement=quality_improvement,
                removed_rows=removed_rows,
                removed_columns=removed_columns,
                transformed_columns=[],
            )

        except Exception as e:
            self.logger.error(f"데이터 정제 실패: {e}")
            return CleaningResult(
                cleaned_data=data,
                cleaning_steps=[],
                quality_improvement={"original_quality": 0, "final_quality": 0, "improvement": 0},
                removed_rows=0,
                removed_columns=0,
                transformed_columns=[],
            )

    def _calculate_quality_score(self, data: UnifiedDataFrame) -> float:
        """데이터 품질 점수 계산

        Args:
            data: 분석할 데이터

        Returns:
            품질 점수 (0-1)
        """
        try:
            df = data.to_pandas()

            # 결측값 비율
            missing_ratio = df.isnull().sum().sum() / df.size

            # 중복 비율
            duplicate_ratio = df.duplicated().sum() / len(df)

            # 품질 점수 (1 - 결측값 비율 - 중복 비율)
            quality_score = 1 - missing_ratio - duplicate_ratio
            return max(0, min(1, quality_score))

        except Exception:
            return 0.0

    # 기본 정제 함수들
    @staticmethod
    def remove_duplicates(
        data: UnifiedDataFrame, subset: Optional[List[str]] = None, keep: str = "first"
    ) -> UnifiedDataFrame:
        """중복 행 제거

        Args:
            data: 정제할 데이터
            subset: 중복 검사할 컬럼 목록
            keep: 유지할 중복 행 ('first', 'last', False)

        Returns:
            UnifiedDataFrame: 중복 제거된 데이터
        """
        try:
            df = data.to_pandas()
            cleaned_df = df.drop_duplicates(subset=subset, keep=keep)
            return UnifiedDataFrame(cleaned_df)
        except Exception as e:
            logger.error(f"중복 제거 실패: {e}")
            return data

    @staticmethod
    def remove_missing_rows(data: UnifiedDataFrame, threshold: float = 0.5) -> UnifiedDataFrame:
        """결측값이 많은 행 제거

        Args:
            data: 정제할 데이터
            threshold: 결측값 임계값 (0-1)

        Returns:
            UnifiedDataFrame: 정제된 데이터
        """
        try:
            df = data.to_pandas()

            # 각 행의 결측값 비율 계산
            missing_ratio = df.isnull().sum(axis=1) / len(df.columns)

            # 임계값 이하인 행만 유지
            cleaned_df = df[missing_ratio <= threshold]

            return UnifiedDataFrame(cleaned_df)
        except Exception as e:
            logger.error(f"결측값 행 제거 실패: {e}")
            return data

    @staticmethod
    def remove_missing_columns(data: UnifiedDataFrame, threshold: float = 0.5) -> UnifiedDataFrame:
        """결측값이 많은 컬럼 제거

        Args:
            data: 정제할 데이터
            threshold: 결측값 임계값 (0-1)

        Returns:
            UnifiedDataFrame: 정제된 데이터
        """
        try:
            df = data.to_pandas()

            # 각 컬럼의 결측값 비율 계산
            missing_ratio = df.isnull().sum() / len(df)

            # 임계값 이하인 컬럼만 유지
            cleaned_df = df.loc[:, missing_ratio <= threshold]

            return UnifiedDataFrame(cleaned_df)
        except Exception as e:
            logger.error(f"결측값 컬럼 제거 실패: {e}")
            return data

    @staticmethod
    def impute_missing_values(
        data: UnifiedDataFrame, strategy: str = "mean", columns: Optional[List[str]] = None
    ) -> UnifiedDataFrame:
        """결측값 대체

        Args:
            data: 정제할 데이터
            strategy: 대체 전략 ('mean', 'median', 'mode', 'knn')
            columns: 대체할 컬럼 목록

        Returns:
            UnifiedDataFrame: 정제된 데이터
        """
        try:
            df = data.to_pandas()

            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            for col in columns:
                if col not in df.columns:
                    continue

                if strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == "mode":
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        df[col] = df[col].fillna(mode_value[0])
                elif strategy == "knn":
                    # KNN 대체
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        imputer = KNNImputer(n_neighbors=5)
                        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

            return UnifiedDataFrame(df)
        except Exception as e:
            logger.error(f"결측값 대체 실패: {e}")
            return data

    @staticmethod
    def remove_outliers(
        data: UnifiedDataFrame, method: str = "iqr", columns: Optional[List[str]] = None
    ) -> UnifiedDataFrame:
        """이상치 제거

        Args:
            data: 정제할 데이터
            method: 이상치 탐지 방법 ('iqr', 'zscore')
            columns: 이상치 탐지할 컬럼 목록

        Returns:
            UnifiedDataFrame: 정제된 데이터
        """
        try:
            df = data.to_pandas()

            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            for col in columns:
                if col not in df.columns:
                    continue

                if method == "iqr":
                    # IQR 방법
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                elif method == "zscore":
                    # Z-score 방법
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    df = df[z_scores < 3]

            return UnifiedDataFrame(df)
        except Exception as e:
            logger.error(f"이상치 제거 실패: {e}")
            return data

    @staticmethod
    def standardize_columns(
        data: UnifiedDataFrame, columns: Optional[List[str]] = None, method: str = "standard"
    ) -> UnifiedDataFrame:
        """컬럼 표준화

        Args:
            data: 정제할 데이터
            columns: 표준화할 컬럼 목록
            method: 표준화 방법 ('standard', 'minmax')

        Returns:
            UnifiedDataFrame: 정제된 데이터
        """
        try:
            df = data.to_pandas()

            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()

            for col in columns:
                if col not in df.columns:
                    continue

                if method == "standard":
                    # Z-score 표준화
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
                elif method == "minmax":
                    # Min-Max 표준화
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

            return UnifiedDataFrame(df)
        except Exception as e:
            logger.error(f"컬럼 표준화 실패: {e}")
            return data

    @staticmethod
    def encode_categorical(
        data: UnifiedDataFrame, columns: Optional[List[str]] = None, method: str = "label"
    ) -> UnifiedDataFrame:
        """범주형 변수 인코딩

        Args:
            data: 정제할 데이터
            columns: 인코딩할 컬럼 목록
            method: 인코딩 방법 ('label', 'onehot')

        Returns:
            UnifiedDataFrame: 정제된 데이터
        """
        try:
            df = data.to_pandas()

            if columns is None:
                columns = df.select_dtypes(include=["object"]).columns.tolist()

            if method == "label":
                # Label 인코딩
                for col in columns:
                    if col in df.columns:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
            elif method == "onehot":
                # One-hot 인코딩
                df = pd.get_dummies(df, columns=columns, prefix=columns)

            return UnifiedDataFrame(df)
        except Exception as e:
            logger.error(f"범주형 변수 인코딩 실패: {e}")
            return data

    def create_default_pipeline(self):
        """기본 정제 파이프라인 생성"""
        self.cleaning_steps = []

        # 1. 중복 행 제거
        self.add_step("remove_duplicates", self.remove_duplicates, {"keep": "first"})

        # 2. 결측값이 많은 행 제거
        self.add_step("remove_missing_rows", self.remove_missing_rows, {"threshold": 0.5})

        # 3. 결측값이 많은 컬럼 제거
        self.add_step("remove_missing_columns", self.remove_missing_columns, {"threshold": 0.5})

        # 4. 결측값 대체
        self.add_step("impute_missing", self.impute_missing_values, {"strategy": "mean"})

        # 5. 이상치 제거
        self.add_step("remove_outliers", self.remove_outliers, {"method": "iqr"})

        # 6. 범주형 변수 인코딩
        self.add_step("encode_categorical", self.encode_categorical, {"method": "label"})

        # 7. 컬럼 표준화
        self.add_step("standardize_columns", self.standardize_columns, {"method": "standard"})

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """파이프라인 요약 정보 반환

        Returns:
            파이프라인 요약 딕셔너리
        """
        return {
            "total_steps": len(self.cleaning_steps),
            "enabled_steps": len([step for step in self.cleaning_steps if step.enabled]),
            "steps": [
                {"name": step.name, "enabled": step.enabled, "parameters": step.parameters}
                for step in self.cleaning_steps
            ],
        }
