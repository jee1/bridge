"""시계열 분석 모듈.

CA 마일스톤 3.2: 고급 통계 분석 및 시각화
시계열 분석 및 예측 기능을 제공하는 모듈입니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from bridge.analytics.core.data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)


class TimeSeriesAnalysis:
    """시계열 분석 클래스.

    시계열 분석 및 예측 기능을 제공합니다.
    """

    def __init__(self):
        """TimeSeriesAnalysis를 초기화합니다."""
        self.scaler = StandardScaler()

    def decompose_time_series(
        self,
        data: UnifiedDataFrame,
        time_column: str,
        value_column: str,
        model: str = "additive",
        period: Optional[int] = None,
    ) -> Dict[str, Any]:
        """시계열을 분해합니다.

        Args:
            data: 분석할 데이터
            time_column: 시간 컬럼
            value_column: 값 컬럼
            model: 분해 모델 ("additive", "multiplicative")
            period: 계절성 주기 (None이면 자동 감지)

        Returns:
            Dict[str, Any]: 시계열 분해 결과
        """
        df = data.to_pandas()

        # 시간 컬럼을 datetime으로 변환
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)

        # 시계열 데이터 준비
        ts_data = df.set_index(time_column)[value_column].dropna()

        if len(ts_data) < 2:
            return {"error": "시계열 분석을 위한 충분한 데이터가 없습니다."}

        # 계절성 주기 자동 감지
        if period is None:
            period = self._detect_seasonality(ts_data)

        # 시계열 분해
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            decomposition = seasonal_decompose(ts_data, model=model, period=period)

            # 분해 결과
            trend = decomposition.trend.dropna()
            seasonal = decomposition.seasonal.dropna()
            residual = decomposition.resid.dropna()

            # 통계 정보
            stats_info = {
                "trend_strength": self._calculate_trend_strength(ts_data, trend),
                "seasonal_strength": self._calculate_seasonal_strength(ts_data, seasonal),
                "residual_strength": self._calculate_residual_strength(ts_data, residual),
                "period": period,
                "model": model,
            }

            return {
                "decomposition": {
                    "original": ts_data.to_dict(),
                    "trend": trend.to_dict(),
                    "seasonal": seasonal.to_dict(),
                    "residual": residual.to_dict(),
                },
                "statistics": stats_info,
                "summary": {
                    "total_observations": len(ts_data),
                    "period": period,
                    "model": model,
                    "trend_direction": self._analyze_trend_direction(trend),
                    "seasonal_pattern": self._analyze_seasonal_pattern(seasonal),
                },
            }

        except ImportError:
            return {
                "error": "statsmodels가 설치되지 않았습니다. pip install statsmodels를 실행하세요."
            }

    def detect_trend(
        self, data: UnifiedDataFrame, time_column: str, value_column: str, method: str = "linear"
    ) -> Dict[str, Any]:
        """트렌드를 감지합니다.

        Args:
            data: 분석할 데이터
            time_column: 시간 컬럼
            value_column: 값 컬럼
            method: 감지 방법 ("linear", "polynomial", "moving_average")

        Returns:
            Dict[str, Any]: 트렌드 감지 결과
        """
        df = data.to_pandas()

        # 시간 컬럼을 datetime으로 변환
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)

        # 시계열 데이터 준비
        ts_data = df.set_index(time_column)[value_column].dropna()

        if len(ts_data) < 2:
            return {"error": "트렌드 분석을 위한 충분한 데이터가 없습니다."}

        # 시간 인덱스 생성
        time_index = np.arange(len(ts_data))

        if method == "linear":
            return self._linear_trend_analysis(ts_data, time_index)
        elif method == "polynomial":
            return self._polynomial_trend_analysis(ts_data, time_index)
        elif method == "moving_average":
            return self._moving_average_trend_analysis(ts_data)
        else:
            return {"error": f"지원하지 않는 트렌드 감지 방법: {method}"}

    def detect_seasonality(
        self, data: UnifiedDataFrame, time_column: str, value_column: str, max_period: int = 12
    ) -> Dict[str, Any]:
        """계절성을 감지합니다.

        Args:
            data: 분석할 데이터
            time_column: 시간 컬럼
            value_column: 값 컬럼
            max_period: 최대 주기

        Returns:
            Dict[str, Any]: 계절성 감지 결과
        """
        df = data.to_pandas()

        # 시간 컬럼을 datetime으로 변환
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)

        # 시계열 데이터 준비
        ts_data = df.set_index(time_column)[value_column].dropna()

        if len(ts_data) < 2:
            return {"error": "계절성 분석을 위한 충분한 데이터가 없습니다."}

        # 계절성 주기 감지
        period = self._detect_seasonality(ts_data, max_period)

        # 계절성 강도 계산 (간단한 구현)
        seasonal_strength = 0.0  # 기본값으로 설정

        # 계절성 패턴 분석
        seasonal_pattern = self._analyze_seasonal_pattern_detailed(ts_data, period)

        return {
            "detected_period": period,
            "seasonal_strength": seasonal_strength,
            "seasonal_pattern": seasonal_pattern,
            "is_seasonal": seasonal_strength > 0.1,
            "recommendations": self._get_seasonality_recommendations(period, seasonal_strength),
        }

    def forecast_time_series(
        self,
        data: UnifiedDataFrame,
        time_column: str,
        value_column: str,
        forecast_periods: int = 12,
        method: str = "linear",
    ) -> Dict[str, Any]:
        """시계열을 예측합니다.

        Args:
            data: 분석할 데이터
            time_column: 시간 컬럼
            value_column: 값 컬럼
            forecast_periods: 예측 기간
            method: 예측 방법 ("linear", "exponential", "arima")

        Returns:
            Dict[str, Any]: 시계열 예측 결과
        """
        df = data.to_pandas()

        # 시간 컬럼을 datetime으로 변환
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)

        # 시계열 데이터 준비
        ts_data = df.set_index(time_column)[value_column].dropna()

        if len(ts_data) < 2:
            return {"error": "예측을 위한 충분한 데이터가 없습니다."}

        if method == "linear":
            return self._linear_forecast(ts_data, forecast_periods)
        elif method == "exponential":
            return self._exponential_forecast(ts_data, forecast_periods)
        elif method == "arima":
            return self._arima_forecast(ts_data, forecast_periods)
        else:
            return {"error": f"지원하지 않는 예측 방법: {method}"}

    def analyze_anomalies(
        self,
        data: UnifiedDataFrame,
        time_column: str,
        value_column: str,
        method: str = "zscore",
        threshold: float = 3.0,
    ) -> Dict[str, Any]:
        """시계열 이상치를 분석합니다.

        Args:
            data: 분석할 데이터
            time_column: 시간 컬럼
            value_column: 값 컬럼
            method: 이상치 감지 방법 ("zscore", "iqr", "isolation_forest")
            threshold: 임계값

        Returns:
            Dict[str, Any]: 이상치 분석 결과
        """
        df = data.to_pandas()

        # 시간 컬럼을 datetime으로 변환
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)

        # 시계열 데이터 준비
        ts_data = df.set_index(time_column)[value_column].dropna()

        if len(ts_data) < 2:
            return {"error": "이상치 분석을 위한 충분한 데이터가 없습니다."}

        if method == "zscore":
            return self._zscore_anomaly_detection(ts_data, threshold)
        elif method == "iqr":
            return self._iqr_anomaly_detection(ts_data)
        elif method == "isolation_forest":
            return self._isolation_forest_anomaly_detection(ts_data)
        else:
            return {"error": f"지원하지 않는 이상치 감지 방법: {method}"}

    def _detect_seasonality(self, ts_data: pd.Series, max_period: int = 12) -> int:
        """계절성 주기를 감지합니다."""
        if len(ts_data) < 4:
            return 1

        # 자기상관함수를 사용한 계절성 감지
        from statsmodels.tsa.stattools import acf

        try:
            acf_values = acf(ts_data, nlags=min(max_period, len(ts_data) // 2))

            # 첫 번째 피크를 찾음
            peaks = []
            for i in range(1, len(acf_values) - 1):
                if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
                    peaks.append((i, acf_values[i]))

            if peaks:
                # 가장 높은 피크의 주기
                best_peak = max(peaks, key=lambda x: x[1])
                return best_peak[0]
            else:
                return 1
        except:
            return 1

    def _calculate_trend_strength(self, original: pd.Series, trend: pd.Series) -> float:
        """트렌드 강도를 계산합니다."""
        if len(original) == 0 or len(trend) == 0:
            return 0.0

        # 원본 데이터의 분산
        original_var = original.var()

        # 트렌드 제거 후의 분산
        detrended = original - trend
        detrended_var = detrended.var()

        # 트렌드 강도 = 1 - (트렌드 제거 후 분산 / 원본 분산)
        if original_var == 0:
            return 0.0

        return max(0, 1 - (detrended_var / original_var))

    def _calculate_seasonal_strength(self, original: pd.Series, seasonal: pd.Series) -> float:
        """계절성 강도를 계산합니다."""
        if len(original) == 0 or len(seasonal) == 0:
            return 0.0

        # 원본 데이터의 분산
        original_var = original.var()

        # 계절성 제거 후의 분산
        deseasonalized = original - seasonal
        deseasonalized_var = deseasonalized.var()

        # 계절성 강도 = 1 - (계절성 제거 후 분산 / 원본 분산)
        if original_var == 0:
            return 0.0

        return max(0, 1 - (deseasonalized_var / original_var))

    def _calculate_residual_strength(self, original: pd.Series, residual: pd.Series) -> float:
        """잔차 강도를 계산합니다."""
        if len(original) == 0 or len(residual) == 0:
            return 0.0

        # 원본 데이터의 분산
        original_var = original.var()

        # 잔차 분산
        residual_var = residual.var()

        # 잔차 강도 = 잔차 분산 / 원본 분산
        if original_var == 0:
            return 0.0

        return min(1, residual_var / original_var)

    def _analyze_trend_direction(self, trend: pd.Series) -> str:
        """트렌드 방향을 분석합니다."""
        if len(trend) < 2:
            return "분석 불가"

        # 선형 회귀를 사용한 트렌드 방향 분석
        x = np.arange(len(trend))
        y = trend.values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        if p_value < 0.05:  # 통계적으로 유의
            if slope > 0:
                return "상승 추세"
            else:
                return "하락 추세"
        else:
            return "추세 없음"

    def _analyze_seasonal_pattern(self, seasonal: pd.Series) -> str:
        """계절성 패턴을 분석합니다."""
        if len(seasonal) == 0:
            return "분석 불가"

        # 계절성의 변동성 분석
        seasonal_std = seasonal.std()
        seasonal_mean = abs(seasonal.mean())

        if seasonal_std > seasonal_mean * 0.1:
            return "강한 계절성"
        elif seasonal_std > seasonal_mean * 0.05:
            return "보통 계절성"
        else:
            return "약한 계절성"

    def _analyze_seasonal_pattern_detailed(self, ts_data: pd.Series, period: int) -> Dict[str, Any]:
        """상세한 계절성 패턴을 분석합니다."""
        if period <= 1:
            return {"pattern": "계절성 없음", "strength": 0.0}

        # 계절성 패턴 추출
        seasonal_values = []
        for i in range(period):
            seasonal_values.append(ts_data.iloc[i::period].mean())

        # 패턴 분석
        pattern_std = np.std(seasonal_values)
        pattern_mean = np.mean(np.abs(seasonal_values))

        if pattern_std > pattern_mean * 0.2:
            pattern_type = "강한 계절성"
        elif pattern_std > pattern_mean * 0.1:
            pattern_type = "보통 계절성"
        else:
            pattern_type = "약한 계절성"

        return {
            "pattern": pattern_type,
            "strength": pattern_std / pattern_mean if pattern_mean > 0 else 0,
            "seasonal_values": seasonal_values,
            "period": period,
        }

    def _linear_trend_analysis(self, ts_data: pd.Series, time_index: np.ndarray) -> Dict[str, Any]:
        """선형 트렌드 분석을 수행합니다."""
        y = ts_data.values

        # 선형 회귀
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, y)

        # 트렌드 라인
        trend_line = slope * time_index + intercept

        # 트렌드 강도
        trend_strength = r_value**2

        return {
            "method": "linear",
            "slope": slope,
            "intercept": intercept,
            "r_squared": trend_strength,
            "p_value": p_value,
            "trend_direction": "상승" if slope > 0 else "하락" if slope < 0 else "수평",
            "is_significant": p_value < 0.05,
            "trend_line": dict(zip(ts_data.index, trend_line)),
        }

    def _polynomial_trend_analysis(
        self, ts_data: pd.Series, time_index: np.ndarray
    ) -> Dict[str, Any]:
        """다항식 트렌드 분석을 수행합니다."""
        y = ts_data.values

        # 2차 다항식 피팅
        coeffs = np.polyfit(time_index, y, 2)
        poly_func = np.poly1d(coeffs)

        # 트렌드 라인
        trend_line = poly_func(time_index)

        # R² 계산
        ss_res = np.sum((y - trend_line) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "method": "polynomial",
            "coefficients": coeffs.tolist(),
            "r_squared": r_squared,
            "trend_line": dict(zip(ts_data.index, trend_line)),
        }

    def _moving_average_trend_analysis(self, ts_data: pd.Series) -> Dict[str, Any]:
        """이동평균 트렌드 분석을 수행합니다."""
        # 이동평균 계산
        window_size = min(7, len(ts_data) // 4)  # 적절한 윈도우 크기
        if window_size < 2:
            window_size = 2

        moving_avg = ts_data.rolling(window=window_size, center=True).mean()

        # 트렌드 방향 분석
        if len(moving_avg.dropna()) >= 2:
            first_half = moving_avg.iloc[: len(moving_avg) // 2].mean()
            second_half = moving_avg.iloc[len(moving_avg) // 2 :].mean()
            trend_direction = (
                "상승"
                if second_half > first_half
                else "하락" if second_half < first_half else "수평"
            )
        else:
            trend_direction = "분석 불가"

        return {
            "method": "moving_average",
            "window_size": window_size,
            "trend_direction": trend_direction,
            "trend_line": moving_avg.to_dict(),
        }

    def _linear_forecast(self, ts_data: pd.Series, forecast_periods: int) -> Dict[str, Any]:
        """선형 예측을 수행합니다."""
        time_index = np.arange(len(ts_data))
        y = ts_data.values

        # 선형 회귀
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, y)

        # 예측
        future_time_index = np.arange(len(ts_data), len(ts_data) + forecast_periods)
        forecast_values = slope * future_time_index + intercept

        # 예측 구간 (간단한 구현)
        forecast_std = std_err * np.sqrt(
            1
            + 1 / len(ts_data)
            + (future_time_index - np.mean(time_index)) ** 2
            / np.sum((time_index - np.mean(time_index)) ** 2)
        )

        return {
            "method": "linear",
            "forecast_periods": forecast_periods,
            "forecast_values": forecast_values.tolist(),
            "confidence_interval": {
                "lower": (forecast_values - 1.96 * forecast_std).tolist(),
                "upper": (forecast_values + 1.96 * forecast_std).tolist(),
            },
            "model_stats": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
            },
        }

    def _exponential_forecast(self, ts_data: pd.Series, forecast_periods: int) -> Dict[str, Any]:
        """지수 예측을 수행합니다."""
        # 로그 변환
        log_ts_data = np.log(ts_data + 1)  # 0 값 방지

        time_index = np.arange(len(log_ts_data))
        y = log_ts_data.values

        # 선형 회귀 (로그 공간에서)
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_index, y)

        # 예측 (로그 공간에서)
        future_time_index = np.arange(len(log_ts_data), len(log_ts_data) + forecast_periods)
        log_forecast_values = slope * future_time_index + intercept

        # 원래 공간으로 변환
        forecast_values = np.exp(log_forecast_values) - 1

        return {
            "method": "exponential",
            "forecast_periods": forecast_periods,
            "forecast_values": forecast_values.tolist(),
            "model_stats": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
            },
        }

    def _arima_forecast(self, ts_data: pd.Series, forecast_periods: int) -> Dict[str, Any]:
        """ARIMA 예측을 수행합니다."""
        try:
            from statsmodels.tsa.arima.model import ARIMA

            # 간단한 ARIMA(1,1,1) 모델
            model = ARIMA(ts_data, order=(1, 1, 1))
            fitted_model = model.fit()

            # 예측
            forecast = fitted_model.forecast(steps=forecast_periods)
            conf_int = fitted_model.get_forecast(steps=forecast_periods).conf_int()

            return {
                "method": "arima",
                "forecast_periods": forecast_periods,
                "forecast_values": forecast.tolist(),
                "confidence_interval": {
                    "lower": conf_int.iloc[:, 0].tolist(),
                    "upper": conf_int.iloc[:, 1].tolist(),
                },
                "model_summary": str(fitted_model.summary()),
            }
        except ImportError:
            return {
                "error": "statsmodels가 설치되지 않았습니다. pip install statsmodels를 실행하세요."
            }

    def _zscore_anomaly_detection(self, ts_data: pd.Series, threshold: float) -> Dict[str, Any]:
        """Z-score를 사용한 이상치 감지를 수행합니다."""
        z_scores = np.abs(stats.zscore(ts_data))
        anomalies = ts_data[z_scores > threshold]

        return {
            "method": "zscore",
            "threshold": threshold,
            "anomaly_count": len(anomalies),
            "anomaly_ratio": len(anomalies) / len(ts_data),
            "anomalies": anomalies.to_dict(),
            "z_scores": dict(zip(ts_data.index, z_scores)),
        }

    def _iqr_anomaly_detection(self, ts_data: pd.Series) -> Dict[str, Any]:
        """IQR을 사용한 이상치 감지를 수행합니다."""
        Q1 = ts_data.quantile(0.25)
        Q3 = ts_data.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        anomalies = ts_data[(ts_data < lower_bound) | (ts_data > upper_bound)]

        return {
            "method": "iqr",
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "anomaly_count": len(anomalies),
            "anomaly_ratio": len(anomalies) / len(ts_data),
            "anomalies": anomalies.to_dict(),
        }

    def _isolation_forest_anomaly_detection(self, ts_data: pd.Series) -> Dict[str, Any]:
        """Isolation Forest를 사용한 이상치 감지를 수행합니다."""
        try:
            from sklearn.ensemble import IsolationForest

            # 1차원 데이터를 2차원으로 변환
            X = ts_data.values.reshape(-1, 1)

            # Isolation Forest 모델
            model = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = model.fit_predict(X)

            # 이상치 추출
            anomalies = ts_data[anomaly_labels == -1]

            return {
                "method": "isolation_forest",
                "anomaly_count": len(anomalies),
                "anomaly_ratio": len(anomalies) / len(ts_data),
                "anomalies": anomalies.to_dict(),
            }
        except ImportError:
            return {
                "error": "scikit-learn이 설치되지 않았습니다. pip install scikit-learn을 실행하세요."
            }

    def _get_seasonality_recommendations(self, period: int, strength: float) -> List[str]:
        """계절성 분석에 대한 권장사항을 제공합니다."""
        recommendations = []

        if period > 1 and strength > 0.1:
            recommendations.append("강한 계절성이 감지되었습니다. 계절성 조정을 고려하세요.")
            recommendations.append(f"주기 {period}에 맞춰 데이터를 분석하세요.")

        if strength < 0.05:
            recommendations.append("계절성이 약합니다. 다른 요인을 고려해보세요.")

        return recommendations

    def __repr__(self) -> str:
        """문자열 표현."""
        return "TimeSeriesAnalysis()"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
