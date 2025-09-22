"""시계열 분석 알고리즘 모듈

ARIMA, Prophet, LSTM 등을 활용한 시계열 분석을 제공합니다.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from bridge.analytics.core.data_integration import UnifiedDataFrame


@dataclass
class TimeSeriesResult:
    """시계열 분석 결과를 담는 데이터 클래스"""
    
    model_type: str
    predictions: np.ndarray
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    model_metrics: Optional[Dict[str, float]] = None
    seasonal_components: Optional[Dict[str, np.ndarray]] = None
    trend: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    is_stationary: Optional[bool] = None
    adf_statistic: Optional[float] = None
    adf_pvalue: Optional[float] = None


@dataclass
class ForecastResult:
    """예측 결과를 담는 데이터 클래스"""
    
    forecast_values: np.ndarray
    forecast_dates: List[datetime]
    confidence_intervals: Optional[Tuple[np.ndarray, np.ndarray]] = None
    model_accuracy: Optional[Dict[str, float]] = None


class TimeSeriesAnalyzer:
    """시계열 분석을 수행하는 클래스"""
    
    def __init__(self):
        """시계열 분석기 초기화"""
        self.logger = __import__('logging').getLogger(__name__)
    
    def analyze_stationarity(
        self, 
        df: UnifiedDataFrame, 
        time_column: str, 
        value_column: str
    ) -> Dict[str, Any]:
        """시계열의 정상성 분석
        
        Args:
            df: 시계열 데이터
            time_column: 시간 컬럼명
            value_column: 값 컬럼명
            
        Returns:
            정상성 분석 결과
        """
        try:
            pandas_df = df.to_pandas()
            
            # 시간 컬럼을 datetime으로 변환
            pandas_df[time_column] = pd.to_datetime(pandas_df[time_column])
            pandas_df = pandas_df.sort_values(time_column)
            
            # ADF 테스트
            from statsmodels.tsa.stattools import adfuller
            
            adf_result = adfuller(pandas_df[value_column].dropna())
            
            return {
                'is_stationary': adf_result[1] < 0.05,  # p-value < 0.05이면 정상
                'adf_statistic': adf_result[0],
                'adf_pvalue': adf_result[1],
                'critical_values': adf_result[4],
                'series_length': len(pandas_df)
            }
            
        except Exception as e:
            self.logger.error(f"정상성 분석 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def decompose_time_series(
        self, 
        df: UnifiedDataFrame, 
        time_column: str, 
        value_column: str,
        model: str = 'additive'
    ) -> Dict[str, Any]:
        """시계열 분해
        
        Args:
            df: 시계열 데이터
            time_column: 시간 컬럼명
            value_column: 값 컬럼명
            model: 분해 모델 ('additive' 또는 'multiplicative')
            
        Returns:
            분해 결과
        """
        try:
            pandas_df = df.to_pandas()
            
            # 시간 컬럼을 datetime으로 변환하고 정렬
            pandas_df[time_column] = pd.to_datetime(pandas_df[time_column])
            pandas_df = pandas_df.sort_values(time_column)
            
            # 시계열 데이터 준비
            ts_data = pandas_df.set_index(time_column)[value_column]
            
            # 분해 수행
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(
                ts_data, 
                model=model, 
                extrapolate_trend='freq'
            )
            
            return {
                'trend': decomposition.trend.dropna().values,
                'seasonal': decomposition.seasonal.dropna().values,
                'residual': decomposition.resid.dropna().values,
                'observed': decomposition.observed.dropna().values,
                'model': model
            }
            
        except Exception as e:
            self.logger.error(f"시계열 분해 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def fit_arima(
        self, 
        df: UnifiedDataFrame, 
        time_column: str, 
        value_column: str,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ) -> TimeSeriesResult:
        """ARIMA 모델 피팅
        
        Args:
            df: 시계열 데이터
            time_column: 시간 컬럼명
            value_column: 값 컬럼명
            order: ARIMA (p, d, q) 파라미터
            seasonal_order: 계절성 (P, D, Q, s) 파라미터
            
        Returns:
            ARIMA 분석 결과
        """
        try:
            pandas_df = df.to_pandas()
            
            # 시간 컬럼을 datetime으로 변환하고 정렬
            pandas_df[time_column] = pd.to_datetime(pandas_df[time_column])
            pandas_df = pandas_df.sort_values(time_column)
            
            # 시계열 데이터 준비
            ts_data = pandas_df[value_column].dropna()
            
            # ARIMA 모델 피팅
            from statsmodels.tsa.arima.model import ARIMA
            
            model = ARIMA(ts_data, order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit()
            
            # 예측
            predictions = fitted_model.fittedvalues.values
            
            # 잔차
            residuals = fitted_model.resid.values
            
            # 모델 메트릭
            metrics = {
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'log_likelihood': fitted_model.llf,
                'mae': np.mean(np.abs(residuals)),
                'rmse': np.sqrt(np.mean(residuals**2))
            }
            
            return TimeSeriesResult(
                model_type='ARIMA',
                predictions=predictions,
                residuals=residuals,
                model_metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"ARIMA 모델 피팅 중 오류 발생: {e}")
            return TimeSeriesResult(
                model_type='ARIMA',
                predictions=np.array([]),
                model_metrics={'error': str(e)}
            )
    
    def fit_prophet(
        self, 
        df: UnifiedDataFrame, 
        time_column: str, 
        value_column: str,
        **prophet_params
    ) -> TimeSeriesResult:
        """Prophet 모델 피팅
        
        Args:
            df: 시계열 데이터
            time_column: 시간 컬럼명
            value_column: 값 컬럼명
            **prophet_params: Prophet 모델 파라미터
            
        Returns:
            Prophet 분석 결과
        """
        try:
            pandas_df = df.to_pandas()
            
            # Prophet 형식으로 데이터 준비
            prophet_df = pandas_df[[time_column, value_column]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Prophet 모델 피팅
            from prophet import Prophet
            
            model = Prophet(**prophet_params)
            model.fit(prophet_df)
            
            # 예측
            future = model.make_future_dataframe(periods=0)
            forecast = model.predict(future)
            
            predictions = forecast['yhat'].values
            trend = forecast['trend'].values
            
            # 계절성 컴포넌트
            seasonal_components = {}
            for component in ['yearly', 'monthly', 'weekly', 'daily']:
                if component in forecast.columns:
                    seasonal_components[component] = forecast[component].values
            
            # 잔차
            residuals = prophet_df['y'].values - predictions[:len(prophet_df)]
            
            # 모델 메트릭
            metrics = {
                'mae': np.mean(np.abs(residuals)),
                'rmse': np.sqrt(np.mean(residuals**2)),
                'mape': np.mean(np.abs(residuals / prophet_df['y'].values)) * 100
            }
            
            return TimeSeriesResult(
                model_type='Prophet',
                predictions=predictions,
                trend=trend,
                seasonal_components=seasonal_components,
                residuals=residuals,
                model_metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Prophet 모델 피팅 중 오류 발생: {e}")
            return TimeSeriesResult(
                model_type='Prophet',
                predictions=np.array([]),
                model_metrics={'error': str(e)}
            )
    
    def forecast_arima(
        self, 
        df: UnifiedDataFrame, 
        time_column: str, 
        value_column: str,
        periods: int = 30,
        order: Tuple[int, int, int] = (1, 1, 1)
    ) -> ForecastResult:
        """ARIMA 모델을 사용한 예측
        
        Args:
            df: 시계열 데이터
            time_column: 시간 컬럼명
            value_column: 값 컬럼명
            periods: 예측 기간
            order: ARIMA (p, d, q) 파라미터
            
        Returns:
            예측 결과
        """
        try:
            pandas_df = df.to_pandas()
            
            # 시간 컬럼을 datetime으로 변환하고 정렬
            pandas_df[time_column] = pd.to_datetime(pandas_df[time_column])
            pandas_df = pandas_df.sort_values(time_column)
            
            # 시계열 데이터 준비
            ts_data = pandas_df[value_column].dropna()
            
            # ARIMA 모델 피팅
            from statsmodels.tsa.arima.model import ARIMA
            
            model = ARIMA(ts_data, order=order)
            fitted_model = model.fit()
            
            # 예측
            forecast_result = fitted_model.forecast(steps=periods)
            conf_int = fitted_model.get_forecast(steps=periods).conf_int()
            
            # 예측 날짜 생성
            last_date = pandas_df[time_column].max()
            forecast_dates = [
                last_date + timedelta(days=i+1) 
                for i in range(periods)
            ]
            
            return ForecastResult(
                forecast_values=forecast_result.values,
                forecast_dates=forecast_dates,
                confidence_intervals=(conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values),
                model_accuracy={
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
            )
            
        except Exception as e:
            self.logger.error(f"ARIMA 예측 중 오류 발생: {e}")
            return ForecastResult(
                forecast_values=np.array([]),
                forecast_dates=[],
                model_accuracy={'error': str(e)}
            )
    
    def forecast_prophet(
        self, 
        df: UnifiedDataFrame, 
        time_column: str, 
        value_column: str,
        periods: int = 30,
        **prophet_params
    ) -> ForecastResult:
        """Prophet 모델을 사용한 예측
        
        Args:
            df: 시계열 데이터
            time_column: 시간 컬럼명
            value_column: 값 컬럼명
            periods: 예측 기간
            **prophet_params: Prophet 모델 파라미터
            
        Returns:
            예측 결과
        """
        try:
            pandas_df = df.to_pandas()
            
            # Prophet 형식으로 데이터 준비
            prophet_df = pandas_df[[time_column, value_column]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            
            # Prophet 모델 피팅
            from prophet import Prophet
            
            model = Prophet(**prophet_params)
            model.fit(prophet_df)
            
            # 미래 데이터프레임 생성
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # 예측 결과 추출
            forecast_values = forecast['yhat'].tail(periods).values
            forecast_dates = forecast['ds'].tail(periods).dt.to_pydatetime().tolist()
            
            # 신뢰구간
            conf_int = (
                forecast['yhat_lower'].tail(periods).values,
                forecast['yhat_upper'].tail(periods).values
            )
            
            return ForecastResult(
                forecast_values=forecast_values,
                forecast_dates=forecast_dates,
                confidence_intervals=conf_int
            )
            
        except Exception as e:
            self.logger.error(f"Prophet 예측 중 오류 발생: {e}")
            return ForecastResult(
                forecast_values=np.array([]),
                forecast_dates=[],
                model_accuracy={'error': str(e)}
            )
    
    def detect_trend(
        self, 
        df: UnifiedDataFrame, 
        time_column: str, 
        value_column: str
    ) -> Dict[str, Any]:
        """트렌드 탐지
        
        Args:
            df: 시계열 데이터
            time_column: 시간 컬럼명
            value_column: 값 컬럼명
            
        Returns:
            트렌드 분석 결과
        """
        try:
            pandas_df = df.to_pandas()
            
            # 시간 컬럼을 datetime으로 변환하고 정렬
            pandas_df[time_column] = pd.to_datetime(pandas_df[time_column])
            pandas_df = pandas_df.sort_values(time_column)
            
            # 시계열 데이터
            ts_data = pandas_df[value_column].dropna()
            
            # 선형 회귀를 통한 트렌드 분석
            from scipy import stats
            
            x = np.arange(len(ts_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, ts_data)
            
            # 트렌드 방향 결정
            if p_value < 0.05:  # 통계적으로 유의한 트렌드
                if slope > 0:
                    trend_direction = 'increasing'
                else:
                    trend_direction = 'decreasing'
            else:
                trend_direction = 'no_trend'
            
            return {
                'trend_direction': trend_direction,
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'trend_strength': abs(slope)
            }
            
        except Exception as e:
            self.logger.error(f"트렌드 탐지 중 오류 발생: {e}")
            return {'error': str(e)}
    
    def detect_seasonality(
        self, 
        df: UnifiedDataFrame, 
        time_column: str, 
        value_column: str
    ) -> Dict[str, Any]:
        """계절성 탐지
        
        Args:
            df: 시계열 데이터
            time_column: 시간 컬럼명
            value_column: 값 컬럼명
            
        Returns:
            계절성 분석 결과
        """
        try:
            pandas_df = df.to_pandas()
            
            # 시간 컬럼을 datetime으로 변환하고 정렬
            pandas_df[time_column] = pd.to_datetime(pandas_df[time_column])
            pandas_df = pandas_df.sort_values(time_column)
            
            # 시계열 데이터
            ts_data = pandas_df[value_column].dropna()
            
            # FFT를 통한 주기성 분석
            fft = np.fft.fft(ts_data)
            freqs = np.fft.fftfreq(len(ts_data))
            
            # 양의 주파수만 고려
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # 가장 강한 주파수 찾기
            dominant_freq_idx = np.argmax(positive_fft[1:]) + 1  # DC 성분 제외
            dominant_freq = positive_freqs[dominant_freq_idx]
            
            # 주기 계산
            if dominant_freq > 0:
                period = 1 / dominant_freq
            else:
                period = None
            
            return {
                'has_seasonality': dominant_freq > 0 and positive_fft[dominant_freq_idx] > np.mean(positive_fft),
                'dominant_frequency': dominant_freq,
                'period': period,
                'seasonal_strength': positive_fft[dominant_freq_idx] / np.mean(positive_fft) if np.mean(positive_fft) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"계절성 탐지 중 오류 발생: {e}")
            return {'error': str(e)}
