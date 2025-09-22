"""ML 모델 추론 엔진 모듈

실시간 추론, 모델 로딩, 캐싱을 제공합니다.
"""

import logging
import pickle
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd

from bridge.analytics.core.data_integration import UnifiedDataFrame
from bridge.governance.contracts import ModelContract, ModelStatus

logger = logging.getLogger(__name__)


class ModelCache:
    """모델 캐시 관리자"""
    
    def __init__(self, max_size: int = 10, ttl_seconds: int = 3600):
        """모델 캐시 초기화
        
        Args:
            max_size: 최대 캐시 크기
            ttl_seconds: 캐시 TTL (초)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def get(self, model_id: str) -> Optional[Any]:
        """캐시에서 모델 조회
        
        Args:
            model_id: 모델 ID
            
        Returns:
            캐시된 모델 또는 None
        """
        with self._lock:
            if model_id not in self._cache:
                return None
            
            # TTL 확인
            if self._is_expired(model_id):
                self._remove(model_id)
                return None
            
            # 접근 시간 업데이트
            self._access_times[model_id] = datetime.now()
            return self._cache[model_id]['model']
    
    def put(self, model_id: str, model: Any, metadata: Dict[str, Any] = None) -> None:
        """모델을 캐시에 저장
        
        Args:
            model_id: 모델 ID
            model: 저장할 모델
            metadata: 모델 메타데이터
        """
        with self._lock:
            # 캐시 크기 확인 및 정리
            if len(self._cache) >= self.max_size and model_id not in self._cache:
                self._evict_lru()
            
            self._cache[model_id] = {
                'model': model,
                'metadata': metadata or {},
                'loaded_at': datetime.now()
            }
            self._access_times[model_id] = datetime.now()
    
    def remove(self, model_id: str) -> None:
        """캐시에서 모델 제거
        
        Args:
            model_id: 모델 ID
        """
        with self._lock:
            self._remove(model_id)
    
    def _remove(self, model_id: str) -> None:
        """내부 모델 제거 메서드"""
        self._cache.pop(model_id, None)
        self._access_times.pop(model_id, None)
    
    def _is_expired(self, model_id: str) -> bool:
        """모델이 만료되었는지 확인"""
        if model_id not in self._cache:
            return True
        
        loaded_at = self._cache[model_id]['loaded_at']
        return datetime.now() - loaded_at > timedelta(seconds=self.ttl_seconds)
    
    def _evict_lru(self) -> None:
        """LRU 방식으로 모델 제거"""
        if not self._access_times:
            return
        
        # 가장 오래된 접근 시간을 가진 모델 제거
        lru_model_id = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        self._remove(lru_model_id)
    
    def clear(self) -> None:
        """캐시 전체 삭제"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        with self._lock:
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'cached_models': list(self._cache.keys()),
                'ttl_seconds': self.ttl_seconds
            }


class InferenceEngine:
    """ML 모델 추론 엔진"""
    
    def __init__(self, registry, cache_size: int = 10, cache_ttl: int = 3600):
        """추론 엔진 초기화
        
        Args:
            registry: 모델 레지스트리 인스턴스
            cache_size: 모델 캐시 크기
            cache_ttl: 캐시 TTL (초)
        """
        self.registry = registry
        self.cache = ModelCache(max_size=cache_size, ttl_seconds=cache_ttl)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_id: str) -> bool:
        """모델 로딩
        
        Args:
            model_id: 모델 ID
            
        Returns:
            로딩 성공 여부
        """
        try:
            # 캐시에서 먼저 확인
            if self.cache.get(model_id):
                self.logger.debug(f"모델이 이미 캐시에 있습니다: {model_id}")
                return True
            
            # 레지스트리에서 모델 정보 조회
            model_contract = self.registry.get_model(model_id)
            if not model_contract:
                self.logger.error(f"모델을 찾을 수 없습니다: {model_id}")
                return False
            
            if model_contract.status != ModelStatus.READY and model_contract.status != ModelStatus.DEPLOYED:
                self.logger.error(f"모델이 준비되지 않았습니다: {model_id}, 상태: {model_contract.status}")
                return False
            
            # 모델 파일 로딩
            if not model_contract.model_path or not Path(model_contract.model_path).exists():
                self.logger.error(f"모델 파일을 찾을 수 없습니다: {model_contract.model_path}")
                return False
            
            with open(model_contract.model_path, 'rb') as f:
                model = pickle.load(f)
            
            # 캐시에 저장
            metadata = {
                'model_type': model_contract.model_type.value,
                'framework': model_contract.framework,
                'version': model_contract.version,
                'algorithm': model_contract.algorithm
            }
            self.cache.put(model_id, model, metadata)
            
            self.logger.info(f"모델 로딩 완료: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"모델 로딩 중 오류 발생: {e}")
            return False
    
    def predict(
        self, 
        model_id: str, 
        data: Union[UnifiedDataFrame, pd.DataFrame, np.ndarray, List[Dict[str, Any]]],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """모델 예측 수행
        
        Args:
            model_id: 모델 ID
            data: 입력 데이터
            **kwargs: 추가 파라미터
            
        Returns:
            예측 결과 또는 None
        """
        try:
            # 모델 로딩 확인
            model = self.cache.get(model_id)
            if not model:
                if not self.load_model(model_id):
                    return None
                model = self.cache.get(model_id)
            
            if not model:
                self.logger.error(f"모델을 로딩할 수 없습니다: {model_id}")
                return None
            
            # 데이터 전처리
            processed_data = self._preprocess_data(data, model_id)
            if processed_data is None:
                return None
            
            # 예측 수행
            start_time = time.time()
            
            if hasattr(model, 'predict'):
                predictions = model.predict(processed_data)
            elif hasattr(model, 'transform'):
                predictions = model.transform(processed_data)
            else:
                self.logger.error(f"모델에 predict 또는 transform 메서드가 없습니다: {model_id}")
                return None
            
            prediction_time = time.time() - start_time
            
            # 결과 후처리
            result = self._postprocess_predictions(predictions, model_id, **kwargs)
            result['prediction_time'] = prediction_time
            result['model_id'] = model_id
            
            self.logger.debug(f"예측 완료: {model_id}, 소요시간: {prediction_time:.3f}초")
            return result
            
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {e}")
            return None
    
    def _preprocess_data(self, data: Union[UnifiedDataFrame, pd.DataFrame, np.ndarray, List[Dict[str, Any]]], model_id: str) -> Optional[np.ndarray]:
        """데이터 전처리
        
        Args:
            data: 입력 데이터
            model_id: 모델 ID
            
        Returns:
            전처리된 데이터 또는 None
        """
        try:
            # UnifiedDataFrame을 pandas DataFrame으로 변환
            if isinstance(data, UnifiedDataFrame):
                df = data.to_pandas()
            elif isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                self.logger.error(f"지원하지 않는 데이터 타입: {type(data)}")
                return None
            
            # 모델 계약에서 입력 스키마 확인
            model_contract = self.registry.get_model(model_id)
            if model_contract and model_contract.input_schema:
                # 필요한 피처만 선택
                feature_names = [col.name for col in model_contract.input_schema.features]
                if feature_names:
                    df = df[feature_names]
            
            # NaN 값 처리
            df = df.fillna(0)
            
            return df.values
            
        except Exception as e:
            self.logger.error(f"데이터 전처리 중 오류 발생: {e}")
            return None
    
    def _postprocess_predictions(self, predictions: np.ndarray, model_id: str, **kwargs) -> Dict[str, Any]:
        """예측 결과 후처리
        
        Args:
            predictions: 원시 예측 결과
            model_id: 모델 ID
            
        Returns:
            후처리된 예측 결과
        """
        try:
            result = {
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                'prediction_count': len(predictions) if hasattr(predictions, '__len__') else 1
            }
            
            # 확률 예측이 가능한 경우
            model = self.cache.get(model_id)
            if model and hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(predictions)
                    result['probabilities'] = probabilities.tolist() if isinstance(probabilities, np.ndarray) else probabilities
                except:
                    pass
            
            # 신뢰도 점수 계산 (간단한 방법)
            if 'probabilities' in result:
                max_probs = np.max(result['probabilities'], axis=1)
                result['confidence_scores'] = max_probs.tolist()
            
            return result
            
        except Exception as e:
            self.logger.error(f"예측 결과 후처리 중 오류 발생: {e}")
            return {'predictions': predictions, 'error': str(e)}
    
    def batch_predict(
        self, 
        model_id: str, 
        data_batch: List[Union[UnifiedDataFrame, pd.DataFrame, np.ndarray, List[Dict[str, Any]]]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """배치 예측 수행
        
        Args:
            model_id: 모델 ID
            data_batch: 배치 데이터
            **kwargs: 추가 파라미터
            
        Returns:
            배치 예측 결과 목록
        """
        results = []
        for i, data in enumerate(data_batch):
            result = self.predict(model_id, data, **kwargs)
            if result:
                result['batch_index'] = i
            results.append(result)
        return results
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """모델 정보 조회
        
        Args:
            model_id: 모델 ID
            
        Returns:
            모델 정보 또는 None
        """
        model_contract = self.registry.get_model(model_id)
        if not model_contract:
            return None
        
        cache_info = self.cache.get_stats()
        is_loaded = model_id in cache_info['cached_models']
        
        return {
            'model_id': model_id,
            'name': model_contract.name,
            'version': model_contract.version,
            'model_type': model_contract.model_type.value,
            'status': model_contract.status.value,
            'framework': model_contract.framework,
            'algorithm': model_contract.algorithm,
            'is_loaded': is_loaded,
            'performance_metrics': model_contract.performance_metrics.to_dict() if model_contract.performance_metrics else None,
            'created_at': model_contract.created_at.isoformat(),
            'updated_at': model_contract.updated_at.isoformat()
        }
    
    def unload_model(self, model_id: str) -> bool:
        """모델 언로딩
        
        Args:
            model_id: 모델 ID
            
        Returns:
            언로딩 성공 여부
        """
        try:
            self.cache.remove(model_id)
            self.logger.info(f"모델 언로딩 완료: {model_id}")
            return True
        except Exception as e:
            self.logger.error(f"모델 언로딩 중 오류 발생: {e}")
            return False
    
    def clear_cache(self) -> None:
        """캐시 전체 삭제"""
        self.cache.clear()
        self.logger.info("모델 캐시 삭제 완료")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 조회"""
        return self.cache.get_stats()
