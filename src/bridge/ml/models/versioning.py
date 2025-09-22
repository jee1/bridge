"""ML 모델 버전 관리 모듈

모델 버전 관리, A/B 테스트, 롤백 기능을 제공합니다.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from bridge.governance.contracts import ModelContract, ModelStatus

logger = logging.getLogger(__name__)


class VersionStrategy(Enum):
    """버전 관리 전략"""
    SEMANTIC = "semantic"  # semantic versioning (major.minor.patch)
    TIMESTAMP = "timestamp"  # 타임스탬프 기반
    INCREMENTAL = "incremental"  # 순차 증가


@dataclass
class ModelVersion:
    """모델 버전 정보"""
    version: str
    model_id: str
    created_at: datetime
    description: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    is_stable: bool = False
    is_production: bool = False
    parent_version: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ABTestConfig:
    """A/B 테스트 설정"""
    test_id: str
    model_a_id: str
    model_b_id: str
    traffic_split: float  # 0.0 ~ 1.0, A 모델에 할당할 트래픽 비율
    start_date: datetime
    end_date: Optional[datetime] = None
    success_metric: str = "accuracy"
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    is_active: bool = True


@dataclass
class ABTestResult:
    """A/B 테스트 결과"""
    test_id: str
    model_a_metrics: Dict[str, float]
    model_b_metrics: Dict[str, float]
    statistical_significance: bool
    p_value: float
    confidence_interval: Tuple[float, float]
    winner: Optional[str] = None  # 'A', 'B', or None (무승부)
    recommendation: Optional[str] = None


class ModelVersionManager:
    """모델 버전 관리자"""
    
    def __init__(self, registry):
        """버전 관리자 초기화
        
        Args:
            registry: 모델 레지스트리 인스턴스
        """
        self.registry = registry
        self.version_strategy = VersionStrategy.SEMANTIC
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.ab_results: Dict[str, ABTestResult] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_version(
        self, 
        model_id: str, 
        description: Optional[str] = None,
        strategy: Optional[VersionStrategy] = None
    ) -> Optional[str]:
        """새로운 모델 버전 생성
        
        Args:
            model_id: 모델 ID
            description: 버전 설명
            strategy: 버전 관리 전략
            
        Returns:
            생성된 버전 문자열 또는 None
        """
        try:
            model = self.registry.get_model(model_id)
            if not model:
                self.logger.error(f"모델을 찾을 수 없습니다: {model_id}")
                return None
            
            strategy = strategy or self.version_strategy
            new_version = self._generate_version(model.version, strategy)
            
            # 새 버전으로 모델 복사
            new_model = ModelContract(
                id=f"{model_id}_v{new_version}",
                name=model.name,
                version=new_version,
                model_type=model.model_type,
                status=ModelStatus.READY,
                description=description or f"Version {new_version}",
                algorithm=model.algorithm,
                framework=model.framework,
                input_schema=model.input_schema,
                output_schema=model.output_schema,
                performance_metrics=model.performance_metrics,
                training_data_info=model.training_data_info,
                hyperparameters=model.hyperparameters,
                created_by=model.created_by,
                tags=model.tags.copy(),
                dependencies=model.dependencies.copy()
            )
            
            # 새 모델 등록
            if self.registry.register_model(new_model):
                self.logger.info(f"새 버전 생성 완료: {model.name} v{new_version}")
                return new_version
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"버전 생성 중 오류 발생: {e}")
            return None
    
    def _generate_version(self, current_version: str, strategy: VersionStrategy) -> str:
        """버전 문자열 생성
        
        Args:
            current_version: 현재 버전
            strategy: 버전 관리 전략
            
        Returns:
            새 버전 문자열
        """
        if strategy == VersionStrategy.SEMANTIC:
            return self._generate_semantic_version(current_version)
        elif strategy == VersionStrategy.TIMESTAMP:
            return datetime.now().strftime("%Y%m%d_%H%M%S")
        elif strategy == VersionStrategy.INCREMENTAL:
            return self._generate_incremental_version(current_version)
        else:
            return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _generate_semantic_version(self, current_version: str) -> str:
        """Semantic versioning으로 새 버전 생성"""
        try:
            parts = current_version.split('.')
            if len(parts) == 3:
                major, minor, patch = map(int, parts)
                return f"{major}.{minor}.{patch + 1}"
            else:
                return "1.0.0"
        except:
            return "1.0.0"
    
    def _generate_incremental_version(self, current_version: str) -> str:
        """순차 증가 버전 생성"""
        try:
            version_num = int(current_version)
            return str(version_num + 1)
        except:
            return "1"
    
    def promote_to_production(self, model_id: str) -> bool:
        """모델을 프로덕션으로 승격
        
        Args:
            model_id: 모델 ID
            
        Returns:
            승격 성공 여부
        """
        try:
            model = self.registry.get_model(model_id)
            if not model:
                self.logger.error(f"모델을 찾을 수 없습니다: {model_id}")
                return False
            
            # 현재 프로덕션 모델들을 준비 상태로 변경
            production_models = [
                m for m in self.registry.list_models() 
                if m.name == model.name and m.status == ModelStatus.DEPLOYED
            ]
            
            for prod_model in production_models:
                self.registry.update_model(prod_model.id, {'status': ModelStatus.READY})
            
            # 새 모델을 프로덕션으로 승격
            success = self.registry.update_model(model_id, {'status': ModelStatus.DEPLOYED})
            
            if success:
                self.logger.info(f"모델 프로덕션 승격 완료: {model.name} v{model.version}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"프로덕션 승격 중 오류 발생: {e}")
            return False
    
    def rollback_to_version(self, model_name: str, target_version: str) -> bool:
        """특정 버전으로 롤백
        
        Args:
            model_name: 모델 이름
            target_version: 롤백할 버전
            
        Returns:
            롤백 성공 여부
        """
        try:
            # 대상 모델 찾기
            target_model = self.registry.get_model_by_name(model_name, target_version)
            if not target_model:
                self.logger.error(f"대상 모델을 찾을 수 없습니다: {model_name} v{target_version}")
                return False
            
            # 현재 프로덕션 모델들을 준비 상태로 변경
            current_production = self.registry.get_latest_model(model_name)
            if current_production and current_production.status == ModelStatus.DEPLOYED:
                self.registry.update_model(current_production.id, {'status': ModelStatus.READY})
            
            # 대상 모델을 프로덕션으로 승격
            success = self.registry.update_model(target_model.id, {'status': ModelStatus.DEPLOYED})
            
            if success:
                self.logger.info(f"롤백 완료: {model_name} v{target_version}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"롤백 중 오류 발생: {e}")
            return False
    
    def create_ab_test(self, config: ABTestConfig) -> bool:
        """A/B 테스트 생성
        
        Args:
            config: A/B 테스트 설정
            
        Returns:
            생성 성공 여부
        """
        try:
            # 모델 존재 확인
            model_a = self.registry.get_model(config.model_a_id)
            model_b = self.registry.get_model(config.model_b_id)
            
            if not model_a or not model_b:
                self.logger.error("A/B 테스트 모델 중 하나 이상을 찾을 수 없습니다")
                return False
            
            # 트래픽 분할 비율 검증
            if not (0.0 <= config.traffic_split <= 1.0):
                self.logger.error("트래픽 분할 비율은 0.0과 1.0 사이여야 합니다")
                return False
            
            self.ab_tests[config.test_id] = config
            self.logger.info(f"A/B 테스트 생성 완료: {config.test_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"A/B 테스트 생성 중 오류 발생: {e}")
            return False
    
    def get_ab_test_config(self, test_id: str) -> Optional[ABTestConfig]:
        """A/B 테스트 설정 조회
        
        Args:
            test_id: 테스트 ID
            
        Returns:
            A/B 테스트 설정 또는 None
        """
        return self.ab_tests.get(test_id)
    
    def list_ab_tests(self, active_only: bool = True) -> List[ABTestConfig]:
        """A/B 테스트 목록 조회
        
        Args:
            active_only: 활성 테스트만 조회할지 여부
            
        Returns:
            A/B 테스트 목록
        """
        tests = list(self.ab_tests.values())
        if active_only:
            tests = [t for t in tests if t.is_active]
        return tests
    
    def get_version_history(self, model_name: str) -> List[ModelVersion]:
        """모델 버전 히스토리 조회
        
        Args:
            model_name: 모델 이름
            
        Returns:
            버전 히스토리 목록
        """
        models = [m for m in self.registry.list_models() if m.name == model_name]
        versions = []
        
        for model in models:
            version = ModelVersion(
                version=model.version,
                model_id=model.id,
                created_at=model.created_at,
                description=model.description,
                performance_metrics=model.performance_metrics.to_dict() if model.performance_metrics else None,
                is_stable=model.status == ModelStatus.DEPLOYED,
                is_production=model.status == ModelStatus.DEPLOYED,
                tags=model.tags
            )
            versions.append(version)
        
        return sorted(versions, key=lambda x: x.created_at, reverse=True)
