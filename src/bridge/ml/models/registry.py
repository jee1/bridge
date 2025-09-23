"""ML 모델 레지스트리 모듈

모델 등록, 조회, 삭제, 메타데이터 관리를 제공합니다.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bridge.governance.contracts import ModelContract, ModelStatus, ModelType

logger = logging.getLogger(__name__)


class ModelRegistry:
    """ML 모델 레지스트리"""

    def __init__(self, storage_path: str = "models"):
        """모델 레지스트리 초기화

        Args:
            storage_path: 모델 저장 경로
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 메모리 내 모델 캐시
        self._models: Dict[str, ModelContract] = {}
        self._load_models()

        self.logger = logging.getLogger(__name__)

    def _load_models(self) -> None:
        """저장된 모델들을 메모리로 로드"""
        try:
            registry_file = self.storage_path / "registry.json"
            if registry_file.exists():
                with open(registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for model_data in data.get("models", []):
                        model = ModelContract.from_dict(model_data)
                        self._models[model.id] = model
                self.logger.info(f"로드된 모델 수: {len(self._models)}")
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {e}")

    def _save_models(self) -> None:
        """모델들을 파일로 저장"""
        try:
            registry_file = self.storage_path / "registry.json"
            data = {
                "models": [model.to_dict() for model in self._models.values()],
                "last_updated": datetime.now().isoformat(),
            }
            with open(registry_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"모델 저장 중 오류 발생: {e}")

    def register_model(self, model: ModelContract) -> bool:
        """모델을 레지스트리에 등록

        Args:
            model: 등록할 모델 계약

        Returns:
            등록 성공 여부
        """
        try:
            # 모델 ID 중복 확인
            if model.id in self._models:
                self.logger.warning(f"모델 ID가 이미 존재합니다: {model.id}")
                return False

            # 모델 저장 경로 설정
            model_dir = self.storage_path / model.id
            model_dir.mkdir(exist_ok=True)
            model.model_path = str(model_dir / "model.pkl")

            # 모델 등록
            self._models[model.id] = model
            self._save_models()

            self.logger.info(f"모델 등록 완료: {model.id} ({model.name})")
            return True

        except Exception as e:
            self.logger.error(f"모델 등록 중 오류 발생: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[ModelContract]:
        """모델 조회

        Args:
            model_id: 모델 ID

        Returns:
            모델 계약 또는 None
        """
        return self._models.get(model_id)

    def list_models(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
    ) -> List[ModelContract]:
        """모델 목록 조회

        Args:
            model_type: 필터링할 모델 타입
            status: 필터링할 모델 상태
            tags: 필터링할 태그 목록

        Returns:
            필터링된 모델 목록
        """
        models = list(self._models.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if status:
            models = [m for m in models if m.status == status]

        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]

        return sorted(models, key=lambda x: x.updated_at, reverse=True)

    def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """모델 정보 업데이트

        Args:
            model_id: 모델 ID
            updates: 업데이트할 필드들

        Returns:
            업데이트 성공 여부
        """
        try:
            if model_id not in self._models:
                self.logger.warning(f"모델을 찾을 수 없습니다: {model_id}")
                return False

            model = self._models[model_id]

            # 업데이트 가능한 필드들만 업데이트
            updatable_fields = {
                "description",
                "status",
                "performance_metrics",
                "hyperparameters",
                "tags",
                "model_path",
                "dependencies",
            }

            for field, value in updates.items():
                if field in updatable_fields and hasattr(model, field):
                    setattr(model, field, value)

            model.updated_at = datetime.now()
            self._save_models()

            self.logger.info(f"모델 업데이트 완료: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"모델 업데이트 중 오류 발생: {e}")
            return False

    def delete_model(self, model_id: str) -> bool:
        """모델 삭제

        Args:
            model_id: 삭제할 모델 ID

        Returns:
            삭제 성공 여부
        """
        try:
            if model_id not in self._models:
                self.logger.warning(f"모델을 찾을 수 없습니다: {model_id}")
                return False

            model = self._models[model_id]

            # 모델 파일 삭제
            if model.model_path and os.path.exists(model.model_path):
                os.remove(model.model_path)

            # 모델 디렉토리 삭제
            model_dir = self.storage_path / model_id
            if model_dir.exists():
                import shutil

                shutil.rmtree(model_dir)

            # 레지스트리에서 제거
            del self._models[model_id]
            self._save_models()

            self.logger.info(f"모델 삭제 완료: {model_id}")
            return True

        except Exception as e:
            self.logger.error(f"모델 삭제 중 오류 발생: {e}")
            return False

    def get_model_by_name(
        self, name: str, version: Optional[str] = None
    ) -> Optional[ModelContract]:
        """이름으로 모델 조회

        Args:
            name: 모델 이름
            version: 모델 버전 (선택사항)

        Returns:
            모델 계약 또는 None
        """
        for model in self._models.values():
            if model.name == name:
                if version is None or model.version == version:
                    return model
        return None

    def get_latest_model(self, name: str) -> Optional[ModelContract]:
        """최신 버전 모델 조회

        Args:
            name: 모델 이름

        Returns:
            최신 버전 모델 계약 또는 None
        """
        models = [m for m in self._models.values() if m.name == name]
        if not models:
            return None

        return max(models, key=lambda x: x.version)

    def get_model_stats(self) -> Dict[str, Any]:
        """모델 통계 정보 조회

        Returns:
            모델 통계 정보
        """
        total_models = len(self._models)
        models_by_type = {}
        models_by_status = {}

        for model in self._models.values():
            # 타입별 통계
            model_type = model.model_type.value
            models_by_type[model_type] = models_by_type.get(model_type, 0) + 1

            # 상태별 통계
            status = model.status.value
            models_by_status[status] = models_by_status.get(status, 0) + 1

        return {
            "total_models": total_models,
            "models_by_type": models_by_type,
            "models_by_status": models_by_status,
            "last_updated": datetime.now().isoformat(),
        }
