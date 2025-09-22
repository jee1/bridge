"""ML 모델 관리 시스템 테스트"""

import unittest
import tempfile
import shutil
from datetime import datetime
from pathlib import Path

from bridge.ml.models import ModelRegistry, ModelVersionManager, InferenceEngine
from bridge.governance.contracts import (
    ModelContract, 
    ModelType, 
    ModelStatus, 
    ModelMetrics,
    ModelInputSchema,
    ModelOutputSchema,
    ColumnSchema,
    DataType
)


class TestModelRegistry(unittest.TestCase):
    """모델 레지스트리 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(storage_path=self.temp_dir)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_register_model(self):
        """모델 등록 테스트"""
        # 테스트 모델 계약 생성
        input_schema = ModelInputSchema(
            features=[
                ColumnSchema(name="feature1", data_type=DataType.FLOAT),
                ColumnSchema(name="feature2", data_type=DataType.FLOAT)
            ]
        )
        
        output_schema = ModelOutputSchema(
            predictions=[
                ColumnSchema(name="prediction", data_type=DataType.FLOAT)
            ]
        )
        
        model = ModelContract(
            id="test_model_001",
            name="test_classifier",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            status=ModelStatus.READY,
            description="테스트 분류 모델",
            algorithm="RandomForest",
            framework="sklearn",
            input_schema=input_schema,
            output_schema=output_schema,
            performance_metrics=ModelMetrics(accuracy=0.95),
            created_by="test_user"
        )
        
        # 모델 등록
        result = self.registry.register_model(model)
        self.assertTrue(result)
        
        # 등록된 모델 조회
        retrieved_model = self.registry.get_model("test_model_001")
        self.assertIsNotNone(retrieved_model)
        self.assertEqual(retrieved_model.name, "test_classifier")
        self.assertEqual(retrieved_model.version, "1.0.0")
    
    def test_list_models(self):
        """모델 목록 조회 테스트"""
        # 여러 모델 등록
        for i in range(3):
            model = ModelContract(
                id=f"test_model_{i:03d}",
                name=f"test_model_{i}",
                version="1.0.0",
                model_type=ModelType.CLASSIFICATION,
                status=ModelStatus.READY,
                created_by="test_user"
            )
            self.registry.register_model(model)
        
        # 모든 모델 조회
        models = self.registry.list_models()
        self.assertEqual(len(models), 3)
        
        # 분류 모델만 조회
        classification_models = self.registry.list_models(model_type=ModelType.CLASSIFICATION)
        self.assertEqual(len(classification_models), 3)
    
    def test_update_model(self):
        """모델 업데이트 테스트"""
        # 모델 등록
        model = ModelContract(
            id="test_model_002",
            name="test_model",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            status=ModelStatus.READY,
            created_by="test_user"
        )
        self.registry.register_model(model)
        
        # 모델 업데이트
        updates = {
            'description': '업데이트된 설명',
            'status': ModelStatus.DEPLOYED,
            'tags': ['production', 'stable']
        }
        result = self.registry.update_model("test_model_002", updates)
        self.assertTrue(result)
        
        # 업데이트 확인
        updated_model = self.registry.get_model("test_model_002")
        self.assertEqual(updated_model.description, '업데이트된 설명')
        self.assertEqual(updated_model.status, ModelStatus.DEPLOYED)
        self.assertEqual(updated_model.tags, ['production', 'stable'])


class TestModelVersionManager(unittest.TestCase):
    """모델 버전 관리자 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(storage_path=self.temp_dir)
        self.version_manager = ModelVersionManager(self.registry)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_version(self):
        """버전 생성 테스트"""
        # 기본 모델 등록
        model = ModelContract(
            id="test_model_003",
            name="test_model",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            status=ModelStatus.READY,
            created_by="test_user"
        )
        self.registry.register_model(model)
        
        # 새 버전 생성
        new_version = self.version_manager.create_version("test_model_003")
        self.assertIsNotNone(new_version)
        self.assertEqual(new_version, "1.0.1")  # semantic versioning
        
        # 새 버전 모델 확인
        new_model = self.registry.get_model(f"test_model_003_v{new_version}")
        self.assertIsNotNone(new_model)
        self.assertEqual(new_model.version, new_version)
    
    def test_promote_to_production(self):
        """프로덕션 승격 테스트"""
        # 모델 등록
        model = ModelContract(
            id="test_model_004",
            name="test_model",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            status=ModelStatus.READY,
            created_by="test_user"
        )
        self.registry.register_model(model)
        
        # 프로덕션 승격
        result = self.version_manager.promote_to_production("test_model_004")
        self.assertTrue(result)
        
        # 상태 확인
        promoted_model = self.registry.get_model("test_model_004")
        self.assertEqual(promoted_model.status, ModelStatus.DEPLOYED)


class TestInferenceEngine(unittest.TestCase):
    """추론 엔진 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(storage_path=self.temp_dir)
        self.inference_engine = InferenceEngine(self.registry)
    
    def tearDown(self):
        """테스트 정리"""
        shutil.rmtree(self.temp_dir)
    
    def test_model_info(self):
        """모델 정보 조회 테스트"""
        # 모델 등록
        model = ModelContract(
            id="test_model_005",
            name="test_model",
            version="1.0.0",
            model_type=ModelType.CLASSIFICATION,
            status=ModelStatus.READY,
            algorithm="RandomForest",
            framework="sklearn",
            created_by="test_user"
        )
        self.registry.register_model(model)
        
        # 모델 정보 조회
        info = self.inference_engine.get_model_info("test_model_005")
        self.assertIsNotNone(info)
        self.assertEqual(info['name'], "test_model")
        self.assertEqual(info['model_type'], "classification")
        self.assertEqual(info['algorithm'], "RandomForest")
        self.assertFalse(info['is_loaded'])  # 아직 로딩되지 않음


if __name__ == '__main__':
    unittest.main()
