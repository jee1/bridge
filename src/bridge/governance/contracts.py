"""데이터 계약 관리 모듈

데이터 스키마, 품질 규칙, 변환 규칙을 정의하고 관리합니다.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DataType(Enum):
    """지원하는 데이터 타입"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    JSON = "json"
    ARRAY = "array"


class QualityRuleType(Enum):
    """품질 규칙 타입"""
    NOT_NULL = "not_null"
    UNIQUE = "unique"
    RANGE = "range"
    PATTERN = "pattern"
    CUSTOM = "custom"


@dataclass
class ColumnSchema:
    """컬럼 스키마 정의"""
    name: str
    data_type: DataType
    nullable: bool = True
    description: Optional[str] = None
    default_value: Optional[Any] = None
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class QualityRule:
    """품질 규칙 정의"""
    name: str
    rule_type: QualityRuleType
    column: str
    parameters: Dict[str, Any]
    severity: str = "error"  # error, warning, info
    description: Optional[str] = None


@dataclass
class TransformationRule:
    """데이터 변환 규칙 정의"""
    name: str
    source_column: str
    target_column: str
    transformation_type: str  # map, format, calculate, etc.
    parameters: Dict[str, Any]
    description: Optional[str] = None


@dataclass
class DataContract:
    """데이터 계약 정의"""
    id: str
    name: str
    version: str
    description: Optional[str] = None
    source_system: Optional[str] = None
    target_system: Optional[str] = None
    schema: List[ColumnSchema] = None
    quality_rules: List[QualityRule] = None
    transformation_rules: List[TransformationRule] = None
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.schema is None:
            self.schema = []
        if self.quality_rules is None:
            self.quality_rules = []
        if self.transformation_rules is None:
            self.transformation_rules = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        # datetime 객체를 문자열로 변환
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataContract':
        """딕셔너리에서 생성"""
        # datetime 문자열을 datetime 객체로 변환
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # schema를 ColumnSchema 객체로 변환
        if 'schema' in data and data['schema']:
            schema_data = data['schema']
            data['schema'] = [
                ColumnSchema(
                    name=col['name'],
                    data_type=DataType(col['data_type']) if isinstance(col['data_type'], str) else col['data_type'],
                    nullable=col.get('nullable', True),
                    description=col.get('description'),
                    default_value=col.get('default_value'),
                    constraints=col.get('constraints')
                )
                for col in schema_data
            ]
        
        # quality_rules를 QualityRule 객체로 변환
        if 'quality_rules' in data and data['quality_rules']:
            rules_data = data['quality_rules']
            data['quality_rules'] = [
                QualityRule(
                    name=rule['name'],
                    rule_type=QualityRuleType(rule['rule_type']) if isinstance(rule['rule_type'], str) else rule['rule_type'],
                    column=rule['column'],
                    parameters=rule['parameters'],
                    severity=rule.get('severity', 'error'),
                    description=rule.get('description')
                )
                for rule in rules_data
            ]
        
        # transformation_rules를 TransformationRule 객체로 변환
        if 'transformation_rules' in data and data['transformation_rules']:
            trans_data = data['transformation_rules']
            data['transformation_rules'] = [
                TransformationRule(
                    name=rule['name'],
                    source_column=rule['source_column'],
                    target_column=rule['target_column'],
                    transformation_type=rule['transformation_type'],
                    parameters=rule['parameters'],
                    description=rule.get('description')
                )
                for rule in trans_data
            ]
        
        return cls(**data)

    def add_column(self, column: ColumnSchema) -> None:
        """컬럼 추가"""
        self.schema.append(column)
        self.updated_at = datetime.now()

    def add_quality_rule(self, rule: QualityRule) -> None:
        """품질 규칙 추가"""
        self.quality_rules.append(rule)
        self.updated_at = datetime.now()

    def add_transformation_rule(self, rule: TransformationRule) -> None:
        """변환 규칙 추가"""
        self.transformation_rules.append(rule)
        self.updated_at = datetime.now()

    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """컬럼 조회"""
        for column in self.schema:
            if column.name == name:
                return column
        return None

    def validate_schema(self) -> List[str]:
        """스키마 유효성 검사"""
        errors = []
        
        # 컬럼명 중복 검사
        column_names = [col.name for col in self.schema]
        if len(column_names) != len(set(column_names)):
            errors.append("중복된 컬럼명이 있습니다")
        
        # 품질 규칙의 컬럼 존재 여부 검사
        for rule in self.quality_rules:
            if not any(col.name == rule.column for col in self.schema):
                errors.append(f"품질 규칙 '{rule.name}'의 컬럼 '{rule.column}'이 스키마에 없습니다")
        
        # 변환 규칙의 컬럼 존재 여부 검사
        for rule in self.transformation_rules:
            if not any(col.name == rule.source_column for col in self.schema):
                errors.append(f"변환 규칙 '{rule.name}'의 소스 컬럼 '{rule.source_column}'이 스키마에 없습니다")
        
        return errors


class ContractValidator:
    """데이터 계약 검증기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_contract(self, contract: DataContract) -> Dict[str, Any]:
        """계약 유효성 검사"""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "schema_errors": []
        }
        
        # 스키마 유효성 검사
        schema_errors = contract.validate_schema()
        if schema_errors:
            result["schema_errors"] = schema_errors
            result["valid"] = False
        
        # 계약 기본 정보 검사
        if not contract.id:
            result["errors"].append("계약 ID가 필요합니다")
            result["valid"] = False
        
        if not contract.name:
            result["errors"].append("계약 이름이 필요합니다")
            result["valid"] = False
        
        if not contract.version:
            result["errors"].append("계약 버전이 필요합니다")
            result["valid"] = False
        
        # 품질 규칙 검사
        for rule in contract.quality_rules:
            if not rule.name:
                result["warnings"].append("품질 규칙에 이름이 없습니다")
            
            if not rule.column:
                result["errors"].append(f"품질 규칙 '{rule.name}'에 컬럼이 지정되지 않았습니다")
                result["valid"] = False
        
        return result
    
    def validate_data_against_contract(self, data: Dict[str, Any], contract: DataContract) -> Dict[str, Any]:
        """데이터가 계약을 준수하는지 검사"""
        result = {
            "compliant": True,
            "violations": [],
            "warnings": []
        }
        
        # 스키마 준수 검사
        for column in contract.schema:
            if column.name not in data:
                if not column.nullable:
                    result["violations"].append(f"필수 컬럼 '{column.name}'이 없습니다")
                    result["compliant"] = False
            else:
                # 데이터 타입 검사
                value = data[column.name]
                if not self._validate_data_type(value, column.data_type):
                    result["violations"].append(
                        f"컬럼 '{column.name}'의 데이터 타입이 올바르지 않습니다. "
                        f"예상: {column.data_type.value}, 실제: {type(value).__name__}"
                    )
                    result["compliant"] = False
        
        # 품질 규칙 검사
        for rule in contract.quality_rules:
            if rule.column in data:
                violation = self._check_quality_rule(data[rule.column], rule)
                if violation:
                    if rule.severity == "error":
                        result["violations"].append(violation)
                        result["compliant"] = False
                    else:
                        result["warnings"].append(violation)
        
        return result
    
    def _validate_data_type(self, value: Any, expected_type: DataType) -> bool:
        """데이터 타입 검사"""
        if value is None:
            return True  # None은 nullable 컬럼에서 허용
        
        type_mapping = {
            DataType.STRING: str,
            DataType.INTEGER: int,
            DataType.FLOAT: (int, float),
            DataType.BOOLEAN: bool,
            DataType.DATE: str,  # 날짜는 문자열로 처리
            DataType.DATETIME: str,  # 날짜시간은 문자열로 처리
            DataType.JSON: (dict, list),
            DataType.ARRAY: list
        }
        
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True
        
        if isinstance(expected_python_type, tuple):
            return isinstance(value, expected_python_type)
        else:
            return isinstance(value, expected_python_type)
    
    def _check_quality_rule(self, value: Any, rule: QualityRule) -> Optional[str]:
        """품질 규칙 검사"""
        if value is None:
            return None  # None 값은 별도 처리
        
        if rule.rule_type == QualityRuleType.NOT_NULL:
            if value is None:
                return f"컬럼 '{rule.column}'은 null이 될 수 없습니다"
        
        elif rule.rule_type == QualityRuleType.RANGE:
            min_val = rule.parameters.get('min')
            max_val = rule.parameters.get('max')
            if min_val is not None and value < min_val:
                return f"컬럼 '{rule.column}'의 값이 최소값 {min_val}보다 작습니다"
            if max_val is not None and value > max_val:
                return f"컬럼 '{rule.column}'의 값이 최대값 {max_val}보다 큽니다"
        
        elif rule.rule_type == QualityRuleType.PATTERN:
            pattern = rule.parameters.get('pattern')
            if pattern and not re.match(pattern, str(value)):
                return f"컬럼 '{rule.column}'의 값이 패턴 '{pattern}'과 일치하지 않습니다"
        
        return None


class ContractRegistry:
    """데이터 계약 레지스트리"""
    
    def __init__(self):
        self.contracts: Dict[str, DataContract] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_contract(self, contract: DataContract) -> bool:
        """계약 등록"""
        try:
            # 유효성 검사
            validator = ContractValidator()
            validation_result = validator.validate_contract(contract)
            
            if not validation_result["valid"]:
                self.logger.error(f"계약 등록 실패: {validation_result['errors']}")
                return False
            
            # 등록
            self.contracts[contract.id] = contract
            self.logger.info(f"계약 등록 완료: {contract.id} ({contract.name})")
            return True
            
        except Exception as e:
            self.logger.error(f"계약 등록 중 오류 발생: {e}")
            return False
    
    def get_contract(self, contract_id: str) -> Optional[DataContract]:
        """계약 조회"""
        return self.contracts.get(contract_id)
    
    def list_contracts(self, source_system: Optional[str] = None) -> List[DataContract]:
        """계약 목록 조회"""
        contracts = list(self.contracts.values())
        
        if source_system:
            contracts = [c for c in contracts if c.source_system == source_system]
        
        return sorted(contracts, key=lambda x: x.updated_at, reverse=True)
    
    def update_contract(self, contract_id: str, updates: Dict[str, Any]) -> bool:
        """계약 업데이트"""
        contract = self.get_contract(contract_id)
        if not contract:
            return False
        
        try:
            # 업데이트 적용
            for key, value in updates.items():
                if hasattr(contract, key):
                    setattr(contract, key, value)
            
            contract.updated_at = datetime.now()
            
            # 유효성 재검사
            validator = ContractValidator()
            validation_result = validator.validate_contract(contract)
            
            if not validation_result["valid"]:
                self.logger.error(f"계약 업데이트 실패: {validation_result['errors']}")
                return False
            
            self.logger.info(f"계약 업데이트 완료: {contract_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"계약 업데이트 중 오류 발생: {e}")
            return False
    
    def delete_contract(self, contract_id: str) -> bool:
        """계약 삭제"""
        if contract_id in self.contracts:
            del self.contracts[contract_id]
            self.logger.info(f"계약 삭제 완료: {contract_id}")
            return True
        return False
    
    def search_contracts(self, query: str) -> List[DataContract]:
        """계약 검색"""
        results = []
        query_lower = query.lower()
        
        for contract in self.contracts.values():
            if (query_lower in contract.name.lower() or 
                query_lower in (contract.description or "").lower() or
                query_lower in (contract.source_system or "").lower()):
                results.append(contract)
        
        return results
    
    def export_contracts(self, file_path: str) -> bool:
        """계약 내보내기"""
        try:
            data = {
                "contracts": [contract.to_dict() for contract in self.contracts.values()],
                "exported_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"계약 내보내기 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"계약 내보내기 실패: {e}")
            return False
    
    def import_contracts(self, file_path: str) -> bool:
        """계약 가져오기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            for contract_data in data.get("contracts", []):
                contract = DataContract.from_dict(contract_data)
                if self.register_contract(contract):
                    imported_count += 1
            
            self.logger.info(f"계약 가져오기 완료: {imported_count}개 계약")
            return True
            
        except Exception as e:
            self.logger.error(f"계약 가져오기 실패: {e}")
            return False
