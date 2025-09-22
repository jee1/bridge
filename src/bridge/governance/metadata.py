"""메타데이터 카탈로그 모듈

데이터 소스, 테이블, 컬럼 정보를 관리하고 데이터 계보를 추적합니다.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """데이터 소스 타입"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    ELASTICSEARCH = "elasticsearch"
    FILE = "file"
    API = "api"


class ColumnType(Enum):
    """컬럼 타입"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    JSON = "json"
    ARRAY = "array"
    BLOB = "blob"


@dataclass
class DataSource:
    """데이터 소스 정보"""
    id: str
    name: str
    type: DataSourceType
    connection_string: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Table:
    """테이블 정보"""
    id: str
    name: str
    schema_name: Optional[str] = None
    data_source_id: str = None
    description: Optional[str] = None
    row_count: Optional[int] = None
    size_bytes: Optional[int] = None
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class Column:
    """컬럼 정보"""
    id: str
    name: str
    table_id: str = None
    column_type: ColumnType = None
    nullable: bool = True
    primary_key: bool = False
    foreign_key: bool = False
    unique: bool = False
    default_value: Optional[Any] = None
    description: Optional[str] = None
    sample_values: List[Any] = None
    tags: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.sample_values is None:
            self.sample_values = []
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class DataLineage:
    """데이터 계보 정보"""
    id: str
    source_table_id: str
    target_table_id: str
    transformation_type: str  # join, filter, aggregate, etc.
    transformation_details: Dict[str, Any] = None
    created_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.transformation_details is None:
            self.transformation_details = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class MetadataCatalog:
    """메타데이터 카탈로그"""
    
    def __init__(self):
        self.data_sources: Dict[str, DataSource] = {}
        self.tables: Dict[str, Table] = {}
        self.columns: Dict[str, Column] = {}
        self.lineage: Dict[str, DataLineage] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_data_source(self, data_source: DataSource) -> bool:
        """데이터 소스 등록"""
        try:
            self.data_sources[data_source.id] = data_source
            self.logger.info(f"데이터 소스 등록 완료: {data_source.id} ({data_source.name})")
            return True
        except Exception as e:
            self.logger.error(f"데이터 소스 등록 실패: {e}")
            return False
    
    def register_table(self, table: Table) -> bool:
        """테이블 등록"""
        try:
            # 데이터 소스 존재 여부 확인
            if table.data_source_id not in self.data_sources:
                self.logger.error(f"데이터 소스가 존재하지 않습니다: {table.data_source_id}")
                return False
            
            self.tables[table.id] = table
            self.logger.info(f"테이블 등록 완료: {table.id} ({table.name})")
            return True
        except Exception as e:
            self.logger.error(f"테이블 등록 실패: {e}")
            return False
    
    def register_column(self, column: Column) -> bool:
        """컬럼 등록"""
        try:
            # 테이블 존재 여부 확인
            if column.table_id not in self.tables:
                self.logger.error(f"테이블이 존재하지 않습니다: {column.table_id}")
                return False
            
            self.columns[column.id] = column
            self.logger.info(f"컬럼 등록 완료: {column.id} ({column.name})")
            return True
        except Exception as e:
            self.logger.error(f"컬럼 등록 실패: {e}")
            return False
    
    def add_lineage(self, lineage: DataLineage) -> bool:
        """데이터 계보 추가"""
        try:
            # 소스 및 타겟 테이블 존재 여부 확인
            if lineage.source_table_id not in self.tables:
                self.logger.error(f"소스 테이블이 존재하지 않습니다: {lineage.source_table_id}")
                return False
            
            if lineage.target_table_id not in self.tables:
                self.logger.error(f"타겟 테이블이 존재하지 않습니다: {lineage.target_table_id}")
                return False
            
            self.lineage[lineage.id] = lineage
            self.logger.info(f"데이터 계보 추가 완료: {lineage.id}")
            return True
        except Exception as e:
            self.logger.error(f"데이터 계보 추가 실패: {e}")
            return False
    
    def get_data_source(self, source_id: str) -> Optional[DataSource]:
        """데이터 소스 조회"""
        return self.data_sources.get(source_id)
    
    def get_table(self, table_id: str) -> Optional[Table]:
        """테이블 조회"""
        return self.tables.get(table_id)
    
    def get_column(self, column_id: str) -> Optional[Column]:
        """컬럼 조회"""
        return self.columns.get(column_id)
    
    def get_tables_by_data_source(self, data_source_id: str) -> List[Table]:
        """데이터 소스별 테이블 목록"""
        return [table for table in self.tables.values() if table.data_source_id == data_source_id]
    
    def get_columns_by_table(self, table_id: str) -> List[Column]:
        """테이블별 컬럼 목록"""
        return [column for column in self.columns.values() if column.table_id == table_id]
    
    def search_tables(self, query: str) -> List[Table]:
        """테이블 검색"""
        results = []
        query_lower = query.lower()
        
        for table in self.tables.values():
            if (query_lower in table.name.lower() or 
                query_lower in (table.description or "").lower()):
                results.append(table)
        
        return results
    
    def search_columns(self, query: str) -> List[Column]:
        """컬럼 검색"""
        results = []
        query_lower = query.lower()
        
        for column in self.columns.values():
            if (query_lower in column.name.lower() or 
                query_lower in (column.description or "").lower()):
                results.append(column)
        
        return results
    
    def get_lineage_path(self, table_id: str) -> List[DataLineage]:
        """테이블의 데이터 계보 경로 조회"""
        # 해당 테이블을 소스로 하는 계보들
        source_lineages = [lineage for lineage in self.lineage.values() if lineage.source_table_id == table_id]
        
        # 해당 테이블을 타겟으로 하는 계보들
        target_lineages = [lineage for lineage in self.lineage.values() if lineage.target_table_id == table_id]
        
        return source_lineages + target_lineages
    
    def get_full_lineage_tree(self, table_id: str) -> Dict[str, Any]:
        """전체 데이터 계보 트리 조회"""
        def build_tree(current_table_id: str, visited: Set[str]) -> Dict[str, Any]:
            if current_table_id in visited:
                return {"table_id": current_table_id, "circular": True}
            
            visited.add(current_table_id)
            table = self.get_table(current_table_id)
            if not table:
                return {"table_id": current_table_id, "error": "Table not found"}
            
            # 이 테이블을 소스로 하는 계보들
            source_lineages = [lineage for lineage in self.lineage.values() if lineage.source_table_id == current_table_id]
            
            # 이 테이블을 타겟으로 하는 계보들
            target_lineages = [lineage for lineage in self.lineage.values() if lineage.target_table_id == current_table_id]
            
            result = {
                "table_id": current_table_id,
                "table_name": table.name,
                "data_source_id": table.data_source_id,
                "source_lineages": [],
                "target_lineages": []
            }
            
            # 소스 계보들 처리
            for lineage in source_lineages:
                target_tree = build_tree(lineage.target_table_id, visited.copy())
                result["source_lineages"].append({
                    "lineage_id": lineage.id,
                    "transformation_type": lineage.transformation_type,
                    "target": target_tree
                })
            
            # 타겟 계보들 처리
            for lineage in target_lineages:
                source_tree = build_tree(lineage.source_table_id, visited.copy())
                result["target_lineages"].append({
                    "lineage_id": lineage.id,
                    "transformation_type": lineage.transformation_type,
                    "source": source_tree
                })
            
            return result
        
        return build_tree(table_id, set())
    
    def export_metadata(self, file_path: str) -> bool:
        """메타데이터 내보내기"""
        try:
            data = {
                "data_sources": [asdict(ds) for ds in self.data_sources.values()],
                "tables": [asdict(t) for t in self.tables.values()],
                "columns": [asdict(c) for c in self.columns.values()],
                "lineage": [asdict(l) for l in self.lineage.values()],
                "exported_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"메타데이터 내보내기 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"메타데이터 내보내기 실패: {e}")
            return False
    
    def import_metadata(self, file_path: str) -> bool:
        """메타데이터 가져오기"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_count = 0
            
            # 데이터 소스 가져오기
            for ds_data in data.get("data_sources", []):
                ds_data['type'] = DataSourceType(ds_data['type'])
                data_source = DataSource(**ds_data)
                if self.register_data_source(data_source):
                    imported_count += 1
            
            # 테이블 가져오기
            for t_data in data.get("tables", []):
                table = Table(**t_data)
                if self.register_table(table):
                    imported_count += 1
            
            # 컬럼 가져오기
            for c_data in data.get("columns", []):
                c_data['column_type'] = ColumnType(c_data['column_type'])
                column = Column(**c_data)
                if self.register_column(column):
                    imported_count += 1
            
            # 계보 가져오기
            for l_data in data.get("lineage", []):
                lineage = DataLineage(**l_data)
                if self.add_lineage(lineage):
                    imported_count += 1
            
            self.logger.info(f"메타데이터 가져오기 완료: {imported_count}개 항목")
            return True
            
        except Exception as e:
            self.logger.error(f"메타데이터 가져오기 실패: {e}")
            return False


class SchemaRegistry:
    """스키마 레지스트리"""
    
    def __init__(self, catalog: MetadataCatalog):
        self.catalog = catalog
        self.logger = logging.getLogger(__name__)
    
    def get_table_schema(self, table_id: str) -> Optional[Dict[str, Any]]:
        """테이블 스키마 조회"""
        table = self.catalog.get_table(table_id)
        if not table:
            return None
        
        columns = self.catalog.get_columns_by_table(table_id)
        
        schema = {
            "table_id": table_id,
            "table_name": table.name,
            "schema_name": table.schema_name,
            "data_source_id": table.data_source_id,
            "columns": []
        }
        
        for column in columns:
            column_info = {
                "column_id": column.id,
                "name": column.name,
                "type": column.column_type.value if column.column_type else None,
                "nullable": column.nullable,
                "primary_key": column.primary_key,
                "foreign_key": column.foreign_key,
                "unique": column.unique,
                "default_value": column.default_value,
                "description": column.description,
                "sample_values": column.sample_values[:5]  # 처음 5개만
            }
            schema["columns"].append(column_info)
        
        return schema
    
    def compare_schemas(self, table_id1: str, table_id2: str) -> Dict[str, Any]:
        """두 테이블 스키마 비교"""
        schema1 = self.get_table_schema(table_id1)
        schema2 = self.get_table_schema(table_id2)
        
        if not schema1 or not schema2:
            return {"error": "테이블을 찾을 수 없습니다"}
        
        comparison = {
            "table1": schema1["table_name"],
            "table2": schema2["table_name"],
            "identical": True,
            "differences": []
        }
        
        # 컬럼 비교
        columns1 = {col["name"]: col for col in schema1["columns"]}
        columns2 = {col["name"]: col for col in schema2["columns"]}
        
        all_columns = set(columns1.keys()) | set(columns2.keys())
        
        for col_name in all_columns:
            if col_name not in columns1:
                comparison["differences"].append(f"컬럼 '{col_name}'이 테이블1에 없습니다")
                comparison["identical"] = False
            elif col_name not in columns2:
                comparison["differences"].append(f"컬럼 '{col_name}'이 테이블2에 없습니다")
                comparison["identical"] = False
            else:
                col1 = columns1[col_name]
                col2 = columns2[col_name]
                
                if col1["type"] != col2["type"]:
                    comparison["differences"].append(
                        f"컬럼 '{col_name}'의 타입이 다릅니다: {col1['type']} vs {col2['type']}"
                    )
                    comparison["identical"] = False
                
                if col1["nullable"] != col2["nullable"]:
                    comparison["differences"].append(
                        f"컬럼 '{col_name}'의 nullable 속성이 다릅니다: {col1['nullable']} vs {col2['nullable']}"
                    )
                    comparison["identical"] = False
        
        return comparison


class DataLineage:
    """데이터 계보 추적기"""
    
    def __init__(self, catalog: MetadataCatalog):
        self.catalog = catalog
        self.logger = logging.getLogger(__name__)
    
    def trace_data_flow(self, table_id: str, direction: str = "both") -> Dict[str, Any]:
        """데이터 흐름 추적"""
        if direction not in ["upstream", "downstream", "both"]:
            return {"error": "direction은 'upstream', 'downstream', 'both' 중 하나여야 합니다"}
        
        result = {
            "table_id": table_id,
            "direction": direction,
            "upstream": [],
            "downstream": []
        }
        
        if direction in ["upstream", "both"]:
            result["upstream"] = self._trace_upstream(table_id)
        
        if direction in ["downstream", "both"]:
            result["downstream"] = self._trace_downstream(table_id)
        
        return result
    
    def _trace_upstream(self, table_id: str) -> List[Dict[str, Any]]:
        """상위 데이터 흐름 추적"""
        upstream = []
        visited = set()
        
        def trace_recursive(current_table_id: str, depth: int = 0):
            if current_table_id in visited or depth > 10:  # 순환 참조 방지
                return
            
            visited.add(current_table_id)
            
            # 이 테이블을 타겟으로 하는 계보들
            target_lineages = [lineage for lineage in self.catalog.lineage.values() 
                             if lineage.target_table_id == current_table_id]
            
            for lineage in target_lineages:
                source_table = self.catalog.get_table(lineage.source_table_id)
                if source_table:
                    upstream_item = {
                        "table_id": source_table.id,
                        "table_name": source_table.name,
                        "transformation_type": lineage.transformation_type,
                        "depth": depth + 1
                    }
                    upstream.append(upstream_item)
                    
                    # 재귀적으로 상위 추적
                    trace_recursive(lineage.source_table_id, depth + 1)
        
        trace_recursive(table_id)
        return upstream
    
    def _trace_downstream(self, table_id: str) -> List[Dict[str, Any]]:
        """하위 데이터 흐름 추적"""
        downstream = []
        visited = set()
        
        def trace_recursive(current_table_id: str, depth: int = 0):
            if current_table_id in visited or depth > 10:  # 순환 참조 방지
                return
            
            visited.add(current_table_id)
            
            # 이 테이블을 소스로 하는 계보들
            source_lineages = [lineage for lineage in self.catalog.lineage.values() 
                             if lineage.source_table_id == current_table_id]
            
            for lineage in source_lineages:
                target_table = self.catalog.get_table(lineage.target_table_id)
                if target_table:
                    downstream_item = {
                        "table_id": target_table.id,
                        "table_name": target_table.name,
                        "transformation_type": lineage.transformation_type,
                        "depth": depth + 1
                    }
                    downstream.append(downstream_item)
                    
                    # 재귀적으로 하위 추적
                    trace_recursive(lineage.target_table_id, depth + 1)
        
        trace_recursive(table_id)
        return downstream
