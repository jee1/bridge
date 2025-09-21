"""크로스 소스 조인 기능.

다중 데이터베이스 간 조인을 DuckDB를 사용하여 처리합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import duckdb
import pyarrow as pa

from .data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)


class CrossSourceJoiner:
    """다중 데이터베이스 간 조인 처리 클래스.
    
    DuckDB를 사용하여 다양한 데이터 소스 간의 조인을 처리합니다.
    """
    
    def __init__(self):
        """CrossSourceJoiner를 초기화합니다."""
        self.connection = duckdb.connect()
        self._registered_tables: Dict[str, str] = {}
    
    def register_table(self, name: str, unified_df: UnifiedDataFrame) -> None:
        """테이블을 DuckDB에 등록합니다.
        
        Args:
            name: 테이블 이름
            unified_df: 등록할 데이터
        """
        try:
            # Arrow Table을 DuckDB에 등록
            self.connection.register(name, unified_df.table)
            self._registered_tables[name] = name
            logger.info(f"테이블 '{name}' 등록 완료: {unified_df.num_rows}행, {unified_df.num_columns}열")
        except Exception as e:
            logger.error(f"테이블 등록 실패: {e}")
            raise
    
    def unregister_table(self, name: str) -> None:
        """테이블을 DuckDB에서 제거합니다.
        
        Args:
            name: 제거할 테이블 이름
        """
        try:
            if name in self._registered_tables:
                self.connection.unregister(name)
                del self._registered_tables[name]
                logger.info(f"테이블 '{name}' 제거 완료")
        except Exception as e:
            logger.error(f"테이블 제거 실패: {e}")
    
    def join_tables(self, 
                   left_table: str, 
                   right_table: str,
                   join_condition: str,
                   join_type: str = "inner") -> UnifiedDataFrame:
        """두 테이블을 조인합니다.
        
        Args:
            left_table: 왼쪽 테이블 이름
            right_table: 오른쪽 테이블 이름
            join_condition: 조인 조건 (SQL WHERE 절 형식)
            join_type: 조인 타입 (inner, left, right, full)
            
        Returns:
            UnifiedDataFrame: 조인 결과
        """
        try:
            # 조인 타입 검증
            valid_join_types = ["inner", "left", "right", "full"]
            if join_type.lower() not in valid_join_types:
                raise ValueError(f"지원하지 않는 조인 타입: {join_type}")
            
            # 테이블 존재 확인
            if left_table not in self._registered_tables:
                raise ValueError(f"테이블 '{left_table}'이 등록되지 않았습니다.")
            if right_table not in self._registered_tables:
                raise ValueError(f"테이블 '{right_table}'이 등록되지 않았습니다.")
            
            # 조인 쿼리 생성
            join_query = self._build_join_query(left_table, right_table, join_condition, join_type)
            
            logger.info(f"조인 쿼리 실행: {join_query}")
            
            # 쿼리 실행
            result = self.connection.execute(join_query).arrow()
            
            # UnifiedDataFrame으로 변환
            unified_result = UnifiedDataFrame(result)
            unified_result.add_metadata("join_type", join_type)
            unified_result.add_metadata("left_table", left_table)
            unified_result.add_metadata("right_table", right_table)
            unified_result.add_metadata("join_condition", join_condition)
            
            logger.info(f"조인 완료: {unified_result.num_rows}행, {unified_result.num_columns}열")
            return unified_result
            
        except Exception as e:
            logger.error(f"테이블 조인 실패: {e}")
            raise
    
    def multi_join(self, 
                  tables: List[str],
                  join_conditions: List[str],
                  join_types: Optional[List[str]] = None) -> UnifiedDataFrame:
        """다중 테이블을 조인합니다.
        
        Args:
            tables: 조인할 테이블 이름 목록
            join_conditions: 조인 조건 목록
            join_types: 조인 타입 목록 (선택사항)
            
        Returns:
            UnifiedDataFrame: 조인 결과
        """
        try:
            if len(tables) < 2:
                raise ValueError("최소 2개의 테이블이 필요합니다.")
            
            if len(join_conditions) != len(tables) - 1:
                raise ValueError("조인 조건의 개수는 테이블 개수 - 1이어야 합니다.")
            
            if join_types is None:
                join_types = ["inner"] * (len(tables) - 1)
            elif len(join_types) != len(tables) - 1:
                raise ValueError("조인 타입의 개수는 테이블 개수 - 1이어야 합니다.")
            
            # 첫 번째 테이블부터 순차적으로 조인
            result_table = tables[0]
            
            for i in range(1, len(tables)):
                next_table = tables[i]
                join_condition = join_conditions[i-1]
                join_type = join_types[i-1]
                
                # 임시 테이블 이름 생성
                temp_table = f"temp_join_{i}"
                
                # 조인 쿼리 생성
                join_query = self._build_join_query(result_table, next_table, join_condition, join_type)
                
                # 임시 테이블로 저장
                temp_query = f"CREATE TEMP TABLE {temp_table} AS {join_query}"
                self.connection.execute(temp_query)
                
                # 이전 임시 테이블 정리
                if i > 1:
                    prev_temp = f"temp_join_{i-1}"
                    self.connection.execute(f"DROP TABLE IF EXISTS {prev_temp}")
                
                result_table = temp_table
            
            # 최종 결과 조회
            final_result = self.connection.execute(f"SELECT * FROM {result_table}").arrow()
            
            # 임시 테이블 정리
            self.connection.execute(f"DROP TABLE IF EXISTS {result_table}")
            
            # UnifiedDataFrame으로 변환
            unified_result = UnifiedDataFrame(final_result)
            unified_result.add_metadata("multi_join", True)
            unified_result.add_metadata("tables", tables)
            unified_result.add_metadata("join_conditions", join_conditions)
            unified_result.add_metadata("join_types", join_types)
            
            logger.info(f"다중 조인 완료: {unified_result.num_rows}행, {unified_result.num_columns}열")
            return unified_result
            
        except Exception as e:
            logger.error(f"다중 조인 실패: {e}")
            raise
    
    def _build_join_query(self, 
                         left_table: str, 
                         right_table: str, 
                         join_condition: str, 
                         join_type: str) -> str:
        """조인 쿼리를 생성합니다.
        
        Args:
            left_table: 왼쪽 테이블 이름
            right_table: 오른쪽 테이블 이름
            join_condition: 조인 조건
            join_type: 조인 타입
            
        Returns:
            str: 생성된 조인 쿼리
        """
        join_type_upper = join_type.upper()
        
        if join_type_upper == "INNER":
            join_clause = "INNER JOIN"
        elif join_type_upper == "LEFT":
            join_clause = "LEFT JOIN"
        elif join_type_upper == "RIGHT":
            join_clause = "RIGHT JOIN"
        elif join_type_upper == "FULL":
            join_clause = "FULL OUTER JOIN"
        else:
            join_clause = "INNER JOIN"
        
        query = f"SELECT * FROM {left_table} {join_clause} {right_table} ON {join_condition}"
        
        return query
    
    def execute_custom_query(self, query: str) -> UnifiedDataFrame:
        """커스텀 SQL 쿼리를 실행합니다.
        
        Args:
            query: 실행할 SQL 쿼리
            
        Returns:
            UnifiedDataFrame: 쿼리 결과
        """
        try:
            logger.info(f"커스텀 쿼리 실행: {query[:100]}...")
            
            result = self.connection.execute(query).arrow()
            unified_result = UnifiedDataFrame(result)
            unified_result.add_metadata("custom_query", True)
            unified_result.add_metadata("query", query)
            
            logger.info(f"커스텀 쿼리 완료: {unified_result.num_rows}행, {unified_result.num_columns}열")
            return unified_result
            
        except Exception as e:
            logger.error(f"커스텀 쿼리 실행 실패: {e}")
            raise
    
    def get_registered_tables(self) -> List[str]:
        """등록된 테이블 목록을 반환합니다.
        
        Returns:
            List[str]: 등록된 테이블 이름 목록
        """
        return list(self._registered_tables.keys())
    
    def clear_tables(self) -> None:
        """모든 등록된 테이블을 제거합니다."""
        try:
            for table_name in list(self._registered_tables.keys()):
                self.unregister_table(table_name)
            logger.info("모든 테이블 제거 완료")
        except Exception as e:
            logger.error(f"테이블 정리 실패: {e}")
    
    def close(self) -> None:
        """DuckDB 연결을 종료합니다."""
        try:
            self.clear_tables()
            self.connection.close()
            logger.info("DuckDB 연결 종료")
        except Exception as e:
            logger.error(f"연결 종료 실패: {e}")
    
    def __enter__(self):
        """컨텍스트 매니저 진입."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료."""
        self.close()
    
    def __repr__(self) -> str:
        """문자열 표현."""
        return f"CrossSourceJoiner(tables={len(self._registered_tables)})"
