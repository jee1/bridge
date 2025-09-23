"""통합 데이터 분석 레이어 - 스트리밍 처리 모듈.

CA 마일스톤 3.1: 통합 데이터 분석 레이어
대용량 데이터 처리를 위한 스트리밍 처리 모듈입니다.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, Iterator, List, Optional, Union, cast

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from bridge.analytics.core.data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)


class StreamingProcessor:
    """스트리밍 데이터 처리 클래스.

    대용량 데이터를 청크 단위로 처리하여 메모리 효율성을 높입니다.
    """

    def __init__(self, chunk_size: int = 10000, memory_limit_mb: int = 1000):
        """StreamingProcessor를 초기화합니다.

        Args:
            chunk_size: 청크 크기 (행 수)
            memory_limit_mb: 메모리 제한 (MB)
        """
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024

    def process_large_table(self, table: pa.Table, processor_func: callable, **kwargs) -> pa.Table:
        """대용량 테이블을 청크 단위로 처리합니다.

        Args:
            table: 처리할 테이블
            processor_func: 처리 함수
            **kwargs: 처리 함수에 전달할 추가 인자

        Returns:
            pa.Table: 처리된 테이블
        """
        logger.info(f"대용량 테이블 처리 시작: {len(table)}행, 청크 크기: {self.chunk_size}")

        # 메모리 사용량 확인
        table_size = table.nbytes
        if table_size > self.memory_limit_bytes:
            logger.info(f"테이블 크기가 메모리 제한을 초과합니다. 스트리밍 처리 모드로 전환합니다.")
            return self._process_in_chunks(table, processor_func, **kwargs)
        else:
            # 메모리 제한 내에서 처리
            return processor_func(table, **kwargs)

    def _process_in_chunks(self, table: pa.Table, processor_func: callable, **kwargs) -> pa.Table:
        """테이블을 청크 단위로 처리합니다.

        Args:
            table: 처리할 테이블
            processor_func: 처리 함수
            **kwargs: 처리 함수에 전달할 추가 인자

        Returns:
            pa.Table: 처리된 테이블
        """
        total_rows = len(table)
        processed_chunks = []

        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk = table.slice(start_idx, end_idx - start_idx)

            logger.debug(f"청크 처리 중: {start_idx}-{end_idx} ({len(chunk)}행)")

            try:
                # 청크 처리
                processed_chunk = processor_func(chunk, **kwargs)
                processed_chunks.append(processed_chunk)

                # 메모리 사용량 모니터링
                self._monitor_memory_usage(processed_chunks)

            except Exception as e:
                logger.error(f"청크 처리 실패 ({start_idx}-{end_idx}): {e}")
                # 실패한 청크는 건너뛰고 계속 진행
                continue

        if not processed_chunks:
            logger.warning("처리된 청크가 없습니다.")
            return pa.table({})

        # 청크들을 하나의 테이블로 결합
        result_table = pa.concat_tables(processed_chunks)
        logger.info(f"스트리밍 처리 완료: {len(result_table)}행")

        return result_table

    def _monitor_memory_usage(self, processed_chunks: List[pa.Table]) -> None:
        """메모리 사용량을 모니터링합니다.

        Args:
            processed_chunks: 처리된 청크 목록
        """
        total_size = sum(chunk.nbytes for chunk in processed_chunks)

        if total_size > self.memory_limit_bytes:
            logger.warning(f"메모리 사용량이 제한을 초과했습니다: {total_size / (1024*1024):.2f}MB")

            # 메모리 압박 시 가비지 컬렉션 강제 실행
            import gc

            gc.collect()

    def stream_data_sources(
        self,
        data_sources: Dict[str, Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]]],
        processor_func: callable,
        **kwargs,
    ) -> Generator[pa.Table, None, None]:
        """다중 데이터 소스를 스트리밍으로 처리합니다.

        Args:
            data_sources: 데이터 소스 딕셔너리
            processor_func: 처리 함수
            **kwargs: 처리 함수에 전달할 추가 인자

        Yields:
            pa.Table: 처리된 테이블 청크
        """
        for source_name, data in data_sources.items():
            logger.info(f"데이터 소스 스트리밍 처리 시작: {source_name}")

            # 데이터를 Arrow Table로 변환
            if isinstance(data, pa.Table):
                table = data
            elif isinstance(data, pd.DataFrame):
                table = pa.Table.from_pandas(data)
            elif isinstance(data, list):
                table = pa.Table.from_pylist(data)
            else:
                logger.warning(f"지원하지 않는 데이터 타입: {type(data)}")
                continue

            # 청크 단위로 처리
            for chunk in self._get_table_chunks(table):
                try:
                    processed_chunk = processor_func(chunk, source_name=source_name, **kwargs)
                    yield processed_chunk
                except Exception as e:
                    logger.error(f"소스 '{source_name}' 청크 처리 실패: {e}")
                    continue

    def _get_table_chunks(self, table: pa.Table) -> Generator[pa.Table, None, None]:
        """테이블을 청크로 분할합니다.

        Args:
            table: 분할할 테이블

        Yields:
            pa.Table: 테이블 청크
        """
        total_rows = len(table)

        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk = table.slice(start_idx, end_idx - start_idx)
            yield chunk

    def aggregate_streaming_results(
        self, processed_chunks: Iterator[pa.Table], aggregation_func: callable, **kwargs
    ) -> pa.Table:
        """스트리밍 처리 결과를 집계합니다.

        Args:
            processed_chunks: 처리된 청크들
            aggregation_func: 집계 함수
            **kwargs: 집계 함수에 전달할 추가 인자

        Returns:
            pa.Table: 집계된 결과
        """
        logger.info("스트리밍 결과 집계 시작")

        aggregated_result = None
        chunk_count = 0

        for chunk in processed_chunks:
            try:
                if aggregated_result is None:
                    aggregated_result = chunk
                else:
                    aggregated_result = aggregation_func(aggregated_result, chunk, **kwargs)

                chunk_count += 1

                # 메모리 사용량 모니터링
                if chunk_count % 10 == 0:  # 10개 청크마다 체크
                    self._monitor_memory_usage([aggregated_result])

            except Exception as e:
                logger.error(f"청크 집계 실패: {e}")
                continue

        if aggregated_result is None:
            logger.warning("집계할 청크가 없습니다.")
            return pa.table({})

        logger.info(f"스트리밍 결과 집계 완료: {chunk_count}개 청크")
        return aggregated_result

    def process_with_memory_management(
        self,
        data: Union[pa.Table, pd.DataFrame, List[Dict[str, Any]]],
        processor_func: callable,
        **kwargs,
    ) -> pa.Table:
        """메모리 관리를 포함한 데이터 처리를 수행합니다.

        Args:
            data: 처리할 데이터
            processor_func: 처리 함수
            **kwargs: 처리 함수에 전달할 추가 인자

        Returns:
            pa.Table: 처리된 테이블
        """
        # 데이터를 Arrow Table로 변환
        if isinstance(data, pa.Table):
            table = data
        elif isinstance(data, pd.DataFrame):
            table = pa.Table.from_pandas(data)
        elif isinstance(data, list):
            table = pa.Table.from_pylist(data)
        else:
            raise ValueError(f"지원하지 않는 데이터 타입: {type(data)}")

        # 메모리 사용량 확인
        table_size = table.nbytes
        logger.info(f"데이터 크기: {table_size / (1024*1024):.2f}MB")

        if table_size > self.memory_limit_bytes:
            # 스트리밍 처리
            return self._process_in_chunks(table, processor_func, **kwargs)
        else:
            # 일반 처리
            return processor_func(table, **kwargs)

    def optimize_chunk_size(self, table: pa.Table) -> int:
        """테이블 크기에 따라 최적의 청크 크기를 계산합니다.

        Args:
            table: 분석할 테이블

        Returns:
            int: 최적의 청크 크기
        """
        table_size = table.nbytes
        row_count = len(table)

        if row_count == 0:
            return self.chunk_size

        # 행당 평균 크기 계산
        avg_row_size = table_size / row_count

        # 메모리 제한 내에서 처리할 수 있는 행 수 계산
        max_rows_per_chunk = int(self.memory_limit_bytes / avg_row_size)

        # 청크 크기를 조정 (최소 1000, 최대 50000)
        optimal_chunk_size = max(1000, min(max_rows_per_chunk, 50000))

        logger.info(
            f"최적 청크 크기 계산: {optimal_chunk_size} (테이블 크기: {table_size / (1024*1024):.2f}MB)"
        )

        return optimal_chunk_size

    def get_memory_usage(self, table: pa.Table) -> Dict[str, Any]:
        """테이블의 메모리 사용량을 분석합니다.

        Args:
            table: 분석할 테이블

        Returns:
            Dict[str, Any]: 메모리 사용량 정보
        """
        total_size = table.nbytes
        row_count = len(table)
        column_count = len(table.schema)

        # 컬럼별 메모리 사용량
        column_sizes = {}
        for i, field in enumerate(table.schema):
            column_array = table.column(i)
            column_sizes[field.name] = {
                "bytes": column_array.nbytes,
                "mb": column_array.nbytes / (1024 * 1024),
                "type": str(field.type),
            }

        return {
            "total_size": {
                "bytes": total_size,
                "mb": total_size / (1024 * 1024),
                "gb": total_size / (1024 * 1024 * 1024),
            },
            "row_count": row_count,
            "column_count": column_count,
            "avg_row_size": total_size / row_count if row_count > 0 else 0,
            "column_sizes": column_sizes,
            "memory_efficiency": (
                "good" if total_size < self.memory_limit_bytes else "needs_optimization"
            ),
        }

    def __repr__(self) -> str:
        """문자열 표현."""
        return f"StreamingProcessor(chunk_size={self.chunk_size}, memory_limit={self.memory_limit_mb}MB)"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
