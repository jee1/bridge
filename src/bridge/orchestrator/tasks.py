"""오케스트레이터 Celery 태스크."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from bridge.connectors import ConnectorNotFoundError, connector_registry
from bridge.connectors.exceptions import ConnectionError, MetadataError, QueryExecutionError

from .celery_app import celery_app

logger = logging.getLogger(__name__)


def _run_maybe_async(result: Any) -> Any:
    """비동기 결과를 동기적으로 실행한다."""
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


def _process_connector(connector, source_name: str) -> Dict[str, Any]:
    """커넥터를 처리하고 결과를 반환한다."""
    try:
        logger.info(f"커넥터 '{source_name}' 발견, 메타데이터 수집 중...")

        metadata = _run_maybe_async(connector.get_metadata())
        logger.info(f"커넥터 '{source_name}' 메타데이터 수집 완료")

        return {"success": True, "source": source_name, "metadata": metadata, "error": None}
    except ConnectionError as e:
        logger.error(f"커넥터 '{source_name}' 연결 실패: {e}")
        return {
            "success": False,
            "source": source_name,
            "metadata": None,
            "error": {"message": str(e), "type": "connection"},
        }
    except MetadataError as e:
        logger.error(f"커넥터 '{source_name}' 메타데이터 조회 실패: {e}")
        return {
            "success": False,
            "source": source_name,
            "metadata": None,
            "error": {"message": str(e), "type": "metadata"},
        }
    except Exception as e:
        logger.error(f"커넥터 '{source_name}' 처리 중 예상치 못한 오류: {e}")
        return {
            "success": False,
            "source": source_name,
            "metadata": None,
            "error": {"message": str(e), "type": "unknown"},
        }


def _determine_status(missing_sources: List[str], errors: List[Dict[str, str]]) -> str:
    """상태를 결정한다."""
    if missing_sources and errors:
        return "failed"
    elif missing_sources or errors:
        return "partial"
    else:
        return "completed"


@celery_app.task(name="bridge.execute_pipeline")
def execute_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    """컨텍스트 수집 및 도구 실행을 모사한 태스크."""
    try:
        intent = payload.get("intent", "")
        sources: List[str] = payload.get("sources", [])
        tools = payload.get("required_tools", [])

        logger.info(f"파이프라인 실행 시작: intent={intent}, sources={sources}, tools={tools}")

        collected_context: List[Dict[str, Any]] = []
        missing_sources: List[str] = []
        errors: List[Dict[str, str]] = []

        for source_name in sources:
            try:
                connector = connector_registry.get(source_name)
                result = _process_connector(connector, source_name)

                if result["success"]:
                    collected_context.append(
                        {
                            "source": result["source"],
                            "metadata": result["metadata"],
                        }
                    )
                else:
                    errors.append(
                        {
                            "source": result["source"],
                            "error": result["error"]["message"],
                            "type": result["error"]["type"],
                        }
                    )

            except ConnectorNotFoundError:
                logger.warning(f"커넥터 '{source_name}'를 찾을 수 없습니다")
                missing_sources.append(source_name)

        status = _determine_status(missing_sources, errors)

        result = {
            "intent": intent,
            "status": status,
            "collected_sources": collected_context,
            "missing_sources": missing_sources,
            "errors": errors,
            "triggered_tools": tools,
        }

        logger.info(
            f"파이프라인 실행 완료: status={status}, collected={len(collected_context)}, missing={len(missing_sources)}, errors={len(errors)}"
        )
        return result

    except Exception as e:
        logger.error(f"파이프라인 실행 중 치명적 오류: {e}")
        return {
            "intent": payload.get("intent", ""),
            "status": "failed",
            "collected_sources": [],
            "missing_sources": payload.get("sources", []),
            "errors": [{"source": "system", "error": str(e), "type": "fatal"}],
            "triggered_tools": payload.get("required_tools", []),
        }
