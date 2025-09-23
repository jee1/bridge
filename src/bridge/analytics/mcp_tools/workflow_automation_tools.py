"""워크플로 자동화 MCP 도구.

CA 마일스톤 3.4: 워크플로 및 자동화 시스템
- 분석 템플릿, 고급 스케줄링, 워크플로 시각화, 성능 최적화
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from bridge.analytics.core.data_integration import UnifiedDataFrame
from bridge.analytics.workflows.analysis_templates import AnalysisTemplateManager

logger = logging.getLogger(__name__)


class WorkflowAutomationTools:
    """워크플로 자동화 MCP 도구 클래스"""

    def __init__(self):
        """워크플로 자동화 도구 초기화"""
        self.logger = logging.getLogger(__name__)
        self.template_manager = AnalysisTemplateManager()

    def execute_analysis_template(
        self,
        template_name: str,
        data: UnifiedDataFrame,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """분석 템플릿 실행

        Args:
            template_name: 템플릿 이름
            data: 분석할 데이터
            parameters: 분석 매개변수 (선택사항)

        Returns:
            분석 결과
        """
        try:
            if parameters is None:
                parameters = {}

            result = self.template_manager.execute_template(template_name, data, parameters)

            return {
                "success": result.success,
                "template_name": result.template_name,
                "results": result.results,
                "visualizations": result.visualizations,
                "recommendations": result.recommendations,
                "metadata": result.metadata,
            }

        except Exception as e:
            self.logger.error(f"분석 템플릿 실행 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "template_name": template_name,
                "results": {},
                "visualizations": [],
                "recommendations": [f"템플릿 실행 실패: {str(e)}"],
                "metadata": {"error": str(e)},
            }

    def list_analysis_templates(self) -> Dict[str, Any]:
        """분석 템플릿 목록 조회

        Returns:
            템플릿 목록
        """
        try:
            templates = self.template_manager.list_templates()

            return {"success": True, "templates": templates, "total_count": len(templates)}

        except Exception as e:
            self.logger.error(f"템플릿 목록 조회 실패: {e}")
            return {"success": False, "error": str(e), "templates": []}

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """템플릿 정보 조회

        Args:
            template_name: 템플릿 이름

        Returns:
            템플릿 정보
        """
        try:
            template = self.template_manager.get_template(template_name)
            if not template:
                return {"success": False, "error": f"템플릿 {template_name}을 찾을 수 없습니다."}

            template_info = template.get_template_info()

            return {
                "success": True,
                "template_info": {
                    "name": template_info.name,
                    "description": template_info.description,
                    "required_columns": template_info.required_columns,
                    "optional_columns": template_info.optional_columns,
                    "parameters": template_info.parameters,
                    "output_format": template_info.output_format,
                },
            }

        except Exception as e:
            self.logger.error(f"템플릿 정보 조회 실패: {e}")
            return {"success": False, "error": str(e)}

    def validate_data_for_template(
        self, template_name: str, data: UnifiedDataFrame
    ) -> Dict[str, Any]:
        """템플릿용 데이터 검증

        Args:
            template_name: 템플릿 이름
            data: 검증할 데이터

        Returns:
            검증 결과
        """
        try:
            template = self.template_manager.get_template(template_name)
            if not template:
                return {"success": False, "error": f"템플릿 {template_name}을 찾을 수 없습니다."}

            is_valid = template.validate_data(data)
            template_info = template.get_template_info()

            # 누락된 컬럼 확인
            df = data.to_pandas()
            missing_required = [
                col for col in template_info.required_columns if col not in df.columns
            ]
            missing_optional = [
                col for col in template_info.optional_columns if col not in df.columns
            ]

            return {
                "success": is_valid,
                "is_valid": is_valid,
                "missing_required_columns": missing_required,
                "missing_optional_columns": missing_optional,
                "available_columns": df.columns.tolist(),
                "required_columns": template_info.required_columns,
                "optional_columns": template_info.optional_columns,
            }

        except Exception as e:
            self.logger.error(f"데이터 검증 실패: {e}")
            return {"success": False, "error": str(e), "is_valid": False}

    def create_workflow_dag(self, workflow_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """워크플로 DAG 생성

        Args:
            workflow_steps: 워크플로 단계 목록

        Returns:
            DAG 정보
        """
        try:
            # DAG 구조 생성
            dag = {"nodes": [], "edges": [], "execution_order": []}

            # 노드 생성
            for i, step in enumerate(workflow_steps):
                node = {
                    "id": step.get("id", f"step_{i}"),
                    "name": step.get("name", f"Step {i+1}"),
                    "type": step.get("type", "analysis"),
                    "template": step.get("template"),
                    "parameters": step.get("parameters", {}),
                    "dependencies": step.get("dependencies", []),
                    "status": "pending",
                }
                dag["nodes"].append(node)

            # 엣지 생성 (의존성 기반)
            for node in dag["nodes"]:
                for dep in node["dependencies"]:
                    edge = {"from": dep, "to": node["id"]}
                    dag["edges"].append(edge)

            # 실행 순서 계산 (위상 정렬)
            dag["execution_order"] = self._calculate_execution_order(dag)

            return {
                "success": True,
                "dag": dag,
                "total_steps": len(workflow_steps),
                "execution_time_estimate": len(dag["execution_order"]) * 30,  # 30초 per step
            }

        except Exception as e:
            self.logger.error(f"워크플로 DAG 생성 실패: {e}")
            return {"success": False, "error": str(e), "dag": None}

    def optimize_workflow_performance(self, workflow_config: Dict[str, Any]) -> Dict[str, Any]:
        """워크플로 성능 최적화

        Args:
            workflow_config: 워크플로 설정

        Returns:
            최적화 결과
        """
        try:
            optimizations = []

            # 병렬 실행 가능한 단계 식별
            parallel_steps = self._identify_parallel_steps(workflow_config)
            if parallel_steps:
                optimizations.append(
                    {
                        "type": "parallel_execution",
                        "description": f"{len(parallel_steps)}개 단계를 병렬로 실행할 수 있습니다.",
                        "steps": parallel_steps,
                        "estimated_time_saving": len(parallel_steps) * 20,  # 20초 per step
                    }
                )

            # 캐싱 가능한 단계 식별
            cacheable_steps = self._identify_cacheable_steps(workflow_config)
            if cacheable_steps:
                optimizations.append(
                    {
                        "type": "caching",
                        "description": f"{len(cacheable_steps)}개 단계에 캐싱을 적용할 수 있습니다.",
                        "steps": cacheable_steps,
                        "estimated_time_saving": len(cacheable_steps) * 15,  # 15초 per step
                    }
                )

            # 데이터 전처리 최적화
            preprocessing_optimizations = self._optimize_preprocessing(workflow_config)
            if preprocessing_optimizations:
                optimizations.append(
                    {
                        "type": "preprocessing_optimization",
                        "description": "데이터 전처리 단계를 최적화할 수 있습니다.",
                        "optimizations": preprocessing_optimizations,
                        "estimated_time_saving": 30,  # 30초
                    }
                )

            return {
                "success": True,
                "optimizations": optimizations,
                "total_estimated_saving": sum(
                    opt.get("estimated_time_saving", 0) for opt in optimizations
                ),
                "optimization_score": min(100, len(optimizations) * 25),  # 0-100 점수
            }

        except Exception as e:
            self.logger.error(f"워크플로 성능 최적화 실패: {e}")
            return {"success": False, "error": str(e), "optimizations": []}

    def _calculate_execution_order(self, dag: Dict[str, Any]) -> List[str]:
        """실행 순서 계산 (위상 정렬)

        Args:
            dag: DAG 정보

        Returns:
            실행 순서
        """
        try:
            # 진입 차수 계산
            in_degree = {node["id"]: 0 for node in dag["nodes"]}
            for edge in dag["edges"]:
                in_degree[edge["to"]] += 1

            # 큐 초기화 (진입 차수가 0인 노드들)
            queue = [node["id"] for node in dag["nodes"] if in_degree[node["id"]] == 0]
            execution_order = []

            while queue:
                current = queue.pop(0)
                execution_order.append(current)

                # 현재 노드에서 나가는 엣지들 처리
                for edge in dag["edges"]:
                    if edge["from"] == current:
                        in_degree[edge["to"]] -= 1
                        if in_degree[edge["to"]] == 0:
                            queue.append(edge["to"])

            return execution_order

        except Exception as e:
            self.logger.error(f"실행 순서 계산 실패: {e}")
            return []

    def _identify_parallel_steps(self, workflow_config: Dict[str, Any]) -> List[str]:
        """병렬 실행 가능한 단계 식별

        Args:
            workflow_config: 워크플로 설정

        Returns:
            병렬 실행 가능한 단계 목록
        """
        try:
            parallel_steps = []

            # 의존성이 없는 단계들을 병렬 실행 가능으로 식별
            if "steps" in workflow_config:
                for step in workflow_config["steps"]:
                    if not step.get("dependencies") or len(step.get("dependencies", [])) == 0:
                        parallel_steps.append(step.get("id", step.get("name", "unknown")))

            return parallel_steps

        except Exception as e:
            self.logger.error(f"병렬 단계 식별 실패: {e}")
            return []

    def _identify_cacheable_steps(self, workflow_config: Dict[str, Any]) -> List[str]:
        """캐싱 가능한 단계 식별

        Args:
            workflow_config: 워크플로 설정

        Returns:
            캐싱 가능한 단계 목록
        """
        try:
            cacheable_steps = []

            # 데이터 읽기나 변환 단계들을 캐싱 가능으로 식별
            if "steps" in workflow_config:
                for step in workflow_config["steps"]:
                    step_type = step.get("type", "")
                    if step_type in ["data_loading", "data_transformation", "preprocessing"]:
                        cacheable_steps.append(step.get("id", step.get("name", "unknown")))

            return cacheable_steps

        except Exception as e:
            self.logger.error(f"캐싱 가능 단계 식별 실패: {e}")
            return []

    def _optimize_preprocessing(self, workflow_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """전처리 최적화 제안

        Args:
            workflow_config: 워크플로 설정

        Returns:
            최적화 제안 목록
        """
        try:
            optimizations = []

            # 데이터 타입 최적화
            optimizations.append(
                {
                    "type": "data_type_optimization",
                    "description": "데이터 타입을 최적화하여 메모리 사용량을 줄일 수 있습니다.",
                    "suggestion": "불필요한 object 타입을 numeric 타입으로 변환하세요.",
                }
            )

            # 컬럼 선택 최적화
            optimizations.append(
                {
                    "type": "column_selection",
                    "description": "사용하지 않는 컬럼을 제거하여 처리 속도를 높일 수 있습니다.",
                    "suggestion": "분석에 필요한 컬럼만 선택하세요.",
                }
            )

            # 인덱싱 최적화
            optimizations.append(
                {
                    "type": "indexing_optimization",
                    "description": "적절한 인덱스를 설정하여 조회 성능을 향상시킬 수 있습니다.",
                    "suggestion": "자주 사용되는 컬럼에 인덱스를 설정하세요.",
                }
            )

            return optimizations

        except Exception as e:
            self.logger.error(f"전처리 최적화 실패: {e}")
            return []
