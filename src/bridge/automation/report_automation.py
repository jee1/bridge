"""자동 리포트 생성 시스템

스케줄링된 분석 리포트를 자동으로 생성하고 배포합니다.
"""

import logging
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from bridge.analytics.core import (
    ChartGenerator,
    DashboardGenerator,
    QualityChecker,
    ReportGenerator,
    StatisticsAnalyzer,
    UnifiedDataFrame,
)

logger = logging.getLogger(__name__)


class ReportStatus(Enum):
    """리포트 상태"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduleType(Enum):
    """스케줄 타입"""

    ONCE = "once"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


@dataclass
class ReportTemplate:
    """리포트 템플릿"""

    id: str
    name: str
    description: Optional[str] = None
    data_source: str = None
    query: Optional[str] = None
    analysis_config: Dict[str, Any] = None
    chart_configs: List[Dict[str, Any]] = None
    dashboard_config: Dict[str, Any] = None
    output_format: str = "html"  # html, pdf, json
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.analysis_config is None:
            self.analysis_config = {}
        if self.chart_configs is None:
            self.chart_configs = []
        if self.dashboard_config is None:
            self.dashboard_config = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class ReportSchedule:
    """리포트 스케줄"""

    id: str
    template_id: str
    schedule_type: ScheduleType
    cron_expression: Optional[str] = None  # CUSTOM 타입일 때 사용
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class ReportJob:
    """리포트 작업"""

    id: str
    template_id: str
    schedule_id: Optional[str] = None
    status: ReportStatus = ReportStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_path: Optional[str] = None
    parameters: Dict[str, Any] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class ReportAutomation:
    """리포트 자동화 시스템"""

    def __init__(self, output_directory: str = "reports"):
        self.output_directory = output_directory
        self.templates: Dict[str, ReportTemplate] = {}
        self.schedules: Dict[str, ReportSchedule] = {}
        self.jobs: List[ReportJob] = []
        self.status = "stopped"
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)

        # 분석 도구들
        self.stats_analyzer = StatisticsAnalyzer()
        self.quality_checker = QualityChecker()
        self.chart_generator = ChartGenerator()
        self.dashboard_generator = DashboardGenerator()
        self.report_generator = ReportGenerator()

        # 콜백 함수들
        self.job_callbacks: List[Callable[[ReportJob], None]] = []

    def create_template(self, template: ReportTemplate) -> bool:
        """리포트 템플릿 생성"""
        try:
            self.templates[template.id] = template
            self.logger.info(f"리포트 템플릿 생성 완료: {template.id}")
            return True
        except Exception as e:
            self.logger.error(f"리포트 템플릿 생성 실패: {e}")
            return False

    def create_schedule(self, schedule: ReportSchedule) -> bool:
        """리포트 스케줄 생성"""
        try:
            if schedule.template_id not in self.templates:
                self.logger.error(f"템플릿이 존재하지 않습니다: {schedule.template_id}")
                return False

            self.schedules[schedule.id] = schedule
            self.logger.info(f"리포트 스케줄 생성 완료: {schedule.id}")
            return True
        except Exception as e:
            self.logger.error(f"리포트 스케줄 생성 실패: {e}")
            return False

    def start_scheduler(self) -> bool:
        """스케줄러 시작"""
        if self.status == "running":
            return True

        try:
            self.stop_event.clear()
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            self.status = "running"
            self.logger.info("리포트 스케줄러 시작")
            return True
        except Exception as e:
            self.logger.error(f"스케줄러 시작 실패: {e}")
            return False

    def stop_scheduler(self) -> bool:
        """스케줄러 중지"""
        try:
            self.stop_event.set()
            if self.scheduler_thread and self.scheduler_thread.is_alive():
                self.scheduler_thread.join(timeout=5)
            self.status = "stopped"
            self.logger.info("리포트 스케줄러 중지")
            return True
        except Exception as e:
            self.logger.error(f"스케줄러 중지 실패: {e}")
            return False

    def _scheduler_loop(self):
        """스케줄러 루프"""
        while not self.stop_event.is_set():
            try:
                self._check_schedules()
                time.sleep(60)  # 1분마다 확인
            except Exception as e:
                self.logger.error(f"스케줄러 루프 오류: {e}")
                break

    def _check_schedules(self):
        """스케줄 확인"""
        now = datetime.now()

        for schedule in self.schedules.values():
            if not schedule.enabled:
                continue

            if self._should_run_schedule(schedule, now):
                self._execute_schedule(schedule)

    def _should_run_schedule(self, schedule: ReportSchedule, now: datetime) -> bool:
        """스케줄 실행 여부 확인"""
        if schedule.schedule_type == ScheduleType.ONCE:
            if schedule.start_time and now >= schedule.start_time:
                return True
        elif schedule.schedule_type == ScheduleType.DAILY:
            # 매일 특정 시간에 실행
            if schedule.start_time:
                target_time = now.replace(
                    hour=schedule.start_time.hour,
                    minute=schedule.start_time.minute,
                    second=0,
                    microsecond=0,
                )
                return now >= target_time
        elif schedule.schedule_type == ScheduleType.WEEKLY:
            # 매주 특정 요일, 시간에 실행
            if schedule.start_time:
                target_time = now.replace(
                    hour=schedule.start_time.hour,
                    minute=schedule.start_time.minute,
                    second=0,
                    microsecond=0,
                )
                return now >= target_time and now.weekday() == schedule.start_time.weekday()
        elif schedule.schedule_type == ScheduleType.MONTHLY:
            # 매월 특정 일, 시간에 실행
            if schedule.start_time:
                target_time = now.replace(
                    day=schedule.start_time.day,
                    hour=schedule.start_time.hour,
                    minute=schedule.start_time.minute,
                    second=0,
                    microsecond=0,
                )
                return now >= target_time

        return False

    def _execute_schedule(self, schedule: ReportSchedule):
        """스케줄 실행"""
        template = self.templates.get(schedule.template_id)
        if not template:
            self.logger.error(f"템플릿을 찾을 수 없습니다: {schedule.template_id}")
            return

        # 리포트 작업 생성
        job = ReportJob(
            id=f"job_{int(time.time())}_{schedule.id}",
            template_id=template.id,
            schedule_id=schedule.id,
        )

        self.jobs.append(job)
        self._execute_report_job(job)

    def execute_template(
        self, template_id: str, parameters: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """템플릿 즉시 실행"""
        template = self.templates.get(template_id)
        if not template:
            self.logger.error(f"템플릿을 찾을 수 없습니다: {template_id}")
            return None

        job = ReportJob(
            id=f"manual_{int(time.time())}_{template_id}",
            template_id=template_id,
            parameters=parameters or {},
        )

        self.jobs.append(job)
        self._execute_report_job(job)

        return job.id

    def _execute_report_job(self, job: ReportJob):
        """리포트 작업 실행"""
        job.status = ReportStatus.RUNNING
        job.started_at = datetime.now()

        try:
            template = self.templates[job.template_id]

            # 데이터 가져오기 (실제 구현에서는 데이터 소스에서 가져옴)
            data = self._fetch_data(template, job.parameters)
            if data is None:
                raise Exception("데이터를 가져올 수 없습니다")

            # 리포트 생성
            report_path = self._generate_report(template, data, job.id)
            if not report_path:
                raise Exception("리포트 생성에 실패했습니다")

            job.status = ReportStatus.COMPLETED
            job.completed_at = datetime.now()
            job.output_path = report_path

            self.logger.info(f"리포트 작업 완료: {job.id}")

        except Exception as e:
            job.status = ReportStatus.FAILED
            job.completed_at = datetime.now()
            job.error_message = str(e)
            self.logger.error(f"리포트 작업 실패 {job.id}: {e}")

        # 콜백 함수 호출
        for callback in self.job_callbacks:
            try:
                callback(job)
            except Exception as e:
                self.logger.error(f"작업 콜백 실행 실패: {e}")

    def _fetch_data(
        self, template: ReportTemplate, parameters: Dict[str, Any]
    ) -> Optional[UnifiedDataFrame]:
        """데이터 가져오기 (모의 구현)"""
        import numpy as np
        import pandas as pd

        # 실제 구현에서는 데이터 소스에서 쿼리 실행
        # 여기서는 모의 데이터 생성
        np.random.seed(42)
        data = {
            "id": list(range(100)),
            "value1": np.random.normal(100, 15, 100).tolist(),
            "value2": np.random.normal(50, 10, 100).tolist(),
            "category": (["A", "B", "C"] * 33 + ["A"])[:100],
            "date": [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(100)],
        }

        try:
            return UnifiedDataFrame(data)
        except Exception as e:
            self.logger.error(f"데이터 생성 실패: {e}")
            return None

    def _generate_report(
        self, template: ReportTemplate, data: UnifiedDataFrame, job_id: str
    ) -> Optional[str]:
        """리포트 생성"""
        try:
            import os

            os.makedirs(self.output_directory, exist_ok=True)

            # 리포트 설정
            from bridge.analytics.core import ReportConfig

            report_config = ReportConfig(
                title=template.name,
                author=template.created_by or "Bridge Analytics",
                sections=["overview", "statistics", "quality", "charts", "dashboard"],
            )

            # 리포트 생성
            report_data = self.report_generator.generate_analytics_report(data, report_config)

            # 파일로 저장
            if template.output_format == "html":
                report_path = self._save_html_report(report_data, job_id)
            elif template.output_format == "json":
                report_path = self._save_json_report(report_data, job_id)
            else:
                report_path = self._save_html_report(report_data, job_id)

            return report_path

        except Exception as e:
            self.logger.error(f"리포트 생성 실패: {e}")
            return None

    def _save_html_report(self, report_data: Dict[str, Any], job_id: str) -> str:
        """HTML 리포트 저장"""
        import os

        report_path = os.path.join(self.output_directory, f"report_{job_id}.html")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{report_data.get('title', 'Analytics Report')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{report_data.get('title', 'Analytics Report')}</h1>
                <p>작성자: {report_data.get('author', 'Bridge Analytics')}</p>
                <p>생성일: {report_data.get('date', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
            </div>
            
            <div class="section">
                <h2>기본 통계</h2>
                <p>데이터 행 수: {report_data.get('basic_stats', {}).get('rows', 'N/A')}</p>
                <p>데이터 열 수: {report_data.get('basic_stats', {}).get('columns', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>품질 점수</h2>
                <p>전체 점수: {report_data.get('quality', {}).get('overall_score', 'N/A')}</p>
                <p>결측값 점수: {report_data.get('quality', {}).get('missing_value_score', 'N/A')}</p>
                <p>이상치 점수: {report_data.get('quality', {}).get('outlier_score', 'N/A')}</p>
                <p>일관성 점수: {report_data.get('quality', {}).get('consistency_score', 'N/A')}</p>
            </div>
            
            <div class="section">
                <h2>권장사항</h2>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in report_data.get('quality', {}).get('recommendations', [])])}
                </ul>
            </div>
        </body>
        </html>
        """

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return report_path

    def _save_json_report(self, report_data: Dict[str, Any], job_id: str) -> str:
        """JSON 리포트 저장"""
        import json
        import os

        report_path = os.path.join(self.output_directory, f"report_{job_id}.json")

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        return report_path

    def get_job_status(self, job_id: str) -> Optional[ReportJob]:
        """작업 상태 조회"""
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None

    def get_jobs(
        self,
        status: Optional[ReportStatus] = None,
        template_id: Optional[str] = None,
        hours: int = 24,
    ) -> List[ReportJob]:
        """작업 목록 조회"""
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_jobs = [job for job in self.jobs if job.created_at >= cutoff_time]

        if status is not None:
            filtered_jobs = [job for job in filtered_jobs if job.status == status]

        if template_id is not None:
            filtered_jobs = [job for job in filtered_jobs if job.template_id == template_id]

        return sorted(filtered_jobs, key=lambda x: x.created_at, reverse=True)

    def cancel_job(self, job_id: str) -> bool:
        """작업 취소"""
        for job in self.jobs:
            if job.id == job_id and job.status in [ReportStatus.PENDING, ReportStatus.RUNNING]:
                job.status = ReportStatus.CANCELLED
                job.completed_at = datetime.now()
                self.logger.info(f"작업 취소 완료: {job_id}")
                return True
        return False

    def add_job_callback(self, callback: Callable[[ReportJob], None]) -> None:
        """작업 콜백 추가"""
        self.job_callbacks.append(callback)

    def get_scheduler_status(self) -> Dict[str, Any]:
        """스케줄러 상태 조회"""
        return {
            "status": self.status,
            "templates_count": len(self.templates),
            "schedules_count": len(self.schedules),
            "jobs_count": len(self.jobs),
            "pending_jobs": len([j for j in self.jobs if j.status == ReportStatus.PENDING]),
            "running_jobs": len([j for j in self.jobs if j.status == ReportStatus.RUNNING]),
            "completed_jobs": len([j for j in self.jobs if j.status == ReportStatus.COMPLETED]),
            "failed_jobs": len([j for j in self.jobs if j.status == ReportStatus.FAILED]),
        }

    def export_templates(self, file_path: str) -> bool:
        """템플릿 내보내기"""
        try:
            import json

            data = {
                "templates": [asdict(template) for template in self.templates.values()],
                "schedules": [asdict(schedule) for schedule in self.schedules.values()],
                "exported_at": datetime.now().isoformat(),
            }

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self.logger.info(f"템플릿 내보내기 완료: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"템플릿 내보내기 실패: {e}")
            return False
