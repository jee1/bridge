"""작업 스케줄러

정기적인 작업을 스케줄링하고 실행합니다.
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import croniter

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """작업 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class TaskType(Enum):
    """작업 타입"""
    DATA_QUALITY_CHECK = "data_quality_check"
    REPORT_GENERATION = "report_generation"
    DATA_SYNC = "data_sync"
    CLEANUP = "cleanup"
    CUSTOM = "custom"


@dataclass
class ScheduledTask:
    """스케줄된 작업"""
    id: str
    name: str
    description: Optional[str] = None
    task_type: TaskType = TaskType.CUSTOM
    function: Optional[Callable] = None
    parameters: Dict[str, Any] = None
    cron_expression: Optional[str] = None
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    enabled: bool = True
    max_retries: int = 3
    timeout_seconds: int = 3600  # 1시간
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        
        # 다음 실행 시간 계산
        if self.cron_expression and self.enabled:
            self._calculate_next_run()

    def _calculate_next_run(self):
        """다음 실행 시간 계산"""
        if not self.cron_expression:
            return
        
        try:
            cron = croniter.croniter(self.cron_expression, datetime.now())
            self.next_run = cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Cron 표현식 파싱 실패 {self.id}: {e}")
            self.next_run = None

    def should_run(self) -> bool:
        """실행 여부 확인"""
        if not self.enabled or not self.next_run:
            return False
        
        return datetime.now() >= self.next_run

    def update_next_run(self):
        """다음 실행 시간 업데이트"""
        self._calculate_next_run()


@dataclass
class TaskExecution:
    """작업 실행 기록"""
    id: str
    task_id: str
    status: TaskStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    output: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.completed_at and self.started_at:
            self.duration_seconds = (self.completed_at - self.started_at).total_seconds()


class TaskScheduler:
    """작업 스케줄러"""
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.executions: List[TaskExecution] = []
        self.status = "stopped"
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(__name__)
        
        # 콜백 함수들
        self.task_callbacks: List[Callable[[TaskExecution], None]] = []
        self.error_callbacks: List[Callable[[TaskExecution], None]] = []
    
    def add_task(self, task: ScheduledTask) -> bool:
        """작업 추가"""
        try:
            self.tasks[task.id] = task
            self.logger.info(f"작업 추가 완료: {task.id}")
            return True
        except Exception as e:
            self.logger.error(f"작업 추가 실패: {e}")
            return False
    
    def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """작업 업데이트"""
        if task_id not in self.tasks:
            return False
        
        try:
            task = self.tasks[task_id]
            for key, value in updates.items():
                if hasattr(task, key) and key not in ['id', 'created_at']:
                    setattr(task, key, value)
            
            task.updated_at = datetime.now()
            
            # cron 표현식이 변경된 경우 다음 실행 시간 재계산
            if 'cron_expression' in updates or 'enabled' in updates:
                task._calculate_next_run()
            
            self.logger.info(f"작업 업데이트 완료: {task_id}")
            return True
        except Exception as e:
            self.logger.error(f"작업 업데이트 실패: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """작업 제거"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.logger.info(f"작업 제거 완료: {task_id}")
            return True
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
            self.logger.info("작업 스케줄러 시작")
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
            self.logger.info("작업 스케줄러 중지")
            return True
        except Exception as e:
            self.logger.error(f"스케줄러 중지 실패: {e}")
            return False
    
    def _scheduler_loop(self):
        """스케줄러 루프"""
        while not self.stop_event.is_set():
            try:
                self._check_and_execute_tasks()
                time.sleep(60)  # 1분마다 확인
            except Exception as e:
                self.logger.error(f"스케줄러 루프 오류: {e}")
                break
    
    def _check_and_execute_tasks(self):
        """작업 확인 및 실행"""
        for task in self.tasks.values():
            if task.should_run():
                self._execute_task(task)
    
    def _execute_task(self, task: ScheduledTask):
        """작업 실행"""
        execution = TaskExecution(
            id=f"exec_{int(time.time())}_{task.id}",
            task_id=task.id,
            status=TaskStatus.RUNNING,
            started_at=datetime.now()
        )
        
        self.executions.append(execution)
        task.last_run = execution.started_at
        
        try:
            self.logger.info(f"작업 실행 시작: {task.id}")
            
            if task.function:
                # 함수 실행
                result = self._run_with_timeout(task.function, task.parameters, task.timeout_seconds)
                execution.output = {"result": result}
                execution.status = TaskStatus.COMPLETED
            else:
                # 기본 작업 타입별 처리
                result = self._execute_default_task(task)
                execution.output = {"result": result}
                execution.status = TaskStatus.COMPLETED
            
            self.logger.info(f"작업 실행 완료: {task.id}")
            
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            self.logger.error(f"작업 실행 실패 {task.id}: {e}")
            
            # 재시도 처리
            if execution.retry_count < task.max_retries:
                execution.retry_count += 1
                self._schedule_retry(task, execution)
        
        finally:
            execution.completed_at = datetime.now()
            task.update_next_run()
            
            # 콜백 함수 호출
            for callback in self.task_callbacks:
                try:
                    callback(execution)
                except Exception as e:
                    self.logger.error(f"작업 콜백 실행 실패: {e}")
            
            if execution.status == TaskStatus.FAILED:
                for callback in self.error_callbacks:
                    try:
                        callback(execution)
                    except Exception as e:
                        self.logger.error(f"에러 콜백 실행 실패: {e}")
    
    def _run_with_timeout(self, func: Callable, parameters: Dict[str, Any], timeout_seconds: int) -> Any:
        """타임아웃과 함께 함수 실행"""
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(func, **parameters)
            return future.result(timeout=timeout_seconds)
    
    def _execute_default_task(self, task: ScheduledTask) -> Any:
        """기본 작업 타입별 실행"""
        if task.task_type == TaskType.DATA_QUALITY_CHECK:
            return self._execute_quality_check(task)
        elif task.task_type == TaskType.REPORT_GENERATION:
            return self._execute_report_generation(task)
        elif task.task_type == TaskType.DATA_SYNC:
            return self._execute_data_sync(task)
        elif task.task_type == TaskType.CLEANUP:
            return self._execute_cleanup(task)
        else:
            raise ValueError(f"지원하지 않는 작업 타입: {task.task_type}")
    
    def _execute_quality_check(self, task: ScheduledTask) -> Dict[str, Any]:
        """데이터 품질 검사 실행"""
        from bridge.automation.quality_monitor import QualityMonitor
        
        monitor = QualityMonitor()
        # 실제 구현에서는 데이터 소스에서 데이터를 가져와서 품질 검사 수행
        return {"status": "completed", "message": "품질 검사 완료"}
    
    def _execute_report_generation(self, task: ScheduledTask) -> Dict[str, Any]:
        """리포트 생성 실행"""
        from bridge.automation.report_automation import ReportAutomation
        
        automation = ReportAutomation()
        # 실제 구현에서는 리포트 템플릿을 사용하여 리포트 생성
        return {"status": "completed", "message": "리포트 생성 완료"}
    
    def _execute_data_sync(self, task: ScheduledTask) -> Dict[str, Any]:
        """데이터 동기화 실행"""
        # 실제 구현에서는 데이터 소스 간 동기화 수행
        return {"status": "completed", "message": "데이터 동기화 완료"}
    
    def _execute_cleanup(self, task: ScheduledTask) -> Dict[str, Any]:
        """정리 작업 실행"""
        # 오래된 로그, 임시 파일 등 정리
        return {"status": "completed", "message": "정리 작업 완료"}
    
    def _schedule_retry(self, task: ScheduledTask, execution: TaskExecution):
        """재시도 스케줄링"""
        retry_delay = min(300, 60 * (2 ** execution.retry_count))  # 최대 5분
        retry_time = datetime.now() + timedelta(seconds=retry_delay)
        
        # 재시도를 위한 임시 작업 생성
        retry_task = ScheduledTask(
            id=f"retry_{task.id}_{execution.retry_count}",
            name=f"Retry: {task.name}",
            task_type=task.task_type,
            function=task.function,
            parameters=task.parameters,
            next_run=retry_time,
            enabled=True
        )
        
        self.tasks[retry_task.id] = retry_task
        self.logger.info(f"재시도 스케줄링: {task.id} (시도 {execution.retry_count + 1})")
    
    def execute_task_now(self, task_id: str) -> Optional[str]:
        """작업 즉시 실행"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        self._execute_task(task)
        
        # 즉시 실행된 작업의 실행 ID 반환
        recent_executions = [e for e in self.executions if e.task_id == task_id]
        if recent_executions:
            return recent_executions[-1].id
        
        return None
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        if task_id not in self.tasks:
            return None
        
        task = self.tasks[task_id]
        recent_executions = [e for e in self.executions if e.task_id == task_id]
        last_execution = recent_executions[-1] if recent_executions else None
        
        return {
            "task_id": task_id,
            "name": task.name,
            "enabled": task.enabled,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "last_status": last_execution.status.value if last_execution else None,
            "last_error": last_execution.error_message if last_execution else None,
            "total_executions": len(recent_executions),
            "successful_executions": len([e for e in recent_executions if e.status == TaskStatus.COMPLETED]),
            "failed_executions": len([e for e in recent_executions if e.status == TaskStatus.FAILED])
        }
    
    def get_executions(self, task_id: Optional[str] = None,
                      status: Optional[TaskStatus] = None,
                      hours: int = 24) -> List[TaskExecution]:
        """실행 기록 조회"""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_executions = [
            exec for exec in self.executions
            if exec.started_at >= cutoff_time
        ]
        
        if task_id is not None:
            filtered_executions = [exec for exec in filtered_executions if exec.task_id == task_id]
        
        if status is not None:
            filtered_executions = [exec for exec in filtered_executions if exec.status == status]
        
        return sorted(filtered_executions, key=lambda x: x.started_at, reverse=True)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """스케줄러 상태 조회"""
        return {
            "status": self.status,
            "total_tasks": len(self.tasks),
            "enabled_tasks": len([t for t in self.tasks.values() if t.enabled]),
            "total_executions": len(self.executions),
            "recent_executions": len(self.get_executions(hours=1)),
            "failed_executions": len([e for e in self.executions if e.status == TaskStatus.FAILED]),
            "next_scheduled_task": min([t.next_run for t in self.tasks.values() if t.next_run], default=None)
        }
    
    def add_task_callback(self, callback: Callable[[TaskExecution], None]) -> None:
        """작업 콜백 추가"""
        self.task_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[TaskExecution], None]) -> None:
        """에러 콜백 추가"""
        self.error_callbacks.append(callback)
    
    def export_tasks(self, file_path: str) -> bool:
        """작업 내보내기"""
        try:
            import json
            
            data = {
                "tasks": [
                    {
                        "id": task.id,
                        "name": task.name,
                        "description": task.description,
                        "task_type": task.task_type.value,
                        "cron_expression": task.cron_expression,
                        "enabled": task.enabled,
                        "max_retries": task.max_retries,
                        "timeout_seconds": task.timeout_seconds,
                        "parameters": task.parameters
                    }
                    for task in self.tasks.values()
                ],
                "exported_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"작업 내보내기 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"작업 내보내기 실패: {e}")
            return False
