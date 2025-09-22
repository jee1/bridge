"""감사 로그 시스템

모든 데이터 접근 및 변경 이력을 기록하고 관리합니다.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """감사 이벤트 타입"""
    # 데이터 접근
    DATA_READ = "data_read"
    DATA_WRITE = "data_write"
    DATA_DELETE = "data_delete"
    
    # 시스템 접근
    LOGIN = "login"
    LOGOUT = "logout"
    TOKEN_CREATE = "token_create"
    TOKEN_REVOKE = "token_revoke"
    
    # 권한 관리
    ROLE_ASSIGN = "role_assign"
    ROLE_REMOVE = "role_remove"
    PERMISSION_GRANT = "permission_grant"
    PERMISSION_REVOKE = "permission_revoke"
    
    # 데이터 관리
    CONTRACT_CREATE = "contract_create"
    CONTRACT_UPDATE = "contract_update"
    CONTRACT_DELETE = "contract_delete"
    
    # 메타데이터 관리
    METADATA_CREATE = "metadata_create"
    METADATA_UPDATE = "metadata_update"
    METADATA_DELETE = "metadata_delete"
    
    # 분석 실행
    ANALYSIS_START = "analysis_start"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_FAILED = "analysis_failed"
    
    # 시스템 관리
    SYSTEM_CONFIG = "system_config"
    SYSTEM_ERROR = "system_error"


class AuditSeverity(Enum):
    """감사 심각도"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """감사 이벤트"""
    id: str
    event_type: AuditEventType
    user_id: Optional[str] = None
    username: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    severity: AuditSeverity = AuditSeverity.LOW
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = None
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """딕셔너리에서 생성"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        
        if 'event_type' in data and isinstance(data['event_type'], str):
            data['event_type'] = AuditEventType(data['event_type'])
        
        if 'severity' in data and isinstance(data['severity'], str):
            data['severity'] = AuditSeverity(data['severity'])
        
        return cls(**data)


class AuditLogger:
    """감사 로거"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.events: List[AuditEvent] = []
        self.logger = logging.getLogger(__name__)
        
        # 파일 로거 설정
        if log_file:
            self._setup_file_logger()
    
    def _setup_file_logger(self):
        """파일 로거 설정"""
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def log_event(self, event: AuditEvent) -> bool:
        """감사 이벤트 로깅"""
        try:
            # 메모리에 저장
            self.events.append(event)
            
            # 파일에 로깅
            if self.log_file:
                self.logger.info(f"AUDIT: {json.dumps(event.to_dict(), ensure_ascii=False)}")
            
            # 심각도에 따른 추가 처리
            if event.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]:
                self._handle_high_severity_event(event)
            
            return True
        except Exception as e:
            self.logger.error(f"감사 이벤트 로깅 실패: {e}")
            return False
    
    def _handle_high_severity_event(self, event: AuditEvent):
        """고심각도 이벤트 처리"""
        # 알림 발송, 추가 로깅 등
        self.logger.warning(f"HIGH SEVERITY AUDIT EVENT: {event.event_type.value} - {event.details}")
    
    def log_data_access(self, user_id: str, username: str, resource_type: str, 
                       resource_id: str, action: str, success: bool = True,
                       details: Optional[Dict[str, Any]] = None) -> bool:
        """데이터 접근 로깅"""
        event_type = AuditEventType.DATA_READ
        if action.lower() in ['write', 'update', 'create']:
            event_type = AuditEventType.DATA_WRITE
        elif action.lower() in ['delete', 'remove']:
            event_type = AuditEventType.DATA_DELETE
        
        event = AuditEvent(
            id=self._generate_event_id(),
            event_type=event_type,
            user_id=user_id,
            username=username,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=details,
            severity=AuditSeverity.MEDIUM,
            success=success
        )
        
        return self.log_event(event)
    
    def log_authentication(self, user_id: str, username: str, action: str,
                          success: bool = True, ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None) -> bool:
        """인증 로깅"""
        event_type = AuditEventType.LOGIN if action.lower() == 'login' else AuditEventType.LOGOUT
        
        event = AuditEvent(
            id=self._generate_event_id(),
            event_type=event_type,
            user_id=user_id,
            username=username,
            action=action,
            severity=AuditSeverity.HIGH if not success else AuditSeverity.MEDIUM,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success
        )
        
        return self.log_event(event)
    
    def log_permission_change(self, user_id: str, username: str, target_user_id: str,
                             action: str, role_id: Optional[str] = None,
                             permission_id: Optional[str] = None) -> bool:
        """권한 변경 로깅"""
        if 'role' in action.lower():
            event_type = AuditEventType.ROLE_ASSIGN if 'assign' in action.lower() else AuditEventType.ROLE_REMOVE
        else:
            event_type = AuditEventType.PERMISSION_GRANT if 'grant' in action.lower() else AuditEventType.PERMISSION_REVOKE
        
        details = {
            "target_user_id": target_user_id,
            "role_id": role_id,
            "permission_id": permission_id
        }
        
        event = AuditEvent(
            id=self._generate_event_id(),
            event_type=event_type,
            user_id=user_id,
            username=username,
            action=action,
            details=details,
            severity=AuditSeverity.HIGH
        )
        
        return self.log_event(event)
    
    def log_system_event(self, event_type: AuditEventType, details: Optional[Dict[str, Any]] = None,
                        severity: AuditSeverity = AuditSeverity.MEDIUM,
                        success: bool = True, error_message: Optional[str] = None) -> bool:
        """시스템 이벤트 로깅"""
        event = AuditEvent(
            id=self._generate_event_id(),
            event_type=event_type,
            details=details,
            severity=severity,
            success=success,
            error_message=error_message
        )
        
        return self.log_event(event)
    
    def _generate_event_id(self) -> str:
        """이벤트 ID 생성"""
        import uuid
        return str(uuid.uuid4())
    
    def get_events(self, user_id: Optional[str] = None, 
                  event_type: Optional[AuditEventType] = None,
                  start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  severity: Optional[AuditSeverity] = None,
                  limit: int = 100) -> List[AuditEvent]:
        """감사 이벤트 조회"""
        filtered_events = self.events
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        # 시간순 정렬 (최신순)
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_events[:limit]
    
    def get_user_activity(self, user_id: str, days: int = 30) -> List[AuditEvent]:
        """사용자 활동 조회"""
        from datetime import timedelta
        start_time = datetime.now() - timedelta(days=days)
        
        return self.get_events(
            user_id=user_id,
            start_time=start_time
        )
    
    def get_failed_events(self, hours: int = 24) -> List[AuditEvent]:
        """실패한 이벤트 조회"""
        from datetime import timedelta
        start_time = datetime.now() - timedelta(hours=hours)
        
        return self.get_events(
            start_time=start_time
        )
    
    def get_high_severity_events(self, hours: int = 24) -> List[AuditEvent]:
        """고심각도 이벤트 조회"""
        from datetime import timedelta
        start_time = datetime.now() - timedelta(hours=hours)
        
        return self.get_events(
            start_time=start_time,
            severity=AuditSeverity.HIGH
        ) + self.get_events(
            start_time=start_time,
            severity=AuditSeverity.CRITICAL
        )
    
    def export_audit_log(self, file_path: str, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> bool:
        """감사 로그 내보내기"""
        try:
            events = self.get_events(start_time=start_time, end_time=end_time)
            
            data = {
                "audit_events": [event.to_dict() for event in events],
                "exported_at": datetime.now().isoformat(),
                "total_events": len(events)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"감사 로그 내보내기 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"감사 로그 내보내기 실패: {e}")
            return False
    
    def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """감사 로그 요약"""
        from datetime import timedelta
        start_time = datetime.now() - timedelta(days=days)
        
        events = self.get_events(start_time=start_time)
        
        summary = {
            "total_events": len(events),
            "successful_events": len([e for e in events if e.success]),
            "failed_events": len([e for e in events if not e.success]),
            "events_by_type": {},
            "events_by_severity": {},
            "events_by_user": {},
            "high_severity_count": len([e for e in events if e.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]])
        }
        
        # 이벤트 타입별 통계
        for event in events:
            event_type = event.event_type.value
            summary["events_by_type"][event_type] = summary["events_by_type"].get(event_type, 0) + 1
        
        # 심각도별 통계
        for event in events:
            severity = event.severity.value
            summary["events_by_severity"][severity] = summary["events_by_severity"].get(severity, 0) + 1
        
        # 사용자별 통계
        for event in events:
            if event.user_id:
                summary["events_by_user"][event.user_id] = summary["events_by_user"].get(event.user_id, 0) + 1
        
        return summary


class AuditTrail:
    """감사 추적기"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
    
    def track_data_access(self, user_id: str, username: str, resource_type: str,
                         resource_id: str, action: str, **kwargs) -> bool:
        """데이터 접근 추적"""
        return self.audit_logger.log_data_access(
            user_id=user_id,
            username=username,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            **kwargs
        )
    
    def track_authentication(self, user_id: str, username: str, action: str, **kwargs) -> bool:
        """인증 추적"""
        return self.audit_logger.log_authentication(
            user_id=user_id,
            username=username,
            action=action,
            **kwargs
        )
    
    def track_permission_change(self, user_id: str, username: str, target_user_id: str,
                               action: str, **kwargs) -> bool:
        """권한 변경 추적"""
        return self.audit_logger.log_permission_change(
            user_id=user_id,
            username=username,
            target_user_id=target_user_id,
            action=action,
            **kwargs
        )
    
    def track_system_event(self, event_type: AuditEventType, **kwargs) -> bool:
        """시스템 이벤트 추적"""
        return self.audit_logger.log_system_event(
            event_type=event_type,
            **kwargs
        )
    
    def get_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """컴플라이언스 리포트 생성"""
        from datetime import timedelta
        start_time = datetime.now() - timedelta(days=days)
        
        events = self.audit_logger.get_events(start_time=start_time)
        
        # 데이터 접근 이벤트
        data_access_events = [e for e in events if e.event_type in [
            AuditEventType.DATA_READ, AuditEventType.DATA_WRITE, AuditEventType.DATA_DELETE
        ]]
        
        # 인증 이벤트
        auth_events = [e for e in events if e.event_type in [
            AuditEventType.LOGIN, AuditEventType.LOGOUT
        ]]
        
        # 권한 변경 이벤트
        permission_events = [e for e in events if e.event_type in [
            AuditEventType.ROLE_ASSIGN, AuditEventType.ROLE_REMOVE,
            AuditEventType.PERMISSION_GRANT, AuditEventType.PERMISSION_REVOKE
        ]]
        
        # 실패한 이벤트
        failed_events = [e for e in events if not e.success]
        
        report = {
            "period": f"{days} days",
            "total_events": len(events),
            "data_access_events": len(data_access_events),
            "authentication_events": len(auth_events),
            "permission_change_events": len(permission_events),
            "failed_events": len(failed_events),
            "compliance_score": self._calculate_compliance_score(events),
            "recommendations": self._generate_recommendations(events)
        }
        
        return report
    
    def _calculate_compliance_score(self, events: List[AuditEvent]) -> float:
        """컴플라이언스 점수 계산"""
        if not events:
            return 100.0
        
        total_events = len(events)
        failed_events = len([e for e in events if not e.success])
        high_severity_events = len([e for e in events if e.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]])
        
        # 기본 점수에서 실패 및 고심각도 이벤트에 따른 감점
        score = 100.0
        score -= (failed_events / total_events) * 30  # 실패 이벤트 30% 감점
        score -= (high_severity_events / total_events) * 20  # 고심각도 이벤트 20% 감점
        
        return max(0.0, score)
    
    def _generate_recommendations(self, events: List[AuditEvent]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 실패한 이벤트가 많은 경우
        failed_events = [e for e in events if not e.success]
        if len(failed_events) > len(events) * 0.1:  # 10% 이상 실패
            recommendations.append("실패한 이벤트가 많습니다. 시스템 상태를 점검하세요.")
        
        # 고심각도 이벤트가 있는 경우
        high_severity_events = [e for e in events if e.severity in [AuditSeverity.HIGH, AuditSeverity.CRITICAL]]
        if high_severity_events:
            recommendations.append("고심각도 이벤트가 발생했습니다. 즉시 조치가 필요합니다.")
        
        # 데이터 접근 패턴 분석
        data_access_events = [e for e in events if e.event_type in [
            AuditEventType.DATA_READ, AuditEventType.DATA_WRITE, AuditEventType.DATA_DELETE
        ]]
        
        if len(data_access_events) == 0:
            recommendations.append("데이터 접근 이벤트가 없습니다. 로깅 설정을 확인하세요.")
        
        return recommendations
