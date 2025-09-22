"""알림 시스템

다양한 채널을 통해 알림을 발송하고 관리합니다.
"""

import logging
import smtplib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """알림 채널"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    FILE = "file"


class AlertPriority(Enum):
    """알림 우선순위"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """알림 규칙"""
    id: str
    name: str
    description: Optional[str] = None
    event_type: str = None
    conditions: Dict[str, Any] = None
    channels: List[str] = None
    priority: AlertPriority = AlertPriority.MEDIUM
    enabled: bool = True
    cooldown_minutes: int = 0  # 알림 간격 (분)
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
        if self.channels is None:
            self.channels = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class NotificationMessage:
    """알림 메시지"""
    id: str
    rule_id: str
    title: str
    message: str
    priority: AlertPriority
    channels: List[str]
    data: Optional[Dict[str, Any]] = None
    sent_at: Optional[datetime] = None
    status: str = "pending"  # pending, sent, failed
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.data is None:
            self.data = {}


@dataclass
class ChannelConfig:
    """채널 설정"""
    channel_type: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True


class NotificationSystem:
    """알림 시스템"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, ChannelConfig] = {}
        self.messages: List[NotificationMessage] = []
        self.last_notifications: Dict[str, datetime] = {}  # 규칙별 마지막 알림 시간
        self.logger = logging.getLogger(__name__)
        
        # 기본 채널 설정
        self._setup_default_channels()
    
    def _setup_default_channels(self):
        """기본 채널 설정"""
        # 콘솔 채널 (기본)
        self.channels["console"] = ChannelConfig(
            channel_type=NotificationChannel.CONSOLE,
            config={},
            enabled=True
        )
        
        # 파일 채널 (기본)
        self.channels["file"] = ChannelConfig(
            channel_type=NotificationChannel.FILE,
            config={"file_path": "notifications.log"},
            enabled=True
        )
    
    def add_rule(self, rule: AlertRule) -> bool:
        """알림 규칙 추가"""
        try:
            self.rules[rule.id] = rule
            self.logger.info(f"알림 규칙 추가 완료: {rule.id}")
            return True
        except Exception as e:
            self.logger.error(f"알림 규칙 추가 실패: {e}")
            return False
    
    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """알림 규칙 업데이트"""
        if rule_id not in self.rules:
            return False
        
        try:
            rule = self.rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key) and key not in ['id', 'created_at']:
                    setattr(rule, key, value)
            
            rule.updated_at = datetime.now()
            self.logger.info(f"알림 규칙 업데이트 완료: {rule_id}")
            return True
        except Exception as e:
            self.logger.error(f"알림 규칙 업데이트 실패: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """알림 규칙 제거"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"알림 규칙 제거 완료: {rule_id}")
            return True
        return False
    
    def add_channel(self, channel_id: str, channel_config: ChannelConfig) -> bool:
        """알림 채널 추가"""
        try:
            self.channels[channel_id] = channel_config
            self.logger.info(f"알림 채널 추가 완료: {channel_id}")
            return True
        except Exception as e:
            self.logger.error(f"알림 채널 추가 실패: {e}")
            return False
    
    def send_notification(self, event_type: str, data: Dict[str, Any]) -> List[str]:
        """알림 발송"""
        sent_messages = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if not self._matches_rule(rule, event_type, data):
                continue
            
            if not self._check_cooldown(rule):
                continue
            
            message = self._create_message(rule, data)
            if message:
                success = self._send_message(message)
                if success:
                    sent_messages.append(message.id)
                    self.last_notifications[rule.id] = datetime.now()
        
        return sent_messages
    
    def _matches_rule(self, rule: AlertRule, event_type: str, data: Dict[str, Any]) -> bool:
        """규칙 매칭 확인"""
        if rule.event_type != event_type:
            return False
        
        # 조건 확인
        for key, expected_value in rule.conditions.items():
            if key not in data:
                return False
            
            actual_value = data[key]
            if isinstance(expected_value, dict):
                # 복잡한 조건 처리
                if "operator" in expected_value:
                    operator = expected_value["operator"]
                    threshold = expected_value["value"]
                    
                    if operator == "gt" and actual_value <= threshold:
                        return False
                    elif operator == "lt" and actual_value >= threshold:
                        return False
                    elif operator == "eq" and actual_value != threshold:
                        return False
                    elif operator == "gte" and actual_value < threshold:
                        return False
                    elif operator == "lte" and actual_value > threshold:
                        return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True
    
    def _check_cooldown(self, rule: AlertRule) -> bool:
        """쿨다운 확인"""
        if rule.cooldown_minutes <= 0:
            return True
        
        last_notification = self.last_notifications.get(rule.id)
        if not last_notification:
            return True
        
        from datetime import timedelta
        cooldown_end = last_notification + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() >= cooldown_end
    
    def _create_message(self, rule: AlertRule, data: Dict[str, Any]) -> Optional[NotificationMessage]:
        """알림 메시지 생성"""
        import uuid
        
        # 메시지 템플릿 생성
        title = self._format_template(rule.name, data)
        message = self._format_template(rule.description or "알림이 발생했습니다.", data)
        
        notification = NotificationMessage(
            id=str(uuid.uuid4()),
            rule_id=rule.id,
            title=title,
            message=message,
            priority=rule.priority,
            channels=rule.channels.copy(),
            data=data
        )
        
        self.messages.append(notification)
        return notification
    
    def _format_template(self, template: str, data: Dict[str, Any]) -> str:
        """템플릿 포맷팅"""
        try:
            return template.format(**data)
        except Exception as e:
            self.logger.error(f"템플릿 포맷팅 실패: {e}")
            return template
    
    def _send_message(self, message: NotificationMessage) -> bool:
        """메시지 발송"""
        success_count = 0
        total_channels = len(message.channels)
        
        for channel_id in message.channels:
            if channel_id not in self.channels:
                self.logger.warning(f"채널을 찾을 수 없습니다: {channel_id}")
                continue
            
            channel_config = self.channels[channel_id]
            if not channel_config.enabled:
                continue
            
            try:
                if self._send_to_channel(message, channel_config):
                    success_count += 1
            except Exception as e:
                self.logger.error(f"채널 {channel_id} 발송 실패: {e}")
                message.error_message = str(e)
        
        if success_count > 0:
            message.status = "sent"
            message.sent_at = datetime.now()
            self.logger.info(f"알림 발송 완료: {message.id} ({success_count}/{total_channels})")
            return True
        else:
            message.status = "failed"
            self.logger.error(f"알림 발송 실패: {message.id}")
            return False
    
    def _send_to_channel(self, message: NotificationMessage, channel_config: ChannelConfig) -> bool:
        """채널별 메시지 발송"""
        channel_type = channel_config.channel_type
        config = channel_config.config
        
        if channel_type == NotificationChannel.CONSOLE:
            return self._send_to_console(message)
        elif channel_type == NotificationChannel.FILE:
            return self._send_to_file(message, config)
        elif channel_type == NotificationChannel.EMAIL:
            return self._send_to_email(message, config)
        elif channel_type == NotificationChannel.SLACK:
            return self._send_to_slack(message, config)
        elif channel_type == NotificationChannel.WEBHOOK:
            return self._send_to_webhook(message, config)
        else:
            self.logger.warning(f"지원하지 않는 채널 타입: {channel_type}")
            return False
    
    def _send_to_console(self, message: NotificationMessage) -> bool:
        """콘솔에 발송"""
        print(f"\n=== 알림 ===")
        print(f"제목: {message.title}")
        print(f"메시지: {message.message}")
        print(f"우선순위: {message.priority.value}")
        print(f"시간: {message.sent_at or datetime.now()}")
        print("=" * 20)
        return True
    
    def _send_to_file(self, message: NotificationMessage, config: Dict[str, Any]) -> bool:
        """파일로 발송"""
        file_path = config.get("file_path", "notifications.log")
        
        log_entry = {
            "timestamp": (message.sent_at or datetime.now()).isoformat(),
            "rule_id": message.rule_id,
            "title": message.title,
            "message": message.message,
            "priority": message.priority.value,
            "data": message.data
        }
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        return True
    
    def _send_to_email(self, message: NotificationMessage, config: Dict[str, Any]) -> bool:
        """이메일로 발송"""
        smtp_server = config.get("smtp_server")
        smtp_port = config.get("smtp_port", 587)
        username = config.get("username")
        password = config.get("password")
        from_email = config.get("from_email")
        to_emails = config.get("to_emails", [])
        
        if not all([smtp_server, username, password, from_email, to_emails]):
            self.logger.error("이메일 설정이 불완전합니다")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{message.priority.value.upper()}] {message.title}"
            
            body = f"""
            {message.message}
            
            우선순위: {message.priority.value}
            시간: {message.sent_at or datetime.now()}
            
            추가 정보:
            {json.dumps(message.data, ensure_ascii=False, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            self.logger.error(f"이메일 발송 실패: {e}")
            return False
    
    def _send_to_slack(self, message: NotificationMessage, config: Dict[str, Any]) -> bool:
        """Slack으로 발송"""
        webhook_url = config.get("webhook_url")
        channel = config.get("channel", "#general")
        
        if not webhook_url:
            self.logger.error("Slack 웹훅 URL이 설정되지 않았습니다")
            return False
        
        try:
            import requests
            
            payload = {
                "channel": channel,
                "username": "Bridge Analytics",
                "icon_emoji": ":chart_with_upwards_trend:",
                "attachments": [{
                    "color": self._get_priority_color(message.priority),
                    "title": message.title,
                    "text": message.message,
                    "fields": [
                        {
                            "title": "우선순위",
                            "value": message.priority.value,
                            "short": True
                        },
                        {
                            "title": "시간",
                            "value": (message.sent_at or datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
                            "short": True
                        }
                    ],
                    "footer": "Bridge Analytics",
                    "ts": int((message.sent_at or datetime.now()).timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            self.logger.error(f"Slack 발송 실패: {e}")
            return False
    
    def _send_to_webhook(self, message: NotificationMessage, config: Dict[str, Any]) -> bool:
        """웹훅으로 발송"""
        webhook_url = config.get("url")
        headers = config.get("headers", {})
        
        if not webhook_url:
            self.logger.error("웹훅 URL이 설정되지 않았습니다")
            return False
        
        try:
            import requests
            
            payload = {
                "title": message.title,
                "message": message.message,
                "priority": message.priority.value,
                "timestamp": (message.sent_at or datetime.now()).isoformat(),
                "data": message.data
            }
            
            response = requests.post(webhook_url, json=payload, headers=headers)
            return response.status_code in [200, 201, 202]
            
        except Exception as e:
            self.logger.error(f"웹훅 발송 실패: {e}")
            return False
    
    def _get_priority_color(self, priority: AlertPriority) -> str:
        """우선순위별 색상"""
        color_map = {
            AlertPriority.LOW: "good",
            AlertPriority.MEDIUM: "warning",
            AlertPriority.HIGH: "danger",
            AlertPriority.CRITICAL: "#8B0000"
        }
        return color_map.get(priority, "good")
    
    def get_messages(self, rule_id: Optional[str] = None,
                    status: Optional[str] = None,
                    priority: Optional[AlertPriority] = None,
                    hours: int = 24) -> List[NotificationMessage]:
        """알림 메시지 조회"""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_messages = [
            msg for msg in self.messages
            if msg.sent_at is None or msg.sent_at >= cutoff_time
        ]
        
        if rule_id is not None:
            filtered_messages = [msg for msg in filtered_messages if msg.rule_id == rule_id]
        
        if status is not None:
            filtered_messages = [msg for msg in filtered_messages if msg.status == status]
        
        if priority is not None:
            filtered_messages = [msg for msg in filtered_messages if msg.priority == priority]
        
        return sorted(filtered_messages, key=lambda x: x.sent_at or datetime.min, reverse=True)
    
    def get_notification_stats(self, hours: int = 24) -> Dict[str, Any]:
        """알림 통계 조회"""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_messages = [
            msg for msg in self.messages
            if msg.sent_at is None or msg.sent_at >= cutoff_time
        ]
        
        stats = {
            "total_messages": len(recent_messages),
            "sent_messages": len([msg for msg in recent_messages if msg.status == "sent"]),
            "failed_messages": len([msg for msg in recent_messages if msg.status == "failed"]),
            "pending_messages": len([msg for msg in recent_messages if msg.status == "pending"]),
            "messages_by_priority": {},
            "messages_by_rule": {}
        }
        
        # 우선순위별 통계
        for priority in AlertPriority:
            count = len([msg for msg in recent_messages if msg.priority == priority])
            stats["messages_by_priority"][priority.value] = count
        
        # 규칙별 통계
        for rule_id in set(msg.rule_id for msg in recent_messages):
            count = len([msg for msg in recent_messages if msg.rule_id == rule_id])
            stats["messages_by_rule"][rule_id] = count
        
        return stats
    
    def test_notification(self, channel_id: str, test_message: str = "테스트 알림입니다.") -> bool:
        """알림 테스트"""
        if channel_id not in self.channels:
            self.logger.error(f"채널을 찾을 수 없습니다: {channel_id}")
            return False
        
        import uuid
        
        test_msg = NotificationMessage(
            id=str(uuid.uuid4()),
            rule_id="test",
            title="테스트 알림",
            message=test_message,
            priority=AlertPriority.LOW,
            channels=[channel_id]
        )
        
        channel_config = self.channels[channel_id]
        return self._send_to_channel(test_msg, channel_config)
    
    def export_configuration(self, file_path: str) -> bool:
        """설정 내보내기"""
        try:
            data = {
                "rules": [asdict(rule) for rule in self.rules.values()],
                "channels": {
                    channel_id: {
                        "channel_type": config.channel_type.value,
                        "config": config.config,
                        "enabled": config.enabled
                    }
                    for channel_id, config in self.channels.items()
                },
                "exported_at": datetime.now().isoformat()
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            
            self.logger.info(f"설정 내보내기 완료: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"설정 내보내기 실패: {e}")
            return False
