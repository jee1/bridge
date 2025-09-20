"""로깅 설정 모듈."""
from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Dict, Any


def setup_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    """로깅을 설정합니다.
    
    Args:
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 로그 파일 경로 (None이면 콘솔에만 출력)
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # python-json-logger가 없으면 기본 formatter 사용
    try:
        import pythonjsonlogger
        json_formatter_class = "pythonjsonlogger.jsonlogger.JsonFormatter"
    except ImportError:
        json_formatter_class = "logging.Formatter"
    
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "detailed": {
                "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "json": {
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                "class": json_formatter_class
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "formatter": "standard",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": log_level,
                "formatter": "detailed",
                "filename": str(log_dir / "bridge.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "detailed",
                "filename": str(log_dir / "bridge_error.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "audit_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": str(log_dir / "audit.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 10
            }
        },
        "loggers": {
            "bridge": {
                "level": log_level,
                "handlers": ["console", "file", "error_file"],
                "propagate": False
            },
            "bridge.audit": {
                "level": "INFO",
                "handlers": ["audit_file"],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console"],
                "propagate": False
            },
            "celery": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False
            }
        },
        "root": {
            "level": "WARNING",
            "handlers": ["console"]
        }
    }
    
    # 로그 파일이 지정된 경우 추가
    if log_file:
        config["handlers"]["custom_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level,
            "formatter": "detailed",
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        config["loggers"]["bridge"]["handlers"].append("custom_file")
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """로거를 가져옵니다.
    
    Args:
        name: 로거 이름
        
    Returns:
        logging.Logger: 설정된 로거
    """
    return logging.getLogger(f"bridge.{name}")
