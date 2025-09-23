"""대시보드 관리자

대시보드 생성, 구성, 관리 기능을 제공합니다.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class WidgetType(Enum):
    """위젯 타입"""

    CHART = "chart"
    METRIC = "metric"
    TABLE = "table"
    ALERT = "alert"
    LOG = "log"
    CUSTOM = "custom"


class LayoutType(Enum):
    """레이아웃 타입"""

    GRID = "grid"
    FLEX = "flex"
    CUSTOM = "custom"


@dataclass
class DashboardConfig:
    """대시보드 설정"""

    id: str
    name: str
    description: Optional[str] = None
    layout_type: LayoutType = LayoutType.GRID
    grid_columns: int = 4
    grid_rows: int = 3
    auto_refresh: bool = True
    refresh_interval: int = 30  # 초
    theme: str = "default"
    created_at: datetime = None
    updated_at: datetime = None
    created_by: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


@dataclass
class DashboardWidget:
    """대시보드 위젯"""

    id: str
    widget_type: WidgetType
    title: str
    position: Dict[str, int]  # {"x": 0, "y": 0, "width": 1, "height": 1}
    config: Dict[str, Any]
    data_source: Optional[str] = None
    refresh_interval: Optional[int] = None
    enabled: bool = True
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class DashboardManager:
    """대시보드 관리자"""

    def __init__(self, storage_path: str = "data/dashboards"):
        self.storage_path = storage_path
        self.dashboards: Dict[str, Dict[str, Any]] = {}
        self.widgets: Dict[str, DashboardWidget] = {}
        self.logger = logging.getLogger(__name__)

        # 대시보드 로드
        self._load_dashboards()

    def _load_dashboards(self):
        """대시보드 로드"""
        import os

        os.makedirs(self.storage_path, exist_ok=True)

        # 실제 구현에서는 파일에서 로드
        # 여기서는 기본 대시보드 생성
        self._create_default_dashboards()

    def _create_default_dashboards(self):
        """기본 대시보드 생성"""
        # 시스템 모니터링 대시보드
        system_dashboard = {
            "id": "system_monitoring",
            "name": "시스템 모니터링",
            "description": "시스템 상태 및 성능 모니터링",
            "layout_type": LayoutType.GRID.value,
            "grid_columns": 4,
            "grid_rows": 3,
            "auto_refresh": True,
            "refresh_interval": 30,
            "theme": "default",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "created_by": "system",
        }

        self.dashboards["system_monitoring"] = system_dashboard

        # 데이터 품질 대시보드
        quality_dashboard = {
            "id": "data_quality",
            "name": "데이터 품질",
            "description": "데이터 품질 모니터링 및 알림",
            "layout_type": LayoutType.GRID.value,
            "grid_columns": 3,
            "grid_rows": 2,
            "auto_refresh": True,
            "refresh_interval": 60,
            "theme": "default",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "created_by": "system",
        }

        self.dashboards["data_quality"] = quality_dashboard

        # 분석 리포트 대시보드
        analytics_dashboard = {
            "id": "analytics_reports",
            "name": "분석 리포트",
            "description": "자동 생성된 분석 리포트 및 차트",
            "layout_type": LayoutType.GRID.value,
            "grid_columns": 2,
            "grid_rows": 2,
            "auto_refresh": False,
            "refresh_interval": 300,
            "theme": "default",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "created_by": "system",
        }

        self.dashboards["analytics_reports"] = analytics_dashboard

    def create_dashboard(self, config: DashboardConfig) -> bool:
        """대시보드 생성"""
        try:
            dashboard_data = {
                "id": config.id,
                "name": config.name,
                "description": config.description,
                "layout_type": config.layout_type.value,
                "grid_columns": config.grid_columns,
                "grid_rows": config.grid_rows,
                "auto_refresh": config.auto_refresh,
                "refresh_interval": config.refresh_interval,
                "theme": config.theme,
                "created_at": config.created_at.isoformat(),
                "updated_at": config.updated_at.isoformat(),
                "created_by": config.created_by,
                "widgets": [],
            }

            self.dashboards[config.id] = dashboard_data
            self._save_dashboard(config.id)

            self.logger.info(f"대시보드 생성 완료: {config.id}")
            return True

        except Exception as e:
            self.logger.error(f"대시보드 생성 실패: {e}")
            return False

    def get_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """대시보드 조회"""
        return self.dashboards.get(dashboard_id)

    def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any]) -> bool:
        """대시보드 업데이트"""
        if dashboard_id not in self.dashboards:
            return False

        try:
            dashboard = self.dashboards[dashboard_id]
            for key, value in updates.items():
                if key in dashboard and key not in ["id", "created_at"]:
                    dashboard[key] = value

            dashboard["updated_at"] = datetime.now().isoformat()
            self._save_dashboard(dashboard_id)

            self.logger.info(f"대시보드 업데이트 완료: {dashboard_id}")
            return True

        except Exception as e:
            self.logger.error(f"대시보드 업데이트 실패: {e}")
            return False

    def delete_dashboard(self, dashboard_id: str) -> bool:
        """대시보드 삭제"""
        if dashboard_id not in self.dashboards:
            return False

        try:
            # 관련 위젯들도 삭제
            dashboard_widgets = [w for w in self.widgets.values() if w.data_source == dashboard_id]
            for widget in dashboard_widgets:
                del self.widgets[widget.id]

            del self.dashboards[dashboard_id]
            self._delete_dashboard_file(dashboard_id)

            self.logger.info(f"대시보드 삭제 완료: {dashboard_id}")
            return True

        except Exception as e:
            self.logger.error(f"대시보드 삭제 실패: {e}")
            return False

    def list_dashboards(self) -> List[Dict[str, Any]]:
        """대시보드 목록 조회"""
        return list(self.dashboards.values())

    def add_widget(self, widget: DashboardWidget, dashboard_id: str) -> bool:
        """위젯 추가"""
        try:
            if dashboard_id not in self.dashboards:
                return False

            self.widgets[widget.id] = widget

            # 대시보드에 위젯 추가
            if "widgets" not in self.dashboards[dashboard_id]:
                self.dashboards[dashboard_id]["widgets"] = []

            self.dashboards[dashboard_id]["widgets"].append(widget.id)
            self._save_dashboard(dashboard_id)

            self.logger.info(f"위젯 추가 완료: {widget.id}")
            return True

        except Exception as e:
            self.logger.error(f"위젯 추가 실패: {e}")
            return False

    def remove_widget(self, widget_id: str, dashboard_id: str) -> bool:
        """위젯 제거"""
        try:
            if widget_id not in self.widgets or dashboard_id not in self.dashboards:
                return False

            del self.widgets[widget_id]

            # 대시보드에서 위젯 제거
            if "widgets" in self.dashboards[dashboard_id]:
                self.dashboards[dashboard_id]["widgets"] = [
                    w for w in self.dashboards[dashboard_id]["widgets"] if w != widget_id
                ]

            self._save_dashboard(dashboard_id)

            self.logger.info(f"위젯 제거 완료: {widget_id}")
            return True

        except Exception as e:
            self.logger.error(f"위젯 제거 실패: {e}")
            return False

    def get_widget(self, widget_id: str) -> Optional[DashboardWidget]:
        """위젯 조회"""
        return self.widgets.get(widget_id)

    def get_dashboard_widgets(self, dashboard_id: str) -> List[DashboardWidget]:
        """대시보드의 위젯 목록 조회"""
        if dashboard_id not in self.dashboards:
            return []

        widget_ids = self.dashboards[dashboard_id].get("widgets", [])
        return [self.widgets[widget_id] for widget_id in widget_ids if widget_id in self.widgets]

    def update_widget(self, widget_id: str, updates: Dict[str, Any]) -> bool:
        """위젯 업데이트"""
        if widget_id not in self.widgets:
            return False

        try:
            widget = self.widgets[widget_id]
            for key, value in updates.items():
                if hasattr(widget, key) and key not in ["id", "created_at"]:
                    setattr(widget, key, value)

            widget.updated_at = datetime.now()

            self.logger.info(f"위젯 업데이트 완료: {widget_id}")
            return True

        except Exception as e:
            self.logger.error(f"위젯 업데이트 실패: {e}")
            return False

    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """대시보드 데이터 조회 (위젯 포함)"""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return None

        widgets = self.get_dashboard_widgets(dashboard_id)
        dashboard_data = dashboard.copy()
        dashboard_data["widgets"] = [asdict(widget) for widget in widgets]

        return dashboard_data

    def _save_dashboard(self, dashboard_id: str):
        """대시보드 저장"""
        import os

        dashboard = self.dashboards[dashboard_id]
        file_path = os.path.join(self.storage_path, f"{dashboard_id}.json")

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(dashboard, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"대시보드 저장 실패: {e}")

    def _delete_dashboard_file(self, dashboard_id: str):
        """대시보드 파일 삭제"""
        import os

        file_path = os.path.join(self.storage_path, f"{dashboard_id}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                self.logger.error(f"대시보드 파일 삭제 실패: {e}")

    def export_dashboard(self, dashboard_id: str, file_path: str) -> bool:
        """대시보드 내보내기"""
        try:
            dashboard_data = self.get_dashboard_data(dashboard_id)
            if not dashboard_data:
                return False

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(dashboard_data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"대시보드 내보내기 완료: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"대시보드 내보내기 실패: {e}")
            return False

    def import_dashboard(self, file_path: str) -> bool:
        """대시보드 가져오기"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                dashboard_data = json.load(f)

            dashboard_id = dashboard_data["id"]
            self.dashboards[dashboard_id] = dashboard_data

            # 위젯들도 복원
            for widget_data in dashboard_data.get("widgets", []):
                widget = DashboardWidget(
                    id=widget_data["id"],
                    widget_type=WidgetType(widget_data["widget_type"]),
                    title=widget_data["title"],
                    position=widget_data["position"],
                    config=widget_data["config"],
                    data_source=widget_data.get("data_source"),
                    refresh_interval=widget_data.get("refresh_interval"),
                    enabled=widget_data.get("enabled", True),
                    created_at=datetime.fromisoformat(widget_data["created_at"]),
                    updated_at=datetime.fromisoformat(widget_data["updated_at"]),
                )
                self.widgets[widget.id] = widget

            self._save_dashboard(dashboard_id)

            self.logger.info(f"대시보드 가져오기 완료: {dashboard_id}")
            return True

        except Exception as e:
            self.logger.error(f"대시보드 가져오기 실패: {e}")
            return False

    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """대시보드 통계 조회"""
        return {
            "total_dashboards": len(self.dashboards),
            "total_widgets": len(self.widgets),
            "widgets_by_type": {
                widget_type.value: len(
                    [w for w in self.widgets.values() if w.widget_type == widget_type]
                )
                for widget_type in WidgetType
            },
            "dashboards_by_theme": {
                theme: len([d for d in self.dashboards.values() if d.get("theme") == theme])
                for theme in set(d.get("theme", "default") for d in self.dashboards.values())
            },
        }
