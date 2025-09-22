"""시각화 엔진

대시보드용 차트 및 시각화 컴포넌트를 제공합니다.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum

import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """차트 타입"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    GAUGE = "gauge"
    METRIC = "metric"
    TABLE = "table"


class LayoutType(Enum):
    """레이아웃 타입"""
    GRID = "grid"
    FLEX = "flex"
    STACK = "stack"
    CUSTOM = "custom"


@dataclass
class ChartConfig:
    """차트 설정"""
    chart_type: ChartType
    title: str
    width: int = 400
    height: int = 300
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color: Optional[str] = None
    show_legend: bool = True
    show_grid: bool = True
    theme: str = "default"
    custom_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_config is None:
            self.custom_config = {}


@dataclass
class LayoutConfig:
    """레이아웃 설정"""
    layout_type: LayoutType
    columns: int = 4
    rows: int = 3
    gap: int = 10
    padding: int = 20
    responsive: bool = True
    custom_css: Optional[str] = None


class ChartRenderer:
    """차트 렌더러"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
    
    def render_chart(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """차트 렌더링"""
        try:
            if config.chart_type == ChartType.LINE:
                return self._render_line_chart(data, config)
            elif config.chart_type == ChartType.BAR:
                return self._render_bar_chart(data, config)
            elif config.chart_type == ChartType.PIE:
                return self._render_pie_chart(data, config)
            elif config.chart_type == ChartType.SCATTER:
                return self._render_scatter_chart(data, config)
            elif config.chart_type == ChartType.HISTOGRAM:
                return self._render_histogram(data, config)
            elif config.chart_type == ChartType.BOX:
                return self._render_box_chart(data, config)
            elif config.chart_type == ChartType.HEATMAP:
                return self._render_heatmap(data, config)
            elif config.chart_type == ChartType.GAUGE:
                return self._render_gauge(data, config)
            elif config.chart_type == ChartType.METRIC:
                return self._render_metric(data, config)
            elif config.chart_type == ChartType.TABLE:
                return self._render_table(data, config)
            else:
                raise ValueError(f"지원하지 않는 차트 타입: {config.chart_type}")
                
        except Exception as e:
            self.logger.error(f"차트 렌더링 실패: {e}")
            return {"error": str(e), "type": "error"}
    
    def _render_line_chart(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """선 차트 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=100)
        
        if config.x_axis and config.y_axis:
            ax.plot(df[config.x_axis], df[config.y_axis], 
                   color=config.color, linewidth=2, marker='o')
            ax.set_xlabel(config.x_axis)
            ax.set_ylabel(config.y_axis)
        else:
            # 첫 번째 숫자형 컬럼을 y축으로 사용
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ax.plot(df[numeric_cols[0]], linewidth=2, marker='o')
                ax.set_ylabel(numeric_cols[0])
        
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        ax.grid(config.show_grid, alpha=0.3)
        
        if config.show_legend:
            ax.legend()
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "type": "chart",
            "chart_type": "line",
            "title": config.title,
            "image": f"data:image/png;base64,{image_base64}",
            "width": config.width,
            "height": config.height
        }
    
    def _render_bar_chart(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """막대 차트 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=100)
        
        if config.x_axis and config.y_axis:
            ax.bar(df[config.x_axis], df[config.y_axis], color=config.color)
            ax.set_xlabel(config.x_axis)
            ax.set_ylabel(config.y_axis)
        else:
            # 첫 번째 숫자형 컬럼을 사용
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ax.bar(range(len(df)), df[numeric_cols[0]], color=config.color)
                ax.set_ylabel(numeric_cols[0])
        
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        ax.grid(config.show_grid, alpha=0.3)
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "type": "chart",
            "chart_type": "bar",
            "title": config.title,
            "image": f"data:image/png;base64,{image_base64}",
            "width": config.width,
            "height": config.height
        }
    
    def _render_pie_chart(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """파이 차트 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=100)
        
        if config.x_axis and config.y_axis:
            ax.pie(df[config.y_axis], labels=df[config.x_axis], autopct='%1.1f%%')
        else:
            # 첫 번째 숫자형 컬럼을 사용
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ax.pie(df[numeric_cols[0]], autopct='%1.1f%%')
        
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "type": "chart",
            "chart_type": "pie",
            "title": config.title,
            "image": f"data:image/png;base64,{image_base64}",
            "width": config.width,
            "height": config.height
        }
    
    def _render_scatter_chart(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """산점도 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=100)
        
        if config.x_axis and config.y_axis:
            ax.scatter(df[config.x_axis], df[config.y_axis], 
                      color=config.color, alpha=0.7)
            ax.set_xlabel(config.x_axis)
            ax.set_ylabel(config.y_axis)
        else:
            # 첫 두 개의 숫자형 컬럼을 사용
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], 
                          color=config.color, alpha=0.7)
                ax.set_xlabel(numeric_cols[0])
                ax.set_ylabel(numeric_cols[1])
        
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        ax.grid(config.show_grid, alpha=0.3)
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "type": "chart",
            "chart_type": "scatter",
            "title": config.title,
            "image": f"data:image/png;base64,{image_base64}",
            "width": config.width,
            "height": config.height
        }
    
    def _render_histogram(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """히스토그램 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=100)
        
        if config.y_axis:
            ax.hist(df[config.y_axis], bins=20, color=config.color, alpha=0.7)
            ax.set_ylabel(config.y_axis)
        else:
            # 첫 번째 숫자형 컬럼을 사용
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ax.hist(df[numeric_cols[0]], bins=20, color=config.color, alpha=0.7)
                ax.set_ylabel(numeric_cols[0])
        
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        ax.grid(config.show_grid, alpha=0.3)
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "type": "chart",
            "chart_type": "histogram",
            "title": config.title,
            "image": f"data:image/png;base64,{image_base64}",
            "width": config.width,
            "height": config.height
        }
    
    def _render_box_chart(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """박스 차트 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=100)
        
        if config.y_axis:
            ax.boxplot(df[config.y_axis])
            ax.set_ylabel(config.y_axis)
        else:
            # 첫 번째 숫자형 컬럼을 사용
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ax.boxplot(df[numeric_cols[0]])
                ax.set_ylabel(numeric_cols[0])
        
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        ax.grid(config.show_grid, alpha=0.3)
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "type": "chart",
            "chart_type": "box",
            "title": config.title,
            "image": f"data:image/png;base64,{image_base64}",
            "width": config.width,
            "height": config.height
        }
    
    def _render_heatmap(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """히트맵 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=100)
        
        # 숫자형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        
        ax.set_title(config.title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "type": "chart",
            "chart_type": "heatmap",
            "title": config.title,
            "image": f"data:image/png;base64,{image_base64}",
            "width": config.width,
            "height": config.height
        }
    
    def _render_gauge(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """게이지 차트 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        # 첫 번째 값 사용
        value = data[0].get('value', 0) if data else 0
        max_value = config.custom_config.get('max_value', 100)
        
        fig, ax = plt.subplots(figsize=(config.width/100, config.height/100), dpi=100)
        
        # 게이지 차트 생성
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        # 배경 원
        ax.plot(theta, r, 'k-', linewidth=2)
        
        # 값에 따른 색상
        if value < max_value * 0.5:
            color = 'green'
        elif value < max_value * 0.8:
            color = 'orange'
        else:
            color = 'red'
        
        # 값 표시
        value_theta = (value / max_value) * np.pi
        ax.plot([0, value_theta], [0, 1], color=color, linewidth=8)
        
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 1.2)
        ax.set_title(f"{config.title}\n{value:.1f}/{max_value}", fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        import io
        import base64
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return {
            "type": "chart",
            "chart_type": "gauge",
            "title": config.title,
            "image": f"data:image/png;base64,{image_base64}",
            "width": config.width,
            "height": config.height,
            "value": value,
            "max_value": max_value
        }
    
    def _render_metric(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """메트릭 카드 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        # 첫 번째 값 사용
        value = data[0].get('value', 0) if data else 0
        unit = data[0].get('unit', '') if data else ''
        
        # 값에 따른 색상 결정
        threshold_warning = config.custom_config.get('threshold_warning', 70)
        threshold_critical = config.custom_config.get('threshold_critical', 90)
        
        if value >= threshold_critical:
            color = '#dc3545'  # 빨간색
        elif value >= threshold_warning:
            color = '#ffc107'  # 노란색
        else:
            color = '#28a745'  # 초록색
        
        return {
            "type": "metric",
            "title": config.title,
            "value": value,
            "unit": unit,
            "color": color,
            "width": config.width,
            "height": config.height
        }
    
    def _render_table(self, data: List[Dict[str, Any]], config: ChartConfig) -> Dict[str, Any]:
        """테이블 렌더링"""
        if not data:
            return {"type": "empty", "message": "데이터가 없습니다"}
        
        df = pd.DataFrame(data)
        
        # 테이블을 HTML로 변환
        table_html = df.to_html(
            classes='table table-striped table-bordered',
            table_id=f"table_{config.title.replace(' ', '_')}",
            escape=False
        )
        
        return {
            "type": "table",
            "title": config.title,
            "html": table_html,
            "data": data,
            "width": config.width,
            "height": config.height
        }


class LayoutManager:
    """레이아웃 관리자"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_layout(self, widgets: List[Dict[str, Any]], config: LayoutConfig) -> Dict[str, Any]:
        """레이아웃 생성"""
        try:
            if config.layout_type == LayoutType.GRID:
                return self._create_grid_layout(widgets, config)
            elif config.layout_type == LayoutType.FLEX:
                return self._create_flex_layout(widgets, config)
            elif config.layout_type == LayoutType.STACK:
                return self._create_stack_layout(widgets, config)
            else:
                return self._create_custom_layout(widgets, config)
                
        except Exception as e:
            self.logger.error(f"레이아웃 생성 실패: {e}")
            return {"error": str(e), "type": "error"}
    
    def _create_grid_layout(self, widgets: List[Dict[str, Any]], config: LayoutConfig) -> Dict[str, Any]:
        """그리드 레이아웃 생성"""
        layout = {
            "type": "grid",
            "columns": config.columns,
            "rows": config.rows,
            "gap": config.gap,
            "padding": config.padding,
            "responsive": config.responsive,
            "widgets": []
        }
        
        for i, widget in enumerate(widgets):
            row = i // config.columns
            col = i % config.columns
            
            widget_layout = {
                "id": widget.get("id", f"widget_{i}"),
                "type": widget.get("type", "chart"),
                "title": widget.get("title", f"Widget {i+1}"),
                "position": {
                    "row": row,
                    "col": col,
                    "width": 1,
                    "height": 1
                },
                "content": widget.get("content", {})
            }
            
            layout["widgets"].append(widget_layout)
        
        return layout
    
    def _create_flex_layout(self, widgets: List[Dict[str, Any]], config: LayoutConfig) -> Dict[str, Any]:
        """플렉스 레이아웃 생성"""
        layout = {
            "type": "flex",
            "direction": "row",
            "gap": config.gap,
            "padding": config.padding,
            "responsive": config.responsive,
            "widgets": []
        }
        
        for i, widget in enumerate(widgets):
            widget_layout = {
                "id": widget.get("id", f"widget_{i}"),
                "type": widget.get("type", "chart"),
                "title": widget.get("title", f"Widget {i+1}"),
                "flex": 1,
                "content": widget.get("content", {})
            }
            
            layout["widgets"].append(widget_layout)
        
        return layout
    
    def _create_stack_layout(self, widgets: List[Dict[str, Any]], config: LayoutConfig) -> Dict[str, Any]:
        """스택 레이아웃 생성"""
        layout = {
            "type": "stack",
            "direction": "vertical",
            "gap": config.gap,
            "padding": config.padding,
            "responsive": config.responsive,
            "widgets": []
        }
        
        for i, widget in enumerate(widgets):
            widget_layout = {
                "id": widget.get("id", f"widget_{i}"),
                "type": widget.get("type", "chart"),
                "title": widget.get("title", f"Widget {i+1}"),
                "order": i,
                "content": widget.get("content", {})
            }
            
            layout["widgets"].append(widget_layout)
        
        return layout
    
    def _create_custom_layout(self, widgets: List[Dict[str, Any]], config: LayoutConfig) -> Dict[str, Any]:
        """커스텀 레이아웃 생성"""
        layout = {
            "type": "custom",
            "gap": config.gap,
            "padding": config.padding,
            "responsive": config.responsive,
            "custom_css": config.custom_css,
            "widgets": []
        }
        
        for i, widget in enumerate(widgets):
            widget_layout = {
                "id": widget.get("id", f"widget_{i}"),
                "type": widget.get("type", "chart"),
                "title": widget.get("title", f"Widget {i+1}"),
                "position": widget.get("position", {"x": 0, "y": 0, "width": 1, "height": 1}),
                "content": widget.get("content", {})
            }
            
            layout["widgets"].append(widget_layout)
        
        return layout


class VisualizationEngine:
    """시각화 엔진"""
    
    def __init__(self):
        self.chart_renderer = ChartRenderer()
        self.layout_manager = LayoutManager()
        self.logger = logging.getLogger(__name__)
    
    def render_dashboard(self, dashboard_config: Dict[str, Any], 
                        widgets_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """대시보드 렌더링"""
        try:
            dashboard_id = dashboard_config.get("id", "unknown")
            widgets = dashboard_config.get("widgets", [])
            
            rendered_widgets = []
            
            for widget in widgets:
                widget_id = widget.get("id")
                widget_type = widget.get("type", "chart")
                widget_config = widget.get("config", {})
                
                # 위젯 데이터 가져오기
                data = widgets_data.get(widget_id, [])
                
                # 위젯 렌더링
                if widget_type == "chart":
                    chart_config = ChartConfig(
                        chart_type=ChartType(widget_config.get("chart_type", "line")),
                        title=widget_config.get("title", "Chart"),
                        width=widget_config.get("width", 400),
                        height=widget_config.get("height", 300),
                        x_axis=widget_config.get("x_axis"),
                        y_axis=widget_config.get("y_axis"),
                        color=widget_config.get("color"),
                        show_legend=widget_config.get("show_legend", True),
                        show_grid=widget_config.get("show_grid", True),
                        theme=widget_config.get("theme", "default"),
                        custom_config=widget_config.get("custom_config", {})
                    )
                    
                    rendered_content = self.chart_renderer.render_chart(data, chart_config)
                
                elif widget_type == "metric":
                    metric_config = ChartConfig(
                        chart_type=ChartType.METRIC,
                        title=widget_config.get("title", "Metric"),
                        width=widget_config.get("width", 200),
                        height=widget_config.get("height", 100),
                        custom_config=widget_config.get("custom_config", {})
                    )
                    
                    rendered_content = self.chart_renderer.render_chart(data, metric_config)
                
                elif widget_type == "table":
                    table_config = ChartConfig(
                        chart_type=ChartType.TABLE,
                        title=widget_config.get("title", "Table"),
                        width=widget_config.get("width", 600),
                        height=widget_config.get("height", 400)
                    )
                    
                    rendered_content = self.chart_renderer.render_chart(data, table_config)
                
                else:
                    rendered_content = {"type": "error", "message": f"지원하지 않는 위젯 타입: {widget_type}"}
                
                rendered_widget = {
                    "id": widget_id,
                    "type": widget_type,
                    "title": widget.get("title", "Widget"),
                    "position": widget.get("position", {"x": 0, "y": 0, "width": 1, "height": 1}),
                    "content": rendered_content
                }
                
                rendered_widgets.append(rendered_widget)
            
            # 레이아웃 생성
            layout_config = LayoutConfig(
                layout_type=LayoutType(dashboard_config.get("layout_type", "grid")),
                columns=dashboard_config.get("grid_columns", 4),
                rows=dashboard_config.get("grid_rows", 3),
                gap=dashboard_config.get("gap", 10),
                padding=dashboard_config.get("padding", 20),
                responsive=dashboard_config.get("responsive", True)
            )
            
            layout = self.layout_manager.create_layout(rendered_widgets, layout_config)
            
            return {
                "dashboard_id": dashboard_id,
                "title": dashboard_config.get("name", "Dashboard"),
                "layout": layout,
                "rendered_at": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"대시보드 렌더링 실패: {e}")
            return {
                "dashboard_id": dashboard_config.get("id", "unknown"),
                "title": dashboard_config.get("name", "Dashboard"),
                "error": str(e),
                "rendered_at": datetime.now().isoformat(),
                "status": "error"
            }
    
    def create_sample_data(self, metric_type: str, count: int = 100) -> List[Dict[str, Any]]:
        """샘플 데이터 생성"""
        import random
        
        data = []
        base_time = datetime.now()
        
        for i in range(count):
            timestamp = base_time - timedelta(minutes=count-i)
            
            if metric_type == "cpu":
                value = random.uniform(20, 80)
                unit = "%"
            elif metric_type == "memory":
                value = random.uniform(30, 90)
                unit = "%"
            elif metric_type == "disk":
                value = random.uniform(40, 85)
                unit = "%"
            elif metric_type == "response_time":
                value = random.uniform(100, 500)
                unit = "ms"
            elif metric_type == "throughput":
                value = random.uniform(50, 200)
                unit = "req/s"
            else:
                value = random.uniform(0, 100)
                unit = ""
            
            data.append({
                "timestamp": timestamp.isoformat(),
                "value": value,
                "unit": unit,
                "metric_type": metric_type
            })
        
        return data
