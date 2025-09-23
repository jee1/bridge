"""고급 시각화 모듈.

CA 마일스톤 3.2: 고급 통계 분석 및 시각화
고급 시각화 기능을 제공하는 모듈입니다.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import seaborn as sns
from matplotlib.figure import Figure
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from bridge.analytics.core.data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


class AdvancedVisualization:
    """고급 시각화 클래스.
    
    고급 시각화 기능을 제공합니다.
    """

    def __init__(self, style: str = "seaborn", figsize: Tuple[int, int] = (10, 6)):
        """AdvancedVisualization을 초기화합니다.
        
        Args:
            style: 시각화 스타일 ("seaborn", "matplotlib", "plotly")
            figsize: 기본 그림 크기
        """
        self.style = style
        self.figsize = figsize
        self._setup_style()

    def _setup_style(self) -> None:
        """시각화 스타일을 설정합니다."""
        if self.style == "seaborn":
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        elif self.style == "matplotlib":
            plt.style.use('default')
        elif self.style == "plotly":
            # Plotly는 별도 설정 불필요
            pass

    def create_interactive_dashboard(
        self,
        data: UnifiedDataFrame,
        chart_configs: List[Dict[str, Any]],
        title: str = "Interactive Dashboard",
        layout: str = "grid"
    ) -> Dict[str, Any]:
        """인터랙티브 대시보드를 생성합니다.
        
        Args:
            data: 시각화할 데이터
            chart_configs: 차트 설정 목록
            title: 대시보드 제목
            layout: 레이아웃 ("grid", "vertical", "horizontal")
            
        Returns:
            Dict[str, Any]: 대시보드 정보
        """
        df = data.to_pandas()
        
        if self.style == "plotly":
            return self._create_plotly_dashboard(df, chart_configs, title, layout)
        else:
            return self._create_matplotlib_dashboard(df, chart_configs, title, layout)

    def create_advanced_chart(
        self,
        data: UnifiedDataFrame,
        chart_type: str,
        x_column: str,
        y_column: Optional[str] = None,
        hue_column: Optional[str] = None,
        title: str = "",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """고급 차트를 생성합니다.
        
        Args:
            data: 시각화할 데이터
            chart_type: 차트 유형
            x_column: X축 컬럼
            y_column: Y축 컬럼
            hue_column: 색상 구분 컬럼
            title: 차트 제목
            config: 추가 설정
            
        Returns:
            Dict[str, Any]: 차트 정보
        """
        df = data.to_pandas()
        
        if chart_type == "scatter_matrix":
            return self._create_scatter_matrix(df, title, config)
        elif chart_type == "heatmap":
            return self._create_heatmap(df, x_column, y_column, title, config)
        elif chart_type == "violin":
            return self._create_violin_plot(df, x_column, y_column, hue_column, title, config)
        elif chart_type == "box":
            return self._create_box_plot(df, x_column, y_column, hue_column, title, config)
        elif chart_type == "histogram":
            return self._create_histogram(df, x_column, hue_column, title, config)
        elif chart_type == "line":
            return self._create_line_plot(df, x_column, y_column, hue_column, title, config)
        elif chart_type == "bar":
            return self._create_bar_plot(df, x_column, y_column, hue_column, title, config)
        elif chart_type == "pie":
            return self._create_pie_chart(df, x_column, y_column, title, config)
        elif chart_type == "correlation":
            return self._create_correlation_plot(df, title, config)
        else:
            return {"error": f"지원하지 않는 차트 유형: {chart_type}"}

    def create_statistical_plots(
        self,
        data: UnifiedDataFrame,
        column: str,
        plot_types: List[str] = ["histogram", "qq", "box", "violin"]
    ) -> Dict[str, Any]:
        """통계적 플롯을 생성합니다.
        
        Args:
            data: 시각화할 데이터
            column: 분석할 컬럼
            plot_types: 생성할 플롯 유형들
            
        Returns:
            Dict[str, Any]: 통계적 플롯 정보
        """
        df = data.to_pandas()
        series = df[column].dropna()
        
        if len(series) == 0:
            return {"error": f"컬럼 '{column}'에 유효한 데이터가 없습니다."}
        
        plots = {}
        
        for plot_type in plot_types:
            if plot_type == "histogram":
                plots["histogram"] = self._create_histogram_plot(series)
            elif plot_type == "qq":
                plots["qq"] = self._create_qq_plot(series)
            elif plot_type == "box":
                plots["box"] = self._create_box_plot_single(series)
            elif plot_type == "violin":
                plots["violin"] = self._create_violin_plot_single(series)
            elif plot_type == "density":
                plots["density"] = self._create_density_plot(series)
            elif plot_type == "ecdf":
                plots["ecdf"] = self._create_ecdf_plot(series)
        
        return {
            "statistical_plots": plots,
            "column": column,
            "sample_size": len(series),
            "statistics": {
                "mean": series.mean(),
                "std": series.std(),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis()
            }
        }

    def create_time_series_plot(
        self,
        data: UnifiedDataFrame,
        time_column: str,
        value_columns: List[str],
        title: str = "Time Series Plot",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """시계열 플롯을 생성합니다.
        
        Args:
            data: 시각화할 데이터
            time_column: 시간 컬럼
            value_columns: 값 컬럼들
            title: 플롯 제목
            config: 추가 설정
            
        Returns:
            Dict[str, Any]: 시계열 플롯 정보
        """
        df = data.to_pandas()
        
        # 시간 컬럼을 datetime으로 변환
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)
        
        if self.style == "plotly":
            return self._create_plotly_time_series(df, time_column, value_columns, title, config)
        else:
            return self._create_matplotlib_time_series(df, time_column, value_columns, title, config)

    def create_advanced_report(
        self,
        data: UnifiedDataFrame,
        analysis_config: Dict[str, Any],
        title: str = "Advanced Analytics Report"
    ) -> Dict[str, Any]:
        """고급 분석 리포트를 생성합니다.
        
        Args:
            data: 분석할 데이터
            analysis_config: 분석 설정
            title: 리포트 제목
            
        Returns:
            Dict[str, Any]: 분석 리포트
        """
        df = data.to_pandas()
        
        report = {
            "title": title,
            "data_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.to_dict()
            },
            "analysis_results": {}
        }
        
        # 기술 통계
        if analysis_config.get("include_descriptive", True):
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_columns:
                report["analysis_results"]["descriptive"] = df[numeric_columns].describe().to_dict()
        
        # 상관관계 분석
        if analysis_config.get("include_correlation", True):
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) > 1:
                correlation_matrix = df[numeric_columns].corr()
                report["analysis_results"]["correlation"] = correlation_matrix.to_dict()
        
        # 시각화
        if analysis_config.get("include_visualizations", True):
            visualizations = []
            
            # 히스토그램
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_columns[:5]:  # 최대 5개 컬럼
                hist_info = self._create_histogram_plot(df[col].dropna())
                visualizations.append({
                    "type": "histogram",
                    "column": col,
                    "data": hist_info
                })
            
            # 상관관계 히트맵
            if len(numeric_columns) > 1:
                corr_plot = self._create_correlation_plot(df, "Correlation Heatmap")
                visualizations.append({
                    "type": "correlation_heatmap",
                    "data": corr_plot
                })
            
            report["analysis_results"]["visualizations"] = visualizations
        
        return report

    def _create_plotly_dashboard(
        self, 
        df: pd.DataFrame, 
        chart_configs: List[Dict[str, Any]], 
        title: str, 
        layout: str
    ) -> Dict[str, Any]:
        """Plotly 대시보드를 생성합니다."""
        if layout == "grid":
            rows = int(np.ceil(len(chart_configs) / 2))
            cols = 2
        else:
            rows = len(chart_configs)
            cols = 1
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[config.get("title", f"Chart {i+1}") for i, config in enumerate(chart_configs)]
        )
        
        for i, config in enumerate(chart_configs):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            chart_type = config.get("type", "scatter")
            x_col = config.get("x_column")
            y_col = config.get("y_column")
            
            if chart_type == "scatter" and x_col and y_col:
                fig.add_trace(
                    go.Scatter(x=df[x_col], y=df[y_col], mode='markers', name=f"{x_col} vs {y_col}"),
                    row=row, col=col
                )
            elif chart_type == "bar" and x_col and y_col:
                fig.add_trace(
                    go.Bar(x=df[x_col], y=df[y_col], name=f"{x_col} vs {y_col}"),
                    row=row, col=col
                )
            elif chart_type == "line" and x_col and y_col:
                fig.add_trace(
                    go.Scatter(x=df[x_col], y=df[y_col], mode='lines', name=f"{x_col} vs {y_col}"),
                    row=row, col=col
                )
        
        fig.update_layout(title=title, height=300 * rows)
        
        return {
            "type": "plotly_dashboard",
            "title": title,
            "chart_count": len(chart_configs),
            "layout": layout,
            "figure": fig.to_dict()
        }

    def _create_matplotlib_dashboard(
        self, 
        df: pd.DataFrame, 
        chart_configs: List[Dict[str, Any]], 
        title: str, 
        layout: str
    ) -> Dict[str, Any]:
        """Matplotlib 대시보드를 생성합니다."""
        if layout == "grid":
            rows = int(np.ceil(len(chart_configs) / 2))
            cols = 2
        else:
            rows = len(chart_configs)
            cols = 1
        
        fig, axes = plt.subplots(rows, cols, figsize=(self.figsize[0] * cols, self.figsize[1] * rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, config in enumerate(chart_configs):
            if i >= len(axes):
                break
                
            ax = axes[i]
            chart_type = config.get("type", "scatter")
            x_col = config.get("x_column")
            y_col = config.get("y_column")
            
            if chart_type == "scatter" and x_col and y_col:
                ax.scatter(df[x_col], df[y_col], alpha=0.6)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
            elif chart_type == "bar" and x_col and y_col:
                ax.bar(df[x_col], df[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
            elif chart_type == "line" and x_col and y_col:
                ax.plot(df[x_col], df[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
            
            ax.set_title(config.get("title", f"Chart {i+1}"))
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # 이미지를 base64로 인코딩
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "matplotlib_dashboard",
            "title": title,
            "chart_count": len(chart_configs),
            "layout": layout,
            "image_base64": image_base64
        }

    def _create_scatter_matrix(self, df: pd.DataFrame, title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """산점도 행렬을 생성합니다."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            return {"error": "산점도 행렬을 위해서는 최소 2개의 숫자 컬럼이 필요합니다."}
        
        if self.style == "plotly":
            fig = px.scatter_matrix(df, dimensions=numeric_columns[:5], title=title)
            return {
                "type": "scatter_matrix",
                "title": title,
                "columns": numeric_columns[:5],
                "figure": fig.to_dict()
            }
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            pd.plotting.scatter_matrix(df[numeric_columns[:5]], ax=ax, alpha=0.6)
            plt.title(title)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "type": "scatter_matrix",
                "title": title,
                "columns": numeric_columns[:5],
                "image_base64": image_base64
            }

    def _create_heatmap(self, df: pd.DataFrame, x_column: str, y_column: str, title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """히트맵을 생성합니다."""
        if self.style == "plotly":
            fig = px.density_heatmap(df, x=x_column, y=y_column, title=title)
            return {
                "type": "heatmap",
                "title": title,
                "x_column": x_column,
                "y_column": y_column,
                "figure": fig.to_dict()
            }
        else:
            fig, ax = plt.subplots(figsize=self.figsize)
            sns.heatmap(df.pivot_table(values=y_column, index=x_column, aggfunc='mean'), 
                       annot=True, fmt='.2f', cmap='viridis', ax=ax)
            plt.title(title)
            plt.tight_layout()
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return {
                "type": "heatmap",
                "title": title,
                "x_column": x_column,
                "y_column": y_column,
                "image_base64": image_base64
            }

    def _create_violin_plot(self, df: pd.DataFrame, x_column: str, y_column: str, hue_column: Optional[str], title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """바이올린 플롯을 생성합니다."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if hue_column:
            sns.violinplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)
        else:
            sns.violinplot(data=df, x=x_column, y=y_column, ax=ax)
        
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "violin_plot",
            "title": title,
            "x_column": x_column,
            "y_column": y_column,
            "hue_column": hue_column,
            "image_base64": image_base64
        }

    def _create_box_plot(self, df: pd.DataFrame, x_column: str, y_column: str, hue_column: Optional[str], title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """박스 플롯을 생성합니다."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if hue_column:
            sns.boxplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)
        else:
            sns.boxplot(data=df, x=x_column, y=y_column, ax=ax)
        
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "box_plot",
            "title": title,
            "x_column": x_column,
            "y_column": y_column,
            "hue_column": hue_column,
            "image_base64": image_base64
        }

    def _create_histogram(self, df: pd.DataFrame, x_column: str, hue_column: Optional[str], title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """히스토그램을 생성합니다."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if hue_column:
            sns.histplot(data=df, x=x_column, hue=hue_column, kde=True, ax=ax)
        else:
            sns.histplot(data=df, x=x_column, kde=True, ax=ax)
        
        plt.title(title)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "histogram",
            "title": title,
            "x_column": x_column,
            "hue_column": hue_column,
            "image_base64": image_base64
        }

    def _create_line_plot(self, df: pd.DataFrame, x_column: str, y_column: str, hue_column: Optional[str], title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """선 그래프를 생성합니다."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if hue_column:
            for value in df[hue_column].unique():
                subset = df[df[hue_column] == value]
                ax.plot(subset[x_column], subset[y_column], label=value, marker='o')
            ax.legend()
        else:
            ax.plot(df[x_column], df[y_column], marker='o')
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        plt.title(title)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "line_plot",
            "title": title,
            "x_column": x_column,
            "y_column": y_column,
            "hue_column": hue_column,
            "image_base64": image_base64
        }

    def _create_bar_plot(self, df: pd.DataFrame, x_column: str, y_column: str, hue_column: Optional[str], title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """막대 그래프를 생성합니다."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if hue_column:
            sns.barplot(data=df, x=x_column, y=y_column, hue=hue_column, ax=ax)
        else:
            sns.barplot(data=df, x=x_column, y=y_column, ax=ax)
        
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "bar_plot",
            "title": title,
            "x_column": x_column,
            "y_column": y_column,
            "hue_column": hue_column,
            "image_base64": image_base64
        }

    def _create_pie_chart(self, df: pd.DataFrame, x_column: str, y_column: str, title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """원형 그래프를 생성합니다."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if y_column:
            # y_column이 있으면 그룹화하여 합계 계산
            pie_data = df.groupby(x_column)[y_column].sum()
        else:
            # y_column이 없으면 x_column의 값 개수 계산
            pie_data = df[x_column].value_counts()
        
        ax.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
        plt.title(title)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "pie_chart",
            "title": title,
            "x_column": x_column,
            "y_column": y_column,
            "image_base64": image_base64
        }

    def _create_correlation_plot(self, df: pd.DataFrame, title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """상관관계 플롯을 생성합니다."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            return {"error": "상관관계 플롯을 위해서는 최소 2개의 숫자 컬럼이 필요합니다."}
        
        fig, ax = plt.subplots(figsize=self.figsize)
        correlation_matrix = df[numeric_columns].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.2f')
        plt.title(title)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "correlation_plot",
            "title": title,
            "columns": numeric_columns,
            "correlation_matrix": correlation_matrix.to_dict(),
            "image_base64": image_base64
        }

    def _create_histogram_plot(self, series: pd.Series) -> Dict[str, Any]:
        """히스토그램 플롯을 생성합니다."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(series, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f'Histogram of {series.name}')
        ax.set_xlabel(series.name)
        ax.set_ylabel('Frequency')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_base64": image_base64,
            "statistics": {
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max()
            }
        }

    def _create_qq_plot(self, series: pd.Series) -> Dict[str, Any]:
        """Q-Q 플롯을 생성합니다."""
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(series, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot of {series.name}')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_base64": image_base64,
            "normality_test": {
                "shapiro_stat": stats.shapiro(series)[0],
                "shapiro_p": stats.shapiro(series)[1]
            }
        }

    def _create_box_plot_single(self, series: pd.Series) -> Dict[str, Any]:
        """단일 박스 플롯을 생성합니다."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(series)
        ax.set_title(f'Box Plot of {series.name}')
        ax.set_ylabel(series.name)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_base64": image_base64,
            "quartiles": {
                "q1": series.quantile(0.25),
                "q2": series.quantile(0.5),
                "q3": series.quantile(0.75)
            }
        }

    def _create_violin_plot_single(self, series: pd.Series) -> Dict[str, Any]:
        """단일 바이올린 플롯을 생성합니다."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.violinplot([series])
        ax.set_title(f'Violin Plot of {series.name}')
        ax.set_ylabel(series.name)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_base64": image_base64,
            "density_info": {
                "skewness": series.skew(),
                "kurtosis": series.kurtosis()
            }
        }

    def _create_density_plot(self, series: pd.Series) -> Dict[str, Any]:
        """밀도 플롯을 생성합니다."""
        fig, ax = plt.subplots(figsize=(8, 6))
        series.plot(kind='density', ax=ax)
        ax.set_title(f'Density Plot of {series.name}')
        ax.set_xlabel(series.name)
        ax.set_ylabel('Density')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_base64": image_base64,
            "density_stats": {
                "mean": series.mean(),
                "std": series.std()
            }
        }

    def _create_ecdf_plot(self, series: pd.Series) -> Dict[str, Any]:
        """경험적 누적분포함수 플롯을 생성합니다."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sorted_data = np.sort(series)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        ax.plot(sorted_data, y, marker='.', linestyle='none')
        ax.set_title(f'ECDF Plot of {series.name}')
        ax.set_xlabel(series.name)
        ax.set_ylabel('Cumulative Probability')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "image_base64": image_base64,
            "ecdf_stats": {
                "median": series.median(),
                "percentiles": {
                    "25th": series.quantile(0.25),
                    "75th": series.quantile(0.75)
                }
            }
        }

    def _create_plotly_time_series(self, df: pd.DataFrame, time_column: str, value_columns: List[str], title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Plotly 시계열 플롯을 생성합니다."""
        fig = go.Figure()
        
        for col in value_columns:
            fig.add_trace(go.Scatter(
                x=df[time_column],
                y=df[col],
                mode='lines+markers',
                name=col
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=time_column,
            yaxis_title="Value",
            hovermode='x unified'
        )
        
        return {
            "type": "plotly_time_series",
            "title": title,
            "time_column": time_column,
            "value_columns": value_columns,
            "figure": fig.to_dict()
        }

    def _create_matplotlib_time_series(self, df: pd.DataFrame, time_column: str, value_columns: List[str], title: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Matplotlib 시계열 플롯을 생성합니다."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for col in value_columns:
            ax.plot(df[time_column], df[col], marker='o', label=col)
        
        ax.set_xlabel(time_column)
        ax.set_ylabel("Value")
        ax.set_title(title)
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            "type": "matplotlib_time_series",
            "title": title,
            "time_column": time_column,
            "value_columns": value_columns,
            "image_base64": image_base64
        }

    def __repr__(self) -> str:
        """문자열 표현."""
        return f"AdvancedVisualization(style={self.style}, figsize={self.figsize})"

    def __str__(self) -> str:
        """문자열 표현."""
        return self.__repr__()
