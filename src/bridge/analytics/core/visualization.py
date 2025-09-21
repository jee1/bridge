"""데이터 시각화 모듈

C1 마일스톤 1.4: 기본 시각화
- 차트 생성: 막대, 선, 산점도, 히스토그램 등
- 대시보드: 여러 차트를 조합한 대시보드
- 리포트: 시각화가 포함된 분석 리포트
"""

import matplotlib

matplotlib.use("Agg")  # GUI 백엔드 비활성화
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bridge.analytics.core.data_integration import UnifiedDataFrame

logger = logging.getLogger(__name__)

# 한글 폰트 설정
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class ChartConfig:
    """차트 설정을 담는 데이터 클래스"""

    title: str
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    figsize: Tuple[int, int] = (10, 6)
    style: str = "whitegrid"
    color_palette: str = "Set2"
    dpi: int = 100


@dataclass
class DashboardConfig:
    """대시보드 설정을 담는 데이터 클래스"""

    title: str
    layout: Tuple[int, int] = (2, 2)  # (rows, cols)
    figsize: Tuple[int, int] = (15, 10)
    style: str = "whitegrid"
    color_palette: str = "Set2"
    dpi: int = 100


@dataclass
class ReportConfig:
    """리포트 설정을 담는 데이터 클래스"""

    title: str
    author: Optional[str] = None
    date: Optional[str] = None
    figsize: Tuple[int, int] = (12, 8)
    style: str = "whitegrid"
    color_palette: str = "Set2"
    dpi: int = 100


class ChartGenerator:
    """데이터 시각화를 위한 차트 생성 클래스"""

    def __init__(self):
        """차트 생성기 초기화"""
        # Seaborn 스타일 설정
        sns.set_style("whitegrid")
        sns.set_palette("Set2")

    def create_bar_chart(
        self,
        df: UnifiedDataFrame,
        x_column: str,
        y_column: Optional[str] = None,
        config: Optional[ChartConfig] = None,
    ) -> plt.Figure:
        """막대 차트를 생성합니다.

        Args:
            df: 데이터프레임
            x_column: X축 컬럼명
            y_column: Y축 컬럼명 (None이면 카운트)
            config: 차트 설정

        Returns:
            matplotlib Figure 객체
        """
        pandas_df = df.to_pandas()

        if config is None:
            config = ChartConfig(title=f"Bar Chart: {x_column}")

        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        if y_column is None:
            # 카운트 차트
            value_counts = pandas_df[x_column].value_counts()
            ax.bar(value_counts.index, value_counts.values)
        else:
            # 그룹별 집계 차트
            grouped = pandas_df.groupby(x_column)[y_column].sum()
            ax.bar(grouped.index, grouped.values)

        ax.set_title(config.title, fontsize=14, fontweight="bold")
        if config.x_label:
            ax.set_xlabel(config.x_label)
        if config.y_label:
            ax.set_ylabel(config.y_label)

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def create_line_chart(
        self,
        df: UnifiedDataFrame,
        x_column: str,
        y_column: str,
        config: Optional[ChartConfig] = None,
    ) -> plt.Figure:
        """선 차트를 생성합니다.

        Args:
            df: 데이터프레임
            x_column: X축 컬럼명
            y_column: Y축 컬럼명
            config: 차트 설정

        Returns:
            matplotlib Figure 객체
        """
        pandas_df = df.to_pandas()

        if config is None:
            config = ChartConfig(title=f"Line Chart: {x_column} vs {y_column}")

        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        # X축 정렬
        sorted_df = pandas_df.sort_values(x_column)
        ax.plot(sorted_df[x_column], sorted_df[y_column], marker="o", linewidth=2)

        ax.set_title(config.title, fontsize=14, fontweight="bold")
        if config.x_label:
            ax.set_xlabel(config.x_label)
        if config.y_label:
            ax.set_ylabel(config.y_label)

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def create_scatter_plot(
        self,
        df: UnifiedDataFrame,
        x_column: str,
        y_column: str,
        hue_column: Optional[str] = None,
        config: Optional[ChartConfig] = None,
    ) -> plt.Figure:
        """산점도를 생성합니다.

        Args:
            df: 데이터프레임
            x_column: X축 컬럼명
            y_column: Y축 컬럼명
            hue_column: 색상 구분 컬럼명 (선택사항)
            config: 차트 설정

        Returns:
            matplotlib Figure 객체
        """
        pandas_df = df.to_pandas()

        if config is None:
            config = ChartConfig(title=f"Scatter Plot: {x_column} vs {y_column}")

        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        if hue_column and hue_column in pandas_df.columns:
            # 색상으로 구분
            unique_values = pandas_df[hue_column].unique()
            colors = plt.cm.Set2(np.linspace(0, 1, len(unique_values)))

            for i, value in enumerate(unique_values):
                subset = pandas_df[pandas_df[hue_column] == value]
                ax.scatter(
                    subset[x_column], subset[y_column], c=[colors[i]], label=str(value), alpha=0.7
                )
            ax.legend()
        else:
            ax.scatter(pandas_df[x_column], pandas_df[y_column], alpha=0.7)

        ax.set_title(config.title, fontsize=14, fontweight="bold")
        if config.x_label:
            ax.set_xlabel(config.x_label)
        if config.y_label:
            ax.set_ylabel(config.y_label)

        plt.tight_layout()

        return fig

    def create_histogram(
        self,
        df: UnifiedDataFrame,
        column: str,
        config: Optional[ChartConfig] = None,
        bins: int = 30,
    ) -> plt.Figure:
        """히스토그램을 생성합니다.

        Args:
            df: 데이터프레임
            column: 히스토그램을 그릴 컬럼명
            bins: 구간 수
            config: 차트 설정

        Returns:
            matplotlib Figure 객체
        """
        pandas_df = df.to_pandas()

        if config is None:
            config = ChartConfig(title=f"Histogram: {column}")

        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        # 결측값 제거
        data = pandas_df[column].dropna()
        ax.hist(data, bins=bins, alpha=0.7, edgecolor="black")

        ax.set_title(config.title, fontsize=14, fontweight="bold")
        if config.x_label:
            ax.set_xlabel(config.x_label)
        if config.y_label:
            ax.set_ylabel(config.y_label)

        plt.tight_layout()

        return fig

    def create_box_plot(
        self,
        df: UnifiedDataFrame,
        x_column: Optional[str],
        y_column: str,
        config: Optional[ChartConfig] = None,
    ) -> plt.Figure:
        """박스 플롯을 생성합니다.

        Args:
            df: 데이터프레임
            x_column: X축 그룹 컬럼명 (None이면 단일 박스)
            y_column: Y축 컬럼명
            config: 차트 설정

        Returns:
            matplotlib Figure 객체
        """
        pandas_df = df.to_pandas()

        if config is None:
            title = (
                f"Box Plot: {y_column}"
                if x_column is None
                else f"Box Plot: {x_column} vs {y_column}"
            )
            config = ChartConfig(title=title)

        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        if x_column is None:
            # 단일 박스 플롯
            data = pandas_df[y_column].dropna()
            ax.boxplot(data)
        else:
            # 그룹별 박스 플롯
            data = [
                pandas_df[pandas_df[x_column] == group][y_column].dropna()
                for group in pandas_df[x_column].unique()
            ]
            ax.boxplot(data, labels=pandas_df[x_column].unique())

        ax.set_title(config.title, fontsize=14, fontweight="bold")
        if config.x_label:
            ax.set_xlabel(config.x_label)
        if config.y_label:
            ax.set_ylabel(config.y_label)

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig

    def create_heatmap(
        self,
        df: UnifiedDataFrame,
        config: Optional[ChartConfig] = None,
        columns: Optional[List[str]] = None,
    ) -> plt.Figure:
        """상관관계 히트맵을 생성합니다.

        Args:
            df: 데이터프레임
            columns: 히트맵에 포함할 컬럼 목록
            config: 차트 설정

        Returns:
            matplotlib Figure 객체
        """
        pandas_df = df.to_pandas()

        if columns is None:
            # 숫자형 컬럼만 선택
            numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [col for col in columns if col in pandas_df.columns]

        if config is None:
            config = ChartConfig(title="Correlation Heatmap")

        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)

        # 상관관계 계산
        corr_matrix = pandas_df[numeric_columns].corr()

        # 히트맵 생성
        im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

        # 컬럼명 설정
        ax.set_xticks(range(len(numeric_columns)))
        ax.set_yticks(range(len(numeric_columns)))
        ax.set_xticklabels(numeric_columns, rotation=45)
        ax.set_yticklabels(numeric_columns)

        # 색상바 추가
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation", rotation=270, labelpad=20)

        # 상관계수 값 표시
        for i in range(len(numeric_columns)):
            for j in range(len(numeric_columns)):
                text = ax.text(
                    j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black"
                )

        ax.set_title(config.title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        return fig


class DashboardGenerator:
    """대시보드 생성을 위한 클래스"""

    def __init__(self):
        """대시보드 생성기 초기화"""
        self.chart_generator = ChartGenerator()

    def create_analytics_dashboard(
        self, df: UnifiedDataFrame, config: Optional[DashboardConfig] = None
    ) -> plt.Figure:
        """분석 대시보드를 생성합니다.

        Args:
            df: 데이터프레임
            config: 대시보드 설정

        Returns:
            matplotlib Figure 객체
        """
        pandas_df = df.to_pandas()

        if config is None:
            config = DashboardConfig(title="Analytics Dashboard")

        # 숫자형 컬럼 선택
        numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = pandas_df.select_dtypes(include=["object"]).columns.tolist()

        # 서브플롯 생성
        rows, cols = config.layout
        fig, axes = plt.subplots(rows, cols, figsize=config.figsize, dpi=config.dpi)
        axes = axes.flatten() if rows * cols > 1 else [axes]

        plot_idx = 0

        # 1. 히스토그램 (첫 번째 숫자형 컬럼)
        if numeric_columns and plot_idx < len(axes):
            fig_hist = self.chart_generator.create_histogram(df, numeric_columns[0])
            axes[plot_idx].set_title(f"Distribution: {numeric_columns[0]}")
            plot_idx += 1

        # 2. 박스 플롯 (첫 번째 숫자형 컬럼)
        if numeric_columns and plot_idx < len(axes):
            fig_box = self.chart_generator.create_box_plot(df, None, numeric_columns[0])
            axes[plot_idx].set_title(f"Box Plot: {numeric_columns[0]}")
            plot_idx += 1

        # 3. 막대 차트 (첫 번째 범주형 컬럼)
        if categorical_columns and plot_idx < len(axes):
            fig_bar = self.chart_generator.create_bar_chart(df, categorical_columns[0])
            axes[plot_idx].set_title(f"Count: {categorical_columns[0]}")
            plot_idx += 1

        # 4. 산점도 (두 개의 숫자형 컬럼)
        if len(numeric_columns) >= 2 and plot_idx < len(axes):
            fig_scatter = self.chart_generator.create_scatter_plot(
                df, numeric_columns[0], numeric_columns[1]
            )
            axes[plot_idx].set_title(f"Scatter: {numeric_columns[0]} vs {numeric_columns[1]}")
            plot_idx += 1

        # 사용하지 않는 서브플롯 숨기기
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(config.title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        return fig


class ReportGenerator:
    """리포트 생성을 위한 클래스"""

    def __init__(self):
        """리포트 생성기 초기화"""
        self.chart_generator = ChartGenerator()
        self.dashboard_generator = DashboardGenerator()

    def generate_analytics_report(
        self, df: UnifiedDataFrame, config: Optional[ReportConfig] = None
    ) -> Dict[str, Any]:
        """분석 리포트를 생성합니다.

        Args:
            df: 데이터프레임
            config: 리포트 설정

        Returns:
            리포트 데이터 딕셔너리
        """
        pandas_df = df.to_pandas()

        if config is None:
            config = ReportConfig(title="Analytics Report")

        # 기본 통계 정보
        basic_stats = {
            "total_rows": len(pandas_df),
            "total_columns": len(pandas_df.columns),
            "numeric_columns": len(pandas_df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(pandas_df.select_dtypes(include=["object"]).columns),
            "missing_values": pandas_df.isnull().sum().sum(),
            "duplicate_rows": pandas_df.duplicated().sum(),
        }

        # 컬럼별 통계
        column_stats = {}
        for col in pandas_df.columns:
            if pandas_df[col].dtype in [np.number, "int64", "float64"]:
                column_stats[col] = {
                    "type": "numeric",
                    "mean": pandas_df[col].mean(),
                    "std": pandas_df[col].std(),
                    "min": pandas_df[col].min(),
                    "max": pandas_df[col].max(),
                    "missing_count": pandas_df[col].isnull().sum(),
                }
            else:
                column_stats[col] = {
                    "type": "categorical",
                    "unique_count": pandas_df[col].nunique(),
                    "most_common": (
                        pandas_df[col].mode().iloc[0] if not pandas_df[col].mode().empty else None
                    ),
                    "missing_count": pandas_df[col].isnull().sum(),
                }

        # 차트 생성
        charts = {}

        # 히스토그램 (숫자형 컬럼)
        numeric_columns = pandas_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            charts["histogram"] = self.chart_generator.create_histogram(
                df,
                numeric_columns[0],
                config=ChartConfig(title=f"Distribution of {numeric_columns[0]}"),
            )

        # 막대 차트 (범주형 컬럼)
        categorical_columns = pandas_df.select_dtypes(include=["object"]).columns.tolist()
        if categorical_columns:
            charts["bar_chart"] = self.chart_generator.create_bar_chart(
                df,
                categorical_columns[0],
                config=ChartConfig(title=f"Count of {categorical_columns[0]}"),
            )

        # 산점도 (두 개의 숫자형 컬럼)
        if len(numeric_columns) >= 2:
            charts["scatter_plot"] = self.chart_generator.create_scatter_plot(
                df,
                numeric_columns[0],
                numeric_columns[1],
                config=ChartConfig(title=f"{numeric_columns[0]} vs {numeric_columns[1]}"),
            )

        # 상관관계 히트맵
        if len(numeric_columns) >= 2:
            charts["heatmap"] = self.chart_generator.create_heatmap(
                df, config=ChartConfig(title="Correlation Matrix"), columns=numeric_columns
            )

        # 대시보드
        dashboard = self.dashboard_generator.create_analytics_dashboard(
            df, DashboardConfig(title="Analytics Dashboard")
        )

        return {
            "title": config.title,
            "author": config.author,
            "date": config.date,
            "basic_stats": basic_stats,
            "column_stats": column_stats,
            "charts": charts,
            "dashboard": dashboard,
        }
