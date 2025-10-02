#!/usr/bin/env python3
"""
analyze_data 함수 사용 예제

Bridge Analytics의 analyze_data 함수를 사용하는 다양한 예제를 제공합니다.
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.bridge.analytics import (
    analyze_data,
    quick_analysis,
    comprehensive_analysis,
    quality_focused_analysis,
    visualization_focused_analysis,
    AnalysisConfig
)

def example_basic_usage():
    """기본 사용법 예제"""
    print("📚 예제 1: 기본 사용법")
    print("-" * 40)
    
    # 샘플 데이터 생성
    data = {
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'age': [25, 30, 35, 28, 32],
        'salary': [50000, 60000, 70000, 55000, 65000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
    }
    
    df = pd.DataFrame(data)
    
    # 기본 분석 실행
    result = quick_analysis(df)
    
    print(f"✅ 분석 완료: {result.success}")
    print(f"📊 데이터 크기: {result.data_summary['total_rows']}행 x {result.data_summary['total_columns']}열")
    print(f"🔍 데이터 품질: {result.quality_metrics['overall_score']:.3f}")
    print()

def example_custom_config():
    """커스텀 설정 예제"""
    print("📚 예제 2: 커스텀 설정")
    print("-" * 40)
    
    # 더 큰 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 0.5, n_samples),
        'spending': np.random.gamma(2, 1000, n_samples),
        'satisfaction': np.random.randint(1, 6, n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 커스텀 설정으로 분석
    config = AnalysisConfig(
        include_descriptive_stats=True,
        include_correlation_analysis=True,
        include_quality_metrics=True,
        generate_charts=True,
        quality_threshold=0.9,
        verbose=True
    )
    
    result = analyze_data(df, config)
    
    print(f"✅ 분석 완료: {result.success}")
    print(f"📊 데이터 크기: {result.data_summary['total_rows']}행 x {result.data_summary['total_columns']}열")
    print(f"🔍 데이터 품질: {result.quality_metrics['overall_score']:.3f}")
    print(f"📈 기술 통계: {len(result.descriptive_stats['descriptive_stats'])}개 컬럼")
    print(f"🔗 상관관계: 강한 {len(result.correlation_analysis['strong_correlations'])}개, 중간 {len(result.correlation_analysis['moderate_correlations'])}개")
    print()

def example_multi_source_data():
    """다중 소스 데이터 예제"""
    print("📚 예제 3: 다중 소스 데이터")
    print("-" * 40)
    
    # 고객 데이터
    customers_df = pd.DataFrame({
        'customer_id': range(1, 101),
        'name': [f'Customer_{i}' for i in range(1, 101)],
        'age': np.random.normal(30, 10, 100),
        'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu'], 100)
    })
    
    # 주문 데이터
    orders_df = pd.DataFrame({
        'order_id': range(1, 201),
        'customer_id': np.random.choice(range(1, 101), 200),
        'product': np.random.choice(['A', 'B', 'C', 'D'], 200),
        'amount': np.random.normal(100, 30, 200),
        'date': pd.date_range('2023-01-01', periods=200, freq='D')
    })
    
    # 다중 소스 데이터
    data_sources = {
        'customers': customers_df,
        'orders': orders_df
    }
    
    # 다중 소스 분석
    result = analyze_data(data_sources)
    
    print(f"✅ 분석 완료: {result.success}")
    print(f"📊 통합된 데이터 크기: {result.data_summary['total_rows']}행 x {result.data_summary['total_columns']}열")
    print(f"🔍 데이터 품질: {result.quality_metrics['overall_score']:.3f}")
    print()

def example_specialized_analysis():
    """전문 분석 예제"""
    print("📚 예제 4: 전문 분석")
    print("-" * 40)
    
    # 품질 중심 분석용 데이터 (일부 결측값 포함)
    np.random.seed(123)
    data = {
        'id': range(1, 501),
        'value1': np.random.normal(100, 20, 500),
        'value2': np.random.normal(50, 10, 500),
        'category': np.random.choice(['A', 'B', 'C'], 500),
        'score': np.random.uniform(0, 100, 500)
    }
    
    df = pd.DataFrame(data)
    
    # 일부 결측값 추가
    df.loc[np.random.choice(df.index, 50), 'value1'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'value2'] = np.nan
    
    print("🔍 품질 중심 분석:")
    quality_result = quality_focused_analysis(df)
    print(f"   품질 점수: {quality_result.quality_metrics['overall_score']:.3f}")
    print(f"   완전성: {quality_result.quality_metrics['completeness']:.3f}")
    print(f"   정확성: {quality_result.quality_metrics['accuracy']:.3f}")
    
    print("\n📊 시각화 중심 분석:")
    viz_result = visualization_focused_analysis(df)
    print(f"   생성된 차트: {len(viz_result.charts)}개")
    print(f"   대시보드: {'생성됨' if viz_result.dashboard else '생성 안됨'}")
    print()

def example_error_handling():
    """오류 처리 예제"""
    print("📚 예제 5: 오류 처리")
    print("-" * 40)
    
    # 잘못된 데이터로 테스트
    invalid_data = {
        'col1': [1, 2, 3],
        'col2': ['a', 'b']  # 길이가 다름
    }
    
    try:
        result = analyze_data(invalid_data)
        print(f"✅ 분석 완료: {result.success}")
        if result.errors:
            print(f"❌ 오류: {result.errors}")
    except Exception as e:
        print(f"❌ 예외 발생: {e}")
    
    print()

def main():
    """메인 함수"""
    print("🚀 Bridge Analytics - analyze_data 함수 사용 예제")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_custom_config()
        example_multi_source_data()
        example_specialized_analysis()
        example_error_handling()
        
        print("=" * 60)
        print("🎉 모든 예제 완료!")
        print("\n💡 사용 팁:")
        print("   • quick_analysis(): 빠른 기본 분석")
        print("   • comprehensive_analysis(): 모든 기능 포함 종합 분석")
        print("   • quality_focused_analysis(): 데이터 품질 중심 분석")
        print("   • visualization_focused_analysis(): 시각화 중심 분석")
        print("   • analyze_data(data, config): 커스텀 설정으로 분석")
        
    except Exception as e:
        print(f"❌ 예제 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
