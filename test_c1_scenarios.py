#!/usr/bin/env python3
"""C1 마일스톤 테스트 시나리오 - Bridge Analytics MVP 기능 검증"""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Bridge Analytics 모듈들
from bridge.analytics.core import UnifiedDataFrame
from bridge.analytics.core.statistics import StatisticsAnalyzer
from bridge.analytics.core.quality import QualityChecker
from bridge.analytics.core.visualization import ChartGenerator
from bridge.analytics.core.cross_source_joiner import CrossSourceJoiner

class C1TestScenarios:
    """C1 마일스톤 테스트 시나리오 실행기"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def log_result(self, scenario: str, success: bool, details: Dict[str, Any]):
        """테스트 결과 로깅"""
        self.results[scenario] = {
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        
        status = "✅ 성공" if success else "❌ 실패"
        print(f"{status} {scenario}: {details.get('message', '')}")
    
    def test_data_integration(self):
        """시나리오 1: 데이터 통합 테스트"""
        print("\n🔗 시나리오 1: 데이터 통합 테스트")
        print("-" * 50)
        
        try:
            # 1. UnifiedDataFrame 생성 테스트
            data1 = [{"id": i, "name": f"user_{i}", "age": 20 + i % 50} for i in range(1000)]
            df1 = UnifiedDataFrame(data1)
            
            data2 = [{"id": i, "amount": i * 100, "category": f"cat_{i % 10}"} for i in range(1000)]
            df2 = UnifiedDataFrame(data2)
            
            # 2. 크로스 소스 조인 테스트
            joiner = CrossSourceJoiner()
            joiner.register_table("df1", df1)
            joiner.register_table("df2", df2)
            result = joiner.join_tables("df1", "df2", "df1.id = df2.id", "inner")
            
            # 3. 데이터 타입 정규화 테스트
            mixed_data = [
                {"id": 1, "value": "100", "date": "2024-01-01"},
                {"id": 2, "value": 200, "date": "2024-01-02"},
                {"id": 3, "value": "300.5", "date": "2024-01-03"}
            ]
            df_mixed = UnifiedDataFrame(mixed_data)
            
            self.log_result("데이터 통합", True, {
                "message": f"UnifiedDataFrame 생성 성공, 조인 결과: {result.num_rows}행",
                "df1_rows": df1.num_rows,
                "df2_rows": df2.num_rows,
                "join_result_rows": result.num_rows,
                "mixed_data_rows": df_mixed.num_rows
            })
            
        except Exception as e:
            self.log_result("데이터 통합", False, {
                "message": f"데이터 통합 실패: {str(e)}",
                "error": str(e)
            })
    
    def test_statistics_analysis(self):
        """시나리오 2: 기본 통계 분석 테스트"""
        print("\n📊 시나리오 2: 기본 통계 분석 테스트")
        print("-" * 50)
        
        try:
            # 1. 고객 세그멘테이션 데이터 생성
            np.random.seed(42)
            customer_data = []
            for i in range(1000):
                age = np.random.normal(40, 15)
                age = max(20, min(70, int(age)))
                
                # 연령과 구매액의 상관관계
                base_spent = age * 1000
                spent = base_spent + np.random.normal(0, base_spent * 0.3)
                spent = max(0, spent)
                
                customer_data.append({
                    "id": i,
                    "age": age,
                    "spent": round(spent, 2),
                    "city": f"city_{i % 10}",
                    "vip": spent > 50000
                })
            
            df_customers = UnifiedDataFrame(customer_data)
            analyzer = StatisticsAnalyzer()
            
            # 2. 기본 통계 계산
            descriptive_stats = analyzer.calculate_descriptive_stats(df_customers, ["age", "spent"])
            age_stats = descriptive_stats.get("age")
            spent_stats = descriptive_stats.get("spent")
            
            # 3. 분포 분석
            distribution_stats = analyzer.calculate_distribution_stats(df_customers, ["age", "spent"])
            
            # 4. 상관관계 분석
            correlation_result = analyzer.calculate_correlation(df_customers, ["age", "spent"])
            
            # 상관관계 값 추출
            correlation_value = 0.0
            if correlation_result.strong_correlations:
                correlation_value = correlation_result.strong_correlations[0]["correlation"]
            elif correlation_result.moderate_correlations:
                correlation_value = correlation_result.moderate_correlations[0]["correlation"]
            
            self.log_result("통계 분석", True, {
                "message": f"통계 분석 완료, 연령-구매액 상관관계: {correlation_value:.3f}",
                "age_mean": age_stats.mean if age_stats else 0,
                "spent_mean": spent_stats.mean if spent_stats else 0,
                "correlation": correlation_value,
                "total_customers": len(customer_data)
            })
            
        except Exception as e:
            self.log_result("통계 분석", False, {
                "message": f"통계 분석 실패: {str(e)}",
                "error": str(e)
            })
    
    def test_data_quality(self):
        """시나리오 3: 데이터 품질 검사 테스트"""
        print("\n🔍 시나리오 3: 데이터 품질 검사 테스트")
        print("-" * 50)
        
        try:
            # 1. 결측값이 있는 데이터 생성
            quality_data = []
            for i in range(1000):
                # 10% 확률로 결측값 생성
                name = f"item_{i}" if np.random.random() > 0.1 else None
                value = i * 100 if np.random.random() > 0.05 else None
                category = f"cat_{i % 5}" if np.random.random() > 0.15 else None
                
                quality_data.append({
                    "id": i,
                    "name": name,
                    "value": value,
                    "category": category,
                    "date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
                })
            
            df_quality = UnifiedDataFrame(quality_data)
            checker = QualityChecker()
            
            # 2. 데이터 품질 검사
            quality_report = checker.generate_quality_report(df_quality)
            
            # 3. 이상치 탐지
            # 정상 데이터 + 이상치 생성
            normal_data = [{"value": np.random.normal(100, 20)} for _ in range(950)]
            outlier_data = [{"value": np.random.normal(500, 50)} for _ in range(50)]
            outlier_test_data = normal_data + outlier_data
            df_outlier = UnifiedDataFrame(outlier_test_data)
            
            outlier_stats = checker.detect_outliers(df_outlier, ["value"])
            
            # 4. 중복값 검사
            duplicate_data = [{"id": i, "name": f"item_{i % 100}"} for i in range(200)]
            df_duplicate = UnifiedDataFrame(duplicate_data)
            consistency_stats = checker.check_consistency(df_duplicate)
            
            # 이상치 개수 계산
            outliers_count = sum(stats.outlier_count for stats in outlier_stats.values())
            
            self.log_result("데이터 품질", True, {
                "message": f"품질 검사 완료, 이상치 {outliers_count}개, 중복 {consistency_stats.duplicate_rows}개",
                "missing_count": quality_report.missing_value_score,
                "outliers_count": outliers_count,
                "duplicates_count": consistency_stats.duplicate_rows,
                "quality_score": quality_report.overall_score
            })
            
        except Exception as e:
            self.log_result("데이터 품질", False, {
                "message": f"데이터 품질 검사 실패: {str(e)}",
                "error": str(e)
            })
    
    def test_visualization(self):
        """시나리오 4: 시각화 테스트"""
        print("\n📈 시나리오 4: 시각화 테스트")
        print("-" * 50)
        
        try:
            # 1. 매출 트렌드 데이터 생성
            sales_data = []
            base_date = datetime(2024, 1, 1)
            
            for i in range(365):  # 1년간의 데이터
                date = base_date + timedelta(days=i)
                # 계절성과 트렌드가 있는 매출 데이터
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)
                trend_factor = 1 + i * 0.001
                random_factor = 1 + np.random.normal(0, 0.2)
                
                sales = 1000000 * seasonal_factor * trend_factor * random_factor
                sales = max(0, sales)
                
                sales_data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "sales": round(sales, 2),
                    "month": date.month,
                    "quarter": (date.month - 1) // 3 + 1
                })
            
            df_sales = UnifiedDataFrame(sales_data)
            chart_generator = ChartGenerator()
            
            # 2. 차트 생성 테스트
            # 막대 차트 (월별 매출) - pandas를 사용해서 그룹화
            pandas_df = df_sales.to_pandas()
            monthly_sales = pandas_df.groupby("month")["sales"].sum().reset_index()
            monthly_sales_df = UnifiedDataFrame(monthly_sales)
            
            bar_chart = chart_generator.create_bar_chart(
                monthly_sales_df, "month", "sales", 
                title="월별 매출 현황"
            )
            
            # 선 차트 (일별 매출 트렌드)
            line_chart = chart_generator.create_line_chart(
                df_sales, "date", "sales",
                title="일별 매출 트렌드"
            )
            
            # 히스토그램 (매출 분포)
            histogram = chart_generator.create_histogram(
                df_sales, "sales",
                title="매출 분포"
            )
            
            # 3. 대시보드 생성
            dashboard = chart_generator.create_dashboard([
                {"type": "bar", "data": monthly_sales, "x": "month", "y": "sales"},
                {"type": "line", "data": df_sales, "x": "date", "y": "sales"},
                {"type": "histogram", "data": df_sales, "column": "sales"}
            ], title="매출 분석 대시보드")
            
            self.log_result("시각화", True, {
                "message": f"시각화 완료, {len(sales_data)}개 데이터 포인트",
                "charts_created": 3,
                "dashboard_created": True,
                "data_points": len(sales_data)
            })
            
        except Exception as e:
            self.log_result("시각화", False, {
                "message": f"시각화 실패: {str(e)}",
                "error": str(e)
            })
    
    def test_performance_benchmark(self):
        """시나리오 5: 성능 벤치마크 테스트"""
        print("\n⚡ 시나리오 5: 성능 벤치마크 테스트")
        print("-" * 50)
        
        try:
            # 1. 대용량 데이터 처리 성능
            print("  대용량 데이터 처리 테스트...")
            large_data = [{"id": i, "value": i * 2, "category": f"cat_{i % 100}"} for i in range(100000)]
            
            start_time = time.time()
            df_large = UnifiedDataFrame(large_data)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # 2. 통계 분석 성능
            print("  통계 분석 성능 테스트...")
            analyzer = StatisticsAnalyzer()
            
            start_time = time.time()
            stats = analyzer.calculate_descriptive_stats(df_large, ["value"])
            end_time = time.time()
            
            analysis_time = end_time - start_time
            
            # 3. 메모리 사용량 측정
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            
            # 4. 크로스 소스 조인 성능
            print("  크로스 소스 조인 성능 테스트...")
            df1 = UnifiedDataFrame([{"id": i, "name": f"user_{i}"} for i in range(10000)])
            df2 = UnifiedDataFrame([{"id": i, "amount": i * 100} for i in range(10000)])
            joiner = CrossSourceJoiner()
            joiner.register_table("df1", df1)
            joiner.register_table("df2", df2)
            
            start_time = time.time()
            result = joiner.join_tables("df1", "df2", "df1.id = df2.id", "inner")
            end_time = time.time()
            
            join_time = end_time - start_time
            
            self.log_result("성능 벤치마크", True, {
                "message": f"성능 테스트 완료, 100K행 처리: {processing_time:.3f}초",
                "processing_time": processing_time,
                "analysis_time": analysis_time,
                "join_time": join_time,
                "memory_usage_mb": memory_usage,
                "large_data_rows": len(large_data),
                "join_result_rows": result.num_rows
            })
            
        except Exception as e:
            self.log_result("성능 벤치마크", False, {
                "message": f"성능 벤치마크 실패: {str(e)}",
                "error": str(e)
            })
    
    def test_cross_source_analysis(self):
        """시나리오 6: 크로스 소스 분석 테스트"""
        print("\n🔄 시나리오 6: 크로스 소스 분석 테스트")
        print("-" * 50)
        
        try:
            # 1. 고객 데이터 (PostgreSQL 시뮬레이션)
            customers_data = []
            for i in range(1000):
                customers_data.append({
                    "customer_id": i,
                    "name": f"customer_{i}",
                    "age": 20 + i % 50,
                    "city": f"city_{i % 10}",
                    "registration_date": f"2024-{(i % 12) + 1:02d}-01"
                })
            
            df_customers = UnifiedDataFrame(customers_data)
            
            # 2. 주문 데이터 (MySQL 시뮬레이션)
            orders_data = []
            for i in range(5000):
                customer_id = i % 1000
                orders_data.append({
                    "order_id": i,
                    "customer_id": customer_id,
                    "product_id": i % 100,
                    "amount": (i % 1000) * 100 + np.random.normal(0, 1000),
                    "order_date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
                })
            
            df_orders = UnifiedDataFrame(orders_data)
            
            # 3. 상품 데이터 (Elasticsearch 시뮬레이션)
            products_data = []
            for i in range(100):
                products_data.append({
                    "product_id": i,
                    "name": f"product_{i}",
                    "category": f"category_{i % 10}",
                    "price": 10000 + i * 1000
                })
            
            df_products = UnifiedDataFrame(products_data)
            
            # 4. 크로스 소스 조인
            joiner = CrossSourceJoiner()
            joiner.register_table("customers", df_customers)
            joiner.register_table("orders", df_orders)
            joiner.register_table("products", df_products)
            
            # 고객-주문 조인
            customer_orders = joiner.join_tables("customers", "orders", "customers.customer_id = orders.customer_id", "inner")
            
            # 주문-상품 조인
            order_products = joiner.join_tables("orders", "products", "orders.product_id = products.product_id", "inner")
            
            # 5. 통합 분석
            analyzer = StatisticsAnalyzer()
            
            # 고객별 총 구매액 - pandas를 사용해서 그룹화
            customer_orders_pandas = customer_orders.to_pandas()
            customer_totals = customer_orders_pandas.groupby("customer_id")["amount"].sum().reset_index()
            customer_totals_df = UnifiedDataFrame(customer_totals)
            customer_stats = analyzer.calculate_descriptive_stats(customer_totals_df, ["amount"])
            
            # 상품별 평균 주문 금액 - pandas를 사용해서 그룹화
            order_products_pandas = order_products.to_pandas()
            product_avg = order_products_pandas.groupby("product_id")["amount"].mean().reset_index()
            product_avg_df = UnifiedDataFrame(product_avg)
            product_stats = analyzer.calculate_descriptive_stats(product_avg_df, ["amount"])
            
            # 통계 값 추출
            customer_amount_stats = customer_stats.get("amount")
            product_amount_stats = product_stats.get("amount")
            
            self.log_result("크로스 소스 분석", True, {
                "message": f"크로스 소스 분석 완료, 고객 평균 구매액: {customer_amount_stats.mean if customer_amount_stats else 0:.0f}원",
                "customers_count": len(customers_data),
                "orders_count": len(orders_data),
                "products_count": len(products_data),
                "customer_orders_rows": customer_orders.num_rows,
                "order_products_rows": order_products.num_rows,
                "avg_customer_spent": customer_amount_stats.mean if customer_amount_stats else 0,
                "avg_product_price": product_amount_stats.mean if product_amount_stats else 0
            })
            
        except Exception as e:
            self.log_result("크로스 소스 분석", False, {
                "message": f"크로스 소스 분석 실패: {str(e)}",
                "error": str(e)
            })
    
    def run_all_scenarios(self):
        """모든 테스트 시나리오 실행"""
        print("🚀 C1 마일스톤 테스트 시나리오 시작")
        print("=" * 60)
        
        # 각 시나리오 실행
        self.test_data_integration()
        self.test_statistics_analysis()
        self.test_data_quality()
        self.test_visualization()
        self.test_performance_benchmark()
        self.test_cross_source_analysis()
        
        # 결과 요약
        self.print_summary()
    
    def print_summary(self):
        """테스트 결과 요약 출력"""
        total_time = time.time() - self.start_time
        
        print("\n" + "=" * 60)
        print("📊 C1 마일스톤 테스트 결과 요약")
        print("=" * 60)
        
        success_count = sum(1 for result in self.results.values() if result["success"])
        total_count = len(self.results)
        
        print(f"⏱️  총 실행 시간: {total_time:.2f}초")
        print(f"✅ 성공: {success_count}/{total_count}")
        print(f"❌ 실패: {total_count - success_count}/{total_count}")
        print(f"📈 성공률: {(success_count / total_count) * 100:.1f}%")
        
        print("\n📋 상세 결과:")
        for scenario, result in self.results.items():
            status = "✅" if result["success"] else "❌"
            message = result["details"].get("message", "")
            print(f"  {status} {scenario}: {message}")
        
        # JSON 결과 저장
        with open("c1_test_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 상세 결과가 'c1_test_results.json'에 저장되었습니다.")
        
        if success_count == total_count:
            print("\n🎉 모든 C1 마일스톤 테스트가 성공적으로 완료되었습니다!")
            return True
        else:
            print(f"\n⚠️  {total_count - success_count}개의 테스트가 실패했습니다.")
            return False

def main():
    """메인 함수"""
    tester = C1TestScenarios()
    success = tester.run_all_scenarios()
    
    if success:
        print("\n✅ C1 마일스톤이 성공적으로 검증되었습니다!")
        exit(0)
    else:
        print("\n❌ C1 마일스톤 검증에 실패했습니다.")
        exit(1)

if __name__ == "__main__":
    main()
