#!/usr/bin/env python3
"""
OGORB 데이터 분석 스크립트
"""

import psycopg2
import pandas as pd

def analyze_ogorb_data():
    """OGORB 데이터 분석"""
    
    # 데이터베이스 연결
    conn = psycopg2.connect(
        host="127.0.0.1",
        port="5432",
        database="postgres",
        user="bridge_user",
        password="bridge_password"
    )
    
    try:
        # 기본 통계
        print("=== OGORB Production Data 기본 통계 ===")
        
        cursor = conn.cursor()
        
        # 전체 레코드 수
        cursor.execute("SELECT COUNT(*) FROM ogorb_production_data")
        total_records = cursor.fetchone()[0]
        print(f"총 레코드 수: {total_records:,}")
        
        # 날짜 범위
        cursor.execute("""
            SELECT 
                MIN(production_date) as earliest_date,
                MAX(production_date) as latest_date
            FROM ogorb_production_data
        """)
        date_range = cursor.fetchone()
        print(f"데이터 기간: {date_range[0]} ~ {date_range[1]}")
        
        # 고유 값 개수
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT state) as unique_states,
                COUNT(DISTINCT county) as unique_counties,
                COUNT(DISTINCT commodity) as unique_commodities,
                COUNT(DISTINCT land_class) as unique_land_classes,
                COUNT(DISTINCT offshore_region) as unique_offshore_regions
            FROM ogorb_production_data
        """)
        unique_counts = cursor.fetchone()
        print(f"고유 주(State) 수: {unique_counts[0]}")
        print(f"고유 카운티 수: {unique_counts[1]}")
        print(f"고유 상품(Commodity) 수: {unique_counts[2]}")
        print(f"고유 토지 분류 수: {unique_counts[3]}")
        print(f"고유 해상 지역 수: {unique_counts[4]}")
        
        # Volume 통계
        cursor.execute("""
            SELECT 
                SUM(volume) as total_volume,
                AVG(volume) as avg_volume,
                MIN(volume) as min_volume,
                MAX(volume) as max_volume
            FROM ogorb_production_data
            WHERE volume IS NOT NULL
        """)
        volume_stats = cursor.fetchone()
        print(f"\n=== Volume 통계 ===")
        print(f"총 Volume: {volume_stats[0]:,}")
        print(f"평균 Volume: {volume_stats[1]:,.2f}")
        print(f"최소 Volume: {volume_stats[2]:,}")
        print(f"최대 Volume: {volume_stats[3]:,}")
        
        # 상위 10개 상품별 통계
        print(f"\n=== 상위 10개 상품별 통계 ===")
        cursor.execute("""
            SELECT 
                commodity,
                COUNT(*) as record_count,
                SUM(volume) as total_volume,
                AVG(volume) as avg_volume
            FROM ogorb_production_data
            WHERE volume IS NOT NULL
            GROUP BY commodity
            ORDER BY total_volume DESC
            LIMIT 10
        """)
        
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]:,}건, 총 {row[2]:,}, 평균 {row[3]:,.2f}")
        
        # 상위 10개 주별 통계
        print(f"\n=== 상위 10개 주별 통계 ===")
        cursor.execute("""
            SELECT 
                state,
                COUNT(*) as record_count,
                SUM(volume) as total_volume
            FROM ogorb_production_data
            WHERE volume IS NOT NULL AND state IS NOT NULL
            GROUP BY state
            ORDER BY total_volume DESC
            LIMIT 10
        """)
        
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]:,}건, 총 {row[2]:,}")
        
        # 토지 분류별 통계
        print(f"\n=== 토지 분류별 통계 ===")
        cursor.execute("""
            SELECT 
                land_class,
                COUNT(*) as record_count,
                SUM(volume) as total_volume
            FROM ogorb_production_data
            WHERE volume IS NOT NULL AND land_class IS NOT NULL
            GROUP BY land_class
            ORDER BY total_volume DESC
        """)
        
        for row in cursor.fetchall():
            print(f"{row[0]}: {row[1]:,}건, 총 {row[2]:,}")
        
        cursor.close()
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
    
    finally:
        conn.close()

if __name__ == "__main__":
    analyze_ogorb_data()
