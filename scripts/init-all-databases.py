#!/usr/bin/env python3
"""모든 데이터베이스 초기화 스크립트 - Bridge 개발 환경용 샘플 데이터 생성"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any

# 데이터베이스 클라이언트들
import asyncpg
import aiomysql
from elasticsearch import AsyncElasticsearch

async def init_postgres():
    """PostgreSQL 데이터베이스 초기화"""
    print("🐘 PostgreSQL 초기화 시작...")
    
    # PostgreSQL 연결 설정
    postgres_config = {
        "host": os.getenv("BRIDGE_POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("BRIDGE_POSTGRES_PORT", "5432")),
        "database": os.getenv("BRIDGE_POSTGRES_DB", "bridge_dev"),
        "user": os.getenv("BRIDGE_POSTGRES_USER", "bridge_user"),
        "password": os.getenv("BRIDGE_POSTGRES_PASSWORD", "bridge_password"),
    }
    
    try:
        # 연결 테스트
        conn = await asyncpg.connect(**postgres_config)
        print("✅ PostgreSQL 연결 성공")
        
        # analytics_db 데이터베이스 생성
        await conn.execute("CREATE DATABASE analytics_db;")
        print("✅ analytics_db 데이터베이스 생성")
        
        # analytics_db로 전환
        await conn.close()
        postgres_config["database"] = "analytics_db"
        conn = await asyncpg.connect(**postgres_config)
        
        # 테이블 생성
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                age INTEGER,
                city VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                product_name VARCHAR(100) NOT NULL,
                amount DECIMAL(10,2) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                category VARCHAR(50),
                price DECIMAL(10,2) NOT NULL,
                stock_quantity INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS customer_segments (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                segment_name VARCHAR(50) NOT NULL,
                score DECIMAL(5,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        print("✅ PostgreSQL 테이블 생성 완료")
        
        # 기존 데이터 삭제
        await conn.execute("DELETE FROM customer_segments;")
        await conn.execute("DELETE FROM orders;")
        await conn.execute("DELETE FROM products;")
        await conn.execute("DELETE FROM users;")
        
        # 샘플 데이터 삽입
        users_data = [
            ('김철수', 'kim@example.com', 28, '서울'),
            ('이영희', 'lee@example.com', 35, '부산'),
            ('박민수', 'park@example.com', 42, '대구'),
            ('최지영', 'choi@example.com', 29, '서울'),
            ('정수현', 'jung@example.com', 31, '인천'),
            ('한소영', 'han@example.com', 26, '광주'),
            ('윤태호', 'yoon@example.com', 38, '대전'),
            ('강미영', 'kang@example.com', 33, '울산'),
        ]
        
        for user in users_data:
            await conn.execute(
                "INSERT INTO users (name, email, age, city) VALUES ($1, $2, $3, $4)",
                *user
            )
        
        products_data = [
            ('노트북', '전자제품', 1200000.00, 50),
            ('마우스', '전자제품', 25000.00, 200),
            ('키보드', '전자제품', 80000.00, 150),
            ('모니터', '전자제품', 300000.00, 75),
            ('책상', '가구', 150000.00, 30),
            ('의자', '가구', 200000.00, 25),
            ('스마트폰', '전자제품', 800000.00, 100),
            ('태블릿', '전자제품', 500000.00, 60),
        ]
        
        for product in products_data:
            await conn.execute(
                "INSERT INTO products (name, category, price, stock_quantity) VALUES ($1, $2, $3, $4)",
                *product
            )
        
        orders_data = [
            (1, '노트북', 1200000.00, 'completed', '2024-01-15 10:30:00'),
            (1, '마우스', 25000.00, 'completed', '2024-01-15 10:35:00'),
            (2, '키보드', 80000.00, 'completed', '2024-01-16 14:20:00'),
            (3, '모니터', 300000.00, 'pending', '2024-01-17 09:15:00'),
            (4, '책상', 150000.00, 'completed', '2024-01-18 16:45:00'),
            (5, '의자', 200000.00, 'shipped', '2024-01-19 11:30:00'),
            (6, '스마트폰', 800000.00, 'completed', '2024-01-20 13:20:00'),
            (7, '태블릿', 500000.00, 'completed', '2024-01-21 15:10:00'),
            (8, '노트북', 1200000.00, 'pending', '2024-01-22 10:00:00'),
            (1, '키보드', 80000.00, 'completed', '2024-01-23 14:30:00'),
        ]
        
        for order in orders_data:
            await conn.execute(
                "INSERT INTO orders (user_id, product_name, amount, status, order_date) VALUES ($1, $2, $3, $4, $5)",
                *order
            )
        
        segments_data = [
            (1, 'VIP', 95.5),
            (2, 'Premium', 85.2),
            (3, 'Standard', 65.8),
            (4, 'Premium', 88.1),
            (5, 'Standard', 72.3),
            (6, 'VIP', 92.7),
            (7, 'Premium', 86.4),
            (8, 'Standard', 68.9),
        ]
        
        for segment in segments_data:
            await conn.execute(
                "INSERT INTO customer_segments (user_id, segment_name, score) VALUES ($1, $2, $3)",
                *segment
            )
        
        print(f"✅ PostgreSQL 샘플 데이터 삽입 완료: {len(users_data)}명 사용자, {len(products_data)}개 제품, {len(orders_data)}개 주문")
        
        await conn.close()
        return True
        
    except Exception as e:
        print(f"❌ PostgreSQL 초기화 실패: {e}")
        return False

async def init_mysql():
    """MySQL 데이터베이스 초기화"""
    print("🐬 MySQL 초기화 시작...")
    
    # MySQL 연결 설정
    mysql_config = {
        "host": os.getenv("BRIDGE_MYSQL_HOST", "localhost"),
        "port": int(os.getenv("BRIDGE_MYSQL_PORT", "3306")),
        "db": os.getenv("BRIDGE_MYSQL_DB", "bridge_dev"),
        "user": os.getenv("BRIDGE_MYSQL_USER", "bridge_user"),
        "password": os.getenv("BRIDGE_MYSQL_PASSWORD", "bridge_password"),
    }
    
    try:
        # 연결 테스트
        conn = await aiomysql.connect(**mysql_config)
        print("✅ MySQL 연결 성공")
        
        cursor = await conn.cursor()
        
        # 테이블 생성
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                age INT,
                city VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            );
        """)
        
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                product_name VARCHAR(100) NOT NULL,
                amount DECIMAL(10,2) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending',
                order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                category VARCHAR(50),
                price DECIMAL(10,2) NOT NULL,
                stock_quantity INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        await cursor.execute("""
            CREATE TABLE IF NOT EXISTS customer_segments (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                segment_name VARCHAR(50) NOT NULL,
                score DECIMAL(5,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)
        
        print("✅ MySQL 테이블 생성 완료")
        
        # 기존 데이터 삭제
        await cursor.execute("DELETE FROM customer_segments;")
        await cursor.execute("DELETE FROM orders;")
        await cursor.execute("DELETE FROM products;")
        await cursor.execute("DELETE FROM users;")
        
        # 샘플 데이터 삽입
        users_data = [
            ('김철수', 'kim@example.com', 28, '서울'),
            ('이영희', 'lee@example.com', 35, '부산'),
            ('박민수', 'park@example.com', 42, '대구'),
            ('최지영', 'choi@example.com', 29, '서울'),
            ('정수현', 'jung@example.com', 31, '인천'),
            ('한소영', 'han@example.com', 26, '광주'),
            ('윤태호', 'yoon@example.com', 38, '대전'),
            ('강미영', 'kang@example.com', 33, '울산'),
        ]
        
        await cursor.executemany(
            "INSERT INTO users (name, email, age, city) VALUES (%s, %s, %s, %s)",
            users_data
        )
        
        products_data = [
            ('노트북', '전자제품', 1200000.00, 50),
            ('마우스', '전자제품', 25000.00, 200),
            ('키보드', '전자제품', 80000.00, 150),
            ('모니터', '전자제품', 300000.00, 75),
            ('책상', '가구', 150000.00, 30),
            ('의자', '가구', 200000.00, 25),
            ('스마트폰', '전자제품', 800000.00, 100),
            ('태블릿', '전자제품', 500000.00, 60),
        ]
        
        await cursor.executemany(
            "INSERT INTO products (name, category, price, stock_quantity) VALUES (%s, %s, %s, %s)",
            products_data
        )
        
        orders_data = [
            (1, '노트북', 1200000.00, 'completed', '2024-01-15 10:30:00'),
            (1, '마우스', 25000.00, 'completed', '2024-01-15 10:35:00'),
            (2, '키보드', 80000.00, 'completed', '2024-01-16 14:20:00'),
            (3, '모니터', 300000.00, 'pending', '2024-01-17 09:15:00'),
            (4, '책상', 150000.00, 'completed', '2024-01-18 16:45:00'),
            (5, '의자', 200000.00, 'shipped', '2024-01-19 11:30:00'),
            (6, '스마트폰', 800000.00, 'completed', '2024-01-20 13:20:00'),
            (7, '태블릿', 500000.00, 'completed', '2024-01-21 15:10:00'),
            (8, '노트북', 1200000.00, 'pending', '2024-01-22 10:00:00'),
            (1, '키보드', 80000.00, 'completed', '2024-01-23 14:30:00'),
        ]
        
        await cursor.executemany(
            "INSERT INTO orders (user_id, product_name, amount, status, order_date) VALUES (%s, %s, %s, %s, %s)",
            orders_data
        )
        
        segments_data = [
            (1, 'VIP', 95.5),
            (2, 'Premium', 85.2),
            (3, 'Standard', 65.8),
            (4, 'Premium', 88.1),
            (5, 'Standard', 72.3),
            (6, 'VIP', 92.7),
            (7, 'Premium', 86.4),
            (8, 'Standard', 68.9),
        ]
        
        await cursor.executemany(
            "INSERT INTO customer_segments (user_id, segment_name, score) VALUES (%s, %s, %s)",
            segments_data
        )
        
        await conn.commit()
        print(f"✅ MySQL 샘플 데이터 삽입 완료: {len(users_data)}명 사용자, {len(products_data)}개 제품, {len(orders_data)}개 주문")
        
        await cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ MySQL 초기화 실패: {e}")
        return False

async def init_elasticsearch():
    """Elasticsearch 데이터베이스 초기화"""
    print("🔍 Elasticsearch 초기화 시작...")
    
    # Elasticsearch 클라이언트 설정
    es_config = {
        "hosts": [f"{os.getenv('BRIDGE_ELASTICSEARCH_HOST', 'localhost')}:{os.getenv('BRIDGE_ELASTICSEARCH_PORT', '9200')}"],
        "verify_certs": False,
        "request_timeout": 30,
    }
    
    # 인증 정보가 있는 경우 추가
    username = os.getenv('BRIDGE_ELASTICSEARCH_USERNAME')
    password = os.getenv('BRIDGE_ELASTICSEARCH_PASSWORD')
    if username and password:
        es_config["basic_auth"] = (username, password)
    
    es = AsyncElasticsearch(**es_config)
    
    try:
        # 연결 테스트
        if not await es.ping():
            print("❌ Elasticsearch 연결 실패")
            return False
        
        print("✅ Elasticsearch 연결 성공")
        
        # 기존 인덱스 삭제 (개발 환경이므로)
        indices_to_delete = ["users", "orders", "products", "customer_segments"]
        for index in indices_to_delete:
            try:
                if await es.indices.exists(index=index):
                    await es.indices.delete(index=index)
                    print(f"🗑️  기존 인덱스 삭제: {index}")
            except Exception as e:
                print(f"⚠️  인덱스 삭제 중 오류 ({index}): {e}")
        
        # 1. 사용자 인덱스 생성
        users_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "text", "analyzer": "korean"},
                    "email": {"type": "keyword"},
                    "age": {"type": "integer"},
                    "city": {"type": "keyword"},
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }
        
        await es.indices.create(index="users", body=users_mapping)
        print("✅ users 인덱스 생성")
        
        # 사용자 데이터 삽입
        users_data = [
            {"id": 1, "name": "김철수", "email": "kim@example.com", "age": 28, "city": "서울", "created_at": "2024-01-01T00:00:00Z"},
            {"id": 2, "name": "이영희", "email": "lee@example.com", "age": 35, "city": "부산", "created_at": "2024-01-02T00:00:00Z"},
            {"id": 3, "name": "박민수", "email": "park@example.com", "age": 42, "city": "대구", "created_at": "2024-01-03T00:00:00Z"},
            {"id": 4, "name": "최지영", "email": "choi@example.com", "age": 29, "city": "서울", "created_at": "2024-01-04T00:00:00Z"},
            {"id": 5, "name": "정수현", "email": "jung@example.com", "age": 31, "city": "인천", "created_at": "2024-01-05T00:00:00Z"},
            {"id": 6, "name": "한소영", "email": "han@example.com", "age": 26, "city": "광주", "created_at": "2024-01-06T00:00:00Z"},
            {"id": 7, "name": "윤태호", "email": "yoon@example.com", "age": 38, "city": "대전", "created_at": "2024-01-07T00:00:00Z"},
            {"id": 8, "name": "강미영", "email": "kang@example.com", "age": 33, "city": "울산", "created_at": "2024-01-08T00:00:00Z"},
        ]
        
        for user in users_data:
            await es.index(index="users", id=user["id"], body=user)
        print(f"✅ {len(users_data)}명의 사용자 데이터 삽입")
        
        # 2. 제품 인덱스 생성
        products_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "text", "analyzer": "korean"},
                    "category": {"type": "keyword"},
                    "price": {"type": "float"},
                    "stock_quantity": {"type": "integer"},
                    "created_at": {"type": "date"}
                }
            }
        }
        
        await es.indices.create(index="products", body=products_mapping)
        print("✅ products 인덱스 생성")
        
        # 제품 데이터 삽입
        products_data = [
            {"id": 1, "name": "노트북", "category": "전자제품", "price": 1200000.0, "stock_quantity": 50, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 2, "name": "마우스", "category": "전자제품", "price": 25000.0, "stock_quantity": 200, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 3, "name": "키보드", "category": "전자제품", "price": 80000.0, "stock_quantity": 150, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 4, "name": "모니터", "category": "전자제품", "price": 300000.0, "stock_quantity": 75, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 5, "name": "책상", "category": "가구", "price": 150000.0, "stock_quantity": 30, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 6, "name": "의자", "category": "가구", "price": 200000.0, "stock_quantity": 25, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 7, "name": "스마트폰", "category": "전자제품", "price": 800000.0, "stock_quantity": 100, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 8, "name": "태블릿", "category": "전자제품", "price": 500000.0, "stock_quantity": 60, "created_at": "2024-01-01T00:00:00Z"},
        ]
        
        for product in products_data:
            await es.index(index="products", id=product["id"], body=product)
        print(f"✅ {len(products_data)}개의 제품 데이터 삽입")
        
        # 3. 주문 인덱스 생성
        orders_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "user_id": {"type": "integer"},
                    "product_name": {"type": "text", "analyzer": "korean"},
                    "amount": {"type": "float"},
                    "status": {"type": "keyword"},
                    "order_date": {"type": "date"}
                }
            }
        }
        
        await es.indices.create(index="orders", body=orders_mapping)
        print("✅ orders 인덱스 생성")
        
        # 주문 데이터 삽입
        orders_data = [
            {"id": 1, "user_id": 1, "product_name": "노트북", "amount": 1200000.0, "status": "completed", "order_date": "2024-01-15T10:30:00Z"},
            {"id": 2, "user_id": 1, "product_name": "마우스", "amount": 25000.0, "status": "completed", "order_date": "2024-01-15T10:35:00Z"},
            {"id": 3, "user_id": 2, "product_name": "키보드", "amount": 80000.0, "status": "completed", "order_date": "2024-01-16T14:20:00Z"},
            {"id": 4, "user_id": 3, "product_name": "모니터", "amount": 300000.0, "status": "pending", "order_date": "2024-01-17T09:15:00Z"},
            {"id": 5, "user_id": 4, "product_name": "책상", "amount": 150000.0, "status": "completed", "order_date": "2024-01-18T16:45:00Z"},
            {"id": 6, "user_id": 5, "product_name": "의자", "amount": 200000.0, "status": "shipped", "order_date": "2024-01-19T11:30:00Z"},
            {"id": 7, "user_id": 6, "product_name": "스마트폰", "amount": 800000.0, "status": "completed", "order_date": "2024-01-20T13:20:00Z"},
            {"id": 8, "user_id": 7, "product_name": "태블릿", "amount": 500000.0, "status": "completed", "order_date": "2024-01-21T15:10:00Z"},
            {"id": 9, "user_id": 8, "product_name": "노트북", "amount": 1200000.0, "status": "pending", "order_date": "2024-01-22T10:00:00Z"},
            {"id": 10, "user_id": 1, "product_name": "키보드", "amount": 80000.0, "status": "completed", "order_date": "2024-01-23T14:30:00Z"},
        ]
        
        for order in orders_data:
            await es.index(index="orders", id=order["id"], body=order)
        print(f"✅ {len(orders_data)}개의 주문 데이터 삽입")
        
        # 4. 고객 세그먼트 인덱스 생성
        segments_mapping = {
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "user_id": {"type": "integer"},
                    "segment_name": {"type": "keyword"},
                    "score": {"type": "float"},
                    "created_at": {"type": "date"}
                }
            }
        }
        
        await es.indices.create(index="customer_segments", body=segments_mapping)
        print("✅ customer_segments 인덱스 생성")
        
        # 고객 세그먼트 데이터 삽입
        segments_data = [
            {"id": 1, "user_id": 1, "segment_name": "VIP", "score": 95.5, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 2, "user_id": 2, "segment_name": "Premium", "score": 85.2, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 3, "user_id": 3, "segment_name": "Standard", "score": 65.8, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 4, "user_id": 4, "segment_name": "Premium", "score": 88.1, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 5, "user_id": 5, "segment_name": "Standard", "score": 72.3, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 6, "user_id": 6, "segment_name": "VIP", "score": 92.7, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 7, "user_id": 7, "segment_name": "Premium", "score": 86.4, "created_at": "2024-01-01T00:00:00Z"},
            {"id": 8, "user_id": 8, "segment_name": "Standard", "score": 68.9, "created_at": "2024-01-01T00:00:00Z"},
        ]
        
        for segment in segments_data:
            await es.index(index="customer_segments", id=segment["id"], body=segment)
        print(f"✅ {len(segments_data)}개의 고객 세그먼트 데이터 삽입")
        
        # 인덱스 새로고침
        await es.indices.refresh(index="_all")
        print("✅ 모든 인덱스 새로고침 완료")
        
        # 통계 출력
        stats = await es.indices.stats(index="_all")
        total_docs = sum(index_stats["total"]["docs"]["count"] for index_stats in stats["indices"].values())
        print(f"📊 총 문서 수: {total_docs}")
        
        await es.close()
        return True
        
    except Exception as e:
        print(f"❌ Elasticsearch 초기화 실패: {e}")
        return False

async def main():
    """메인 함수 - 모든 데이터베이스 초기화"""
    print("🚀 모든 데이터베이스 초기화 시작...")
    print("=" * 50)
    
    # 각 데이터베이스 초기화
    postgres_success = await init_postgres()
    print()
    
    mysql_success = await init_mysql()
    print()
    
    elasticsearch_success = await init_elasticsearch()
    print()
    
    print("=" * 50)
    print("📊 초기화 결과 요약:")
    print(f"  PostgreSQL: {'✅ 성공' if postgres_success else '❌ 실패'}")
    print(f"  MySQL:      {'✅ 성공' if mysql_success else '❌ 실패'}")
    print(f"  Elasticsearch: {'✅ 성공' if elasticsearch_success else '❌ 실패'}")
    
    if postgres_success and mysql_success and elasticsearch_success:
        print("\n🎉 모든 데이터베이스 초기화 완료!")
        print("이제 MCP 서버를 통해 실제 데이터베이스에 연결할 수 있습니다.")
        sys.exit(0)
    else:
        print("\n❌ 일부 데이터베이스 초기화 실패!")
        print("Docker 서비스들이 실행 중인지 확인해주세요.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
