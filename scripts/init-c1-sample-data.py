#!/usr/bin/env python3
"""C1 마일스톤용 대용량 샘플 데이터 생성 스크립트 - Bridge Analytics 테스트용"""

import asyncio
import json
import os
import sys
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from faker import Faker
import numpy as np

# 데이터베이스 클라이언트들
import asyncpg
import aiomysql
from elasticsearch import Elasticsearch

# 한국어 Faker 설정
fake = Faker('ko_KR')

class C1SampleDataGenerator:
    """C1 마일스톤용 샘플 데이터 생성기"""
    
    def __init__(self, scale: str = "medium"):
        """
        Args:
            scale: 데이터 크기 ("small", "medium", "large", "xlarge")
        """
        self.scale = scale
        self.data_sizes = {
            "small": {"customers": 1000, "products": 100, "orders": 5000, "sales": 3000, "activities": 2000},
            "medium": {"customers": 10000, "products": 500, "orders": 50000, "sales": 30000, "activities": 20000},
            "large": {"customers": 100000, "products": 2000, "orders": 500000, "sales": 300000, "activities": 200000},
            "xlarge": {"customers": 1000000, "products": 10000, "orders": 5000000, "sales": 3000000, "activities": 2000000}
        }
        self.sizes = self.data_sizes[scale]
        
        # 한국 도시 목록
        self.korean_cities = [
            "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
            "수원", "고양", "용인", "성남", "부천", "화성", "안산", "안양",
            "평택", "시흥", "김포", "의정부", "광명", "과천", "오산", "의왕",
            "하남", "이천", "안성", "포천", "여주", "양평", "동두천", "가평",
            "연천", "춘천", "원주", "강릉", "태백", "속초", "삼척", "홍천",
            "횡성", "영월", "평창", "정선", "철원", "화천", "양구", "인제"
        ]
        
        # 상품 카테고리
        self.product_categories = [
            "전자제품", "가구", "의류", "화장품", "식품", "도서", "스포츠용품",
            "자동차용품", "생활용품", "완구", "반려동물용품", "건강용품"
        ]
        
        # 지역 정보
        self.regions = [
            {"name": "서울", "country": "한국", "population": 9720846, "gdp_per_capita": 35000},
            {"name": "부산", "country": "한국", "population": 3448737, "gdp_per_capita": 28000},
            {"name": "대구", "country": "한국", "population": 2413076, "gdp_per_capita": 25000},
            {"name": "인천", "country": "한국", "population": 2954314, "gdp_per_capita": 30000},
            {"name": "광주", "country": "한국", "population": 1441970, "gdp_per_capita": 22000},
            {"name": "대전", "country": "한국", "population": 1441970, "gdp_per_capita": 26000},
            {"name": "울산", "country": "한국", "population": 1142190, "gdp_per_capita": 32000},
            {"name": "세종", "country": "한국", "population": 365339, "gdp_per_capita": 40000}
        ]

    async def init_postgres_c1_data(self):
        """PostgreSQL에 C1용 고객 데이터 생성"""
        print("🐘 PostgreSQL C1 데이터 생성 시작...")
        
        postgres_config = {
            "host": os.getenv("BRIDGE_POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("BRIDGE_POSTGRES_PORT", "5432")),
            "database": os.getenv("BRIDGE_POSTGRES_DB", "bridge_dev"),
            "user": os.getenv("BRIDGE_POSTGRES_USER", "bridge_user"),
            "password": os.getenv("BRIDGE_POSTGRES_PASSWORD", "bridge_password"),
        }
        
        try:
            conn = await asyncpg.connect(**postgres_config)
            print("✅ PostgreSQL 연결 성공")
            
            # analytics_db 데이터베이스 생성
            try:
                await conn.execute("CREATE DATABASE analytics_db;")
            except asyncpg.exceptions.DuplicateDatabaseError:
                pass  # 데이터베이스가 이미 존재하는 경우 무시
            await conn.close()
            
            postgres_config["database"] = "analytics_db"
            conn = await asyncpg.connect(**postgres_config)
            
            # 테이블 생성
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS customers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    age INTEGER,
                    city VARCHAR(50),
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    vip_status BOOLEAN DEFAULT FALSE,
                    total_spent DECIMAL(12,2) DEFAULT 0,
                    last_purchase_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    category VARCHAR(50),
                    price DECIMAL(10,2) NOT NULL,
                    stock_quantity INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(id),
                    product_id INTEGER REFERENCES products(id),
                    amount DECIMAL(10,2) NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'pending',
                    payment_method VARCHAR(50),
                    shipping_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS customer_segments (
                    id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(id),
                    segment_name VARCHAR(50) NOT NULL,
                    score DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            print("✅ PostgreSQL 테이블 생성 완료")
            
            # 기존 데이터 삭제 (외래키 제약조건 고려하여 순서대로)
            # 테이블을 드롭하고 재생성하여 완전히 초기화
            await conn.execute("DROP TABLE IF EXISTS customer_segments CASCADE;")
            await conn.execute("DROP TABLE IF EXISTS orders CASCADE;")
            await conn.execute("DROP TABLE IF EXISTS products CASCADE;")
            await conn.execute("DROP TABLE IF EXISTS customers CASCADE;")
            
            # 테이블 재생성
            await conn.execute("""
                CREATE TABLE customers (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    email VARCHAR(100) UNIQUE NOT NULL,
                    age INTEGER,
                    city VARCHAR(50),
                    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    vip_status BOOLEAN DEFAULT FALSE,
                    total_spent DECIMAL(12,2) DEFAULT 0,
                    last_purchase_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await conn.execute("""
                CREATE TABLE products (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    category VARCHAR(50),
                    price DECIMAL(10,2) NOT NULL,
                    stock_quantity INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await conn.execute("""
                CREATE TABLE orders (
                    id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(id),
                    product_id INTEGER REFERENCES products(id),
                    amount DECIMAL(10,2) NOT NULL,
                    quantity INTEGER NOT NULL,
                    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status VARCHAR(20) DEFAULT 'pending',
                    payment_method VARCHAR(50),
                    shipping_address TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await conn.execute("""
                CREATE TABLE customer_segments (
                    id SERIAL PRIMARY KEY,
                    customer_id INTEGER REFERENCES customers(id),
                    segment_name VARCHAR(50) NOT NULL,
                    score DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # 고객 데이터 생성
            print(f"📊 고객 데이터 생성 중... ({self.sizes['customers']:,}명)")
            customers_data = []
            used_emails = set()  # 중복 이메일 방지
            
            for i in range(self.sizes['customers']):
                # 연령 분포: 20-70세, 정규분포
                age = max(20, min(70, int(np.random.normal(40, 15))))
                
                # VIP 고객 비율 5%
                vip_status = random.random() < 0.05
                
                # 등록일: 최근 3년 내
                registration_date = fake.date_time_between(start_date='-3y', end_date='now')
                
                # 고유한 이메일 생성
                email = fake.email()
                while email in used_emails:
                    email = fake.email()
                used_emails.add(email)
                
                customers_data.append((
                    fake.name(),
                    email,
                    age,
                    random.choice(self.korean_cities),
                    registration_date,
                    vip_status,
                    0,  # total_spent는 나중에 계산
                    None  # last_purchase_date는 나중에 설정
                ))
            
            # 고객 데이터 삽입
            await conn.executemany(
                "INSERT INTO customers (name, email, age, city, registration_date, vip_status, total_spent, last_purchase_date) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                customers_data
            )
            print(f"✅ {len(customers_data):,}명의 고객 데이터 삽입 완료")
            
            # 상품 데이터 생성
            print(f"📊 상품 데이터 생성 중... ({self.sizes['products']:,}개)")
            products_data = []
            for i in range(self.sizes['products']):
                category = random.choice(self.product_categories)
                
                # 카테고리별 가격 분포
                if category == "전자제품":
                    price = random.uniform(50000, 2000000)
                elif category == "가구":
                    price = random.uniform(100000, 1000000)
                elif category == "의류":
                    price = random.uniform(10000, 200000)
                elif category == "화장품":
                    price = random.uniform(5000, 100000)
                else:
                    price = random.uniform(1000, 500000)
                
                products_data.append((
                    fake.catch_phrase(),
                    category,
                    round(price, 2),
                    random.randint(0, 1000)
                ))
            
            await conn.executemany(
                "INSERT INTO products (name, category, price, stock_quantity) VALUES ($1, $2, $3, $4)",
                products_data
            )
            print(f"✅ {len(products_data):,}개의 상품 데이터 삽입 완료")
            
            # 주문 데이터 생성
            print(f"📊 주문 데이터 생성 중... ({self.sizes['orders']:,}개)")
            orders_data = []
            customer_totals = {}  # 고객별 총 구매액 추적
            
            for i in range(self.sizes['orders']):
                customer_id = random.randint(1, self.sizes['customers'])
                product_id = random.randint(1, self.sizes['products'])
                
                # 상품 가격 조회 (실제로는 JOIN이지만 여기서는 근사치)
                base_price = random.uniform(1000, 1000000)
                quantity = random.randint(1, 10)
                amount = base_price * quantity
                
                # 주문일: 최근 2년 내
                order_date = fake.date_time_between(start_date='-2y', end_date='now')
                
                # 결제 방법
                payment_methods = ['카드', '현금', '계좌이체', '간편결제', '포인트']
                payment_method = random.choice(payment_methods)
                
                # 주문 상태
                statuses = ['completed', 'pending', 'shipped', 'cancelled']
                status_weights = [0.7, 0.1, 0.15, 0.05]  # completed가 가장 많음
                status = np.random.choice(statuses, p=status_weights)
                
                orders_data.append((
                    customer_id,
                    product_id,
                    round(amount, 2),
                    quantity,
                    order_date,
                    status,
                    payment_method,
                    fake.address()
                ))
                
                # 고객별 총 구매액 업데이트
                if status == 'completed':
                    if customer_id not in customer_totals:
                        customer_totals[customer_id] = 0
                    customer_totals[customer_id] += amount
            
            await conn.executemany(
                "INSERT INTO orders (customer_id, product_id, amount, quantity, order_date, status, payment_method, shipping_address) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                orders_data
            )
            print(f"✅ {len(orders_data):,}개의 주문 데이터 삽입 완료")
            
            # 고객별 총 구매액 및 마지막 구매일 업데이트
            print("📊 고객 통계 업데이트 중...")
            for customer_id, total_spent in customer_totals.items():
                # 마지막 구매일 조회
                last_purchase = await conn.fetchval(
                    "SELECT MAX(order_date) FROM orders WHERE customer_id = $1 AND status = 'completed'",
                    customer_id
                )
                
                await conn.execute(
                    "UPDATE customers SET total_spent = $1, last_purchase_date = $2 WHERE id = $3",
                    total_spent, last_purchase, customer_id
                )
            
            # 고객 세그먼트 생성
            print("📊 고객 세그먼트 생성 중...")
            segments_data = []
            for customer_id in range(1, self.sizes['customers'] + 1):
                # 고객 정보 조회
                customer = await conn.fetchrow("SELECT total_spent, vip_status FROM customers WHERE id = $1", customer_id)
                if customer:
                    total_spent = customer['total_spent'] or 0
                    vip_status = customer['vip_status']
                    
                    # 세그먼트 결정
                    if vip_status or total_spent > 1000000:
                        segment_name = "VIP"
                        score = random.uniform(90, 100)
                    elif total_spent > 500000:
                        segment_name = "Premium"
                        score = random.uniform(80, 89)
                    elif total_spent > 100000:
                        segment_name = "Standard"
                        score = random.uniform(60, 79)
                    else:
                        segment_name = "Basic"
                        score = random.uniform(40, 59)
                    
                    segments_data.append((customer_id, segment_name, round(score, 2)))
            
            await conn.executemany(
                "INSERT INTO customer_segments (customer_id, segment_name, score) VALUES ($1, $2, $3)",
                segments_data
            )
            print(f"✅ {len(segments_data):,}개의 고객 세그먼트 생성 완료")
            
            await conn.close()
            return True
            
        except Exception as e:
            print(f"❌ PostgreSQL C1 데이터 생성 실패: {e}")
            return False

    async def init_mysql_c1_data(self):
        """MySQL에 C1용 매출 데이터 생성"""
        print("🐬 MySQL C1 데이터 생성 시작...")
        
        mysql_config = {
            "host": os.getenv("BRIDGE_MYSQL_HOST", "localhost"),
            "port": int(os.getenv("BRIDGE_MYSQL_PORT", "3306")),
            "db": os.getenv("BRIDGE_MYSQL_DB", "bridge_dev"),
            "user": os.getenv("BRIDGE_MYSQL_USER", "bridge_user"),
            "password": os.getenv("BRIDGE_MYSQL_PASSWORD", "bridge_password"),
        }
        
        try:
            conn = await aiomysql.connect(**mysql_config)
            print("✅ MySQL 연결 성공")
            cursor = await conn.cursor()
            
            # 테이블 생성
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS sales (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    product_id INT NOT NULL,
                    amount DECIMAL(12,2) NOT NULL,
                    quantity INT NOT NULL,
                    sale_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    region_id INT,
                    salesperson_id INT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS regions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    country VARCHAR(50) NOT NULL,
                    population INT,
                    gdp_per_capita DECIMAL(10,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS salespeople (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(100) NOT NULL,
                    department VARCHAR(50),
                    hire_date DATE,
                    salary DECIMAL(10,2),
                    performance_score DECIMAL(5,2),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            print("✅ MySQL 테이블 생성 완료")
            
            # 기존 데이터 삭제
            await cursor.execute("DELETE FROM sales;")
            await cursor.execute("DELETE FROM salespeople;")
            await cursor.execute("DELETE FROM regions;")
            
            # 지역 데이터 생성
            print("📊 지역 데이터 생성 중...")
            regions_data = []
            for i, region in enumerate(self.regions, 1):
                regions_data.append((
                    region['name'],
                    region['country'],
                    region['population'],
                    region['gdp_per_capita']
                ))
            
            await cursor.executemany(
                "INSERT INTO regions (name, country, population, gdp_per_capita) VALUES (%s, %s, %s, %s)",
                regions_data
            )
            print(f"✅ {len(regions_data)}개의 지역 데이터 삽입 완료")
            
            # 영업사원 데이터 생성
            print(f"📊 영업사원 데이터 생성 중... ({self.sizes['customers'] // 100:,}명)")
            salespeople_data = []
            departments = ['영업1팀', '영업2팀', '영업3팀', '마케팅팀', '고객관리팀']
            
            for i in range(self.sizes['customers'] // 100):  # 고객 100명당 영업사원 1명
                hire_date = fake.date_between(start_date='-5y', end_date='-1y')
                salary = random.uniform(30000000, 80000000)  # 3천만원 ~ 8천만원
                performance_score = random.uniform(60, 100)
                
                salespeople_data.append((
                    fake.name(),
                    random.choice(departments),
                    hire_date,
                    round(salary, 2),
                    round(performance_score, 2)
                ))
            
            await cursor.executemany(
                "INSERT INTO salespeople (name, department, hire_date, salary, performance_score) VALUES (%s, %s, %s, %s, %s)",
                salespeople_data
            )
            print(f"✅ {len(salespeople_data):,}명의 영업사원 데이터 삽입 완료")
            
            # 매출 데이터 생성
            print(f"📊 매출 데이터 생성 중... ({self.sizes['sales']:,}개)")
            sales_data = []
            
            for i in range(self.sizes['sales']):
                # 상품 ID (PostgreSQL의 products 테이블과 연동)
                product_id = random.randint(1, self.sizes['products'])
                
                # 매출 금액 (정규분포 + 이상치)
                if random.random() < 0.02:  # 2% 확률로 이상치
                    amount = random.uniform(1000000, 10000000)  # 100만원 ~ 1000만원
                else:
                    amount = random.uniform(10000, 500000)  # 1만원 ~ 50만원
                
                quantity = random.randint(1, 20)
                region_id = random.randint(1, len(self.regions))
                salesperson_id = random.randint(1, len(salespeople_data))
                
                # 매출일: 최근 2년 내
                sale_date = fake.date_time_between(start_date='-2y', end_date='now')
                
                sales_data.append((
                    product_id,
                    round(amount, 2),
                    quantity,
                    sale_date,
                    region_id,
                    salesperson_id
                ))
            
            await cursor.executemany(
                "INSERT INTO sales (product_id, amount, quantity, sale_date, region_id, salesperson_id) VALUES (%s, %s, %s, %s, %s, %s)",
                sales_data
            )
            print(f"✅ {len(sales_data):,}개의 매출 데이터 삽입 완료")
            
            await conn.commit()
            await cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ MySQL C1 데이터 생성 실패: {e}")
            return False

    async def init_elasticsearch_c1_data(self):
        """Elasticsearch에 C1용 로그 데이터 생성"""
        print("🔍 Elasticsearch C1 데이터 생성 시작...")
        
        # Elasticsearch URL 구성 - scheme 포함
        es_host = os.getenv('BRIDGE_ELASTICSEARCH_HOST', 'localhost')
        es_port = os.getenv('BRIDGE_ELASTICSEARCH_PORT', '9200')
        es_use_ssl = os.getenv('BRIDGE_ELASTICSEARCH_USE_SSL', 'false').lower() == 'true'
        
        scheme = 'https' if es_use_ssl else 'http'
        es_url = f"{scheme}://{es_host}:{es_port}"
        
        es_config = {
            "hosts": [es_url],
            "verify_certs": False,
            "request_timeout": 30,
            "retry_on_timeout": True,
            "max_retries": 3,
            "api_key": None,  # Disable API key
            "http_compress": False,  # Disable compression
        }
        
        username = os.getenv('BRIDGE_ELASTICSEARCH_USERNAME')
        password = os.getenv('BRIDGE_ELASTICSEARCH_PASSWORD')
        if username and password:
            es_config["basic_auth"] = (username, password)
        
        try:
            print(f"🔗 Elasticsearch 연결 시도: {es_url}")
            es = Elasticsearch(**es_config)
            
            # 연결 테스트
            try:
                ping_result = es.ping()
                if not ping_result:
                    print("❌ Elasticsearch 연결 실패 - ping 실패")
                    return False
            except Exception as e:
                print(f"❌ Elasticsearch 연결 실패 - ping 오류: {e}")
                print(f"🔧 연결 설정: {es_config}")
                return False
            
            print("✅ Elasticsearch 연결 성공")
            
            # 기존 인덱스 삭제
            indices_to_delete = ["user_activity", "error_logs", "sales_events"]
            for index in indices_to_delete:
                try:
                    if es.indices.exists(index=index):
                        es.indices.delete(index=index)
                        print(f"🗑️  기존 인덱스 삭제: {index}")
                except Exception as e:
                    print(f"⚠️  인덱스 삭제 중 오류 ({index}): {e}")
            
            # 1. 사용자 활동 로그 인덱스 생성
            user_activity_mapping = {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "user_id": {"type": "integer"},
                        "action": {"type": "keyword"},
                        "page": {"type": "text"},
                        "duration": {"type": "integer"},
                        "device": {"type": "keyword"},
                        "browser": {"type": "keyword"},
                        "location": {"type": "geo_point"},
                        "ip_address": {"type": "ip"},
                        "session_id": {"type": "keyword"}
                    }
                }
            }
            
            es.indices.create(index="user_activity", body=user_activity_mapping)
            print("✅ user_activity 인덱스 생성")
            
            # 사용자 활동 로그 데이터 생성
            print(f"📊 사용자 활동 로그 생성 중... ({self.sizes['activities']:,}개)")
            actions = ['login', 'logout', 'view_product', 'add_to_cart', 'purchase', 'search', 'browse']
            pages = ['/home', '/products', '/cart', '/checkout', '/profile', '/search', '/category']
            devices = ['desktop', 'mobile', 'tablet']
            browsers = ['Chrome', 'Firefox', 'Safari', 'Edge', 'Opera']
            
            for i in range(self.sizes['activities']):
                user_id = random.randint(1, self.sizes['customers'])
                action = random.choice(actions)
                page = random.choice(pages)
                duration = random.randint(1, 300)  # 1초 ~ 5분
                device = random.choice(devices)
                browser = random.choice(browsers)
                
                # 위치 (한국 내)
                lat = random.uniform(33.0, 38.5)
                lon = random.uniform(124.0, 132.0)
                
                activity_data = {
                    "timestamp": fake.date_time_between(start_date='-1y', end_date='now').isoformat() + "Z",
                    "user_id": user_id,
                    "action": action,
                    "page": page,
                    "duration": duration,
                    "device": device,
                    "browser": browser,
                    "location": {"lat": lat, "lon": lon},
                    "ip_address": fake.ipv4(),
                    "session_id": fake.uuid4()
                }
                
                es.index(index="user_activity", body=activity_data)
                
                if (i + 1) % 10000 == 0:
                    print(f"  진행률: {i + 1:,}/{self.sizes['activities']:,} ({((i + 1) / self.sizes['activities']) * 100:.1f}%)")
            
            print(f"✅ {self.sizes['activities']:,}개의 사용자 활동 로그 생성 완료")
            
            # 2. 에러 로그 인덱스 생성
            error_logs_mapping = {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "level": {"type": "keyword"},
                        "message": {"type": "text"},
                        "service": {"type": "keyword"},
                        "stack_trace": {"type": "text"},
                        "user_id": {"type": "integer"},
                        "request_id": {"type": "keyword"},
                        "error_code": {"type": "keyword"}
                    }
                }
            }
            
            es.indices.create(index="error_logs", body=error_logs_mapping)
            print("✅ error_logs 인덱스 생성")
            
            # 에러 로그 데이터 생성
            print(f"📊 에러 로그 생성 중... ({self.sizes['activities'] // 100:,}개)")
            levels = ['ERROR', 'WARNING', 'CRITICAL']
            services = ['api-gateway', 'user-service', 'order-service', 'payment-service', 'notification-service']
            error_codes = ['500', '404', '403', '401', '400', 'TIMEOUT', 'CONNECTION_ERROR']
            
            for i in range(self.sizes['activities'] // 100):  # 활동의 1%가 에러
                level = random.choice(levels)
                service = random.choice(services)
                error_code = random.choice(error_codes)
                user_id = random.randint(1, self.sizes['customers']) if random.random() < 0.7 else None
                
                error_data = {
                    "timestamp": fake.date_time_between(start_date='-1y', end_date='now').isoformat() + "Z",
                    "level": level,
                    "message": fake.sentence(),
                    "service": service,
                    "stack_trace": fake.text(max_nb_chars=500),
                    "user_id": user_id,
                    "request_id": fake.uuid4(),
                    "error_code": error_code
                }
                
                es.index(index="error_logs", body=error_data)
            
            print(f"✅ {self.sizes['activities'] // 100:,}개의 에러 로그 생성 완료")
            
            # 3. 매출 이벤트 인덱스 생성
            sales_events_mapping = {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "event_type": {"type": "keyword"},
                        "product_id": {"type": "integer"},
                        "amount": {"type": "float"},
                        "quantity": {"type": "integer"},
                        "region": {"type": "keyword"},
                        "salesperson_id": {"type": "integer"},
                        "customer_id": {"type": "integer"},
                        "metadata": {"type": "object"}
                    }
                }
            }
            
            es.indices.create(index="sales_events", body=sales_events_mapping)
            print("✅ sales_events 인덱스 생성")
            
            # 매출 이벤트 데이터 생성
            print(f"📊 매출 이벤트 생성 중... ({self.sizes['sales'] // 10:,}개)")
            event_types = ['sale_completed', 'sale_cancelled', 'refund_processed', 'payment_failed']
            
            for i in range(self.sizes['sales'] // 10):  # 매출의 10%가 이벤트
                event_type = random.choice(event_types)
                product_id = random.randint(1, self.sizes['products'])
                amount = random.uniform(10000, 1000000)
                quantity = random.randint(1, 10)
                region = random.choice(self.korean_cities)
                salesperson_id = random.randint(1, self.sizes['customers'] // 100)
                customer_id = random.randint(1, self.sizes['customers'])
                
                event_data = {
                    "timestamp": fake.date_time_between(start_date='-2y', end_date='now').isoformat() + "Z",
                    "event_type": event_type,
                    "product_id": product_id,
                    "amount": round(amount, 2),
                    "quantity": quantity,
                    "region": region,
                    "salesperson_id": salesperson_id,
                    "customer_id": customer_id,
                    "metadata": {
                        "payment_method": random.choice(['카드', '현금', '계좌이체']),
                        "discount_applied": random.random() < 0.3,
                        "promotion_code": fake.word() if random.random() < 0.2 else None
                    }
                }
                
                es.index(index="sales_events", body=event_data)
            
            print(f"✅ {self.sizes['sales'] // 10:,}개의 매출 이벤트 생성 완료")
            
            # 인덱스 새로고침
            es.indices.refresh(index="_all")
            print("✅ 모든 인덱스 새로고침 완료")
            
            # 통계 출력
            stats = es.indices.stats(index="_all")
            total_docs = sum(index_stats["total"]["docs"]["count"] for index_stats in stats["indices"].values())
            print(f"📊 총 문서 수: {total_docs:,}")
            
            es.close()
            return True
            
        except Exception as e:
            print(f"❌ Elasticsearch C1 데이터 생성 실패: {e}")
            return False

    def print_data_summary(self):
        """생성된 데이터 요약 출력"""
        print("\n" + "="*60)
        print("📊 C1 마일스톤 샘플 데이터 생성 완료!")
        print("="*60)
        print(f"📈 데이터 규모: {self.scale.upper()}")
        print(f"👥 고객 수: {self.sizes['customers']:,}명")
        print(f"🛍️  상품 수: {self.sizes['products']:,}개")
        print(f"📦 주문 수: {self.sizes['orders']:,}개")
        print(f"💰 매출 수: {self.sizes['sales']:,}개")
        print(f"📱 활동 로그: {self.sizes['activities']:,}개")
        print("\n🎯 테스트 가능한 C1 기능들:")
        print("  • 데이터 통합 (크로스 소스 조인)")
        print("  • 기본 통계 분석 (평균, 중앙값, 표준편차)")
        print("  • 상관관계 분석 (고객-상품, 지역-매출)")
        print("  • 이상치 탐지 (매출, 활동 패턴)")
        print("  • 데이터 품질 검사 (결측값, 중복값)")
        print("  • 시각화 (고객 세그먼트, 매출 트렌드)")
        print("  • 성능 벤치마크 (대용량 데이터 처리)")
        print("\n🚀 다음 단계:")
        print("  make c1-test      # C1 기능 테스트 실행")
        print("  make c1-benchmark # 성능 벤치마크 실행")

async def main():
    """메인 함수 - C1 마일스톤용 샘플 데이터 생성"""
    import argparse
    
    parser = argparse.ArgumentParser(description='C1 마일스톤용 샘플 데이터 생성')
    parser.add_argument('--scale', choices=['small', 'medium', 'large', 'xlarge'], 
                       default='medium', help='데이터 크기 (기본: medium)')
    args = parser.parse_args()
    
    print("🚀 C1 마일스톤용 샘플 데이터 생성 시작...")
    print(f"📊 데이터 규모: {args.scale.upper()}")
    print("="*60)
    
    generator = C1SampleDataGenerator(scale=args.scale)
    
    # 각 데이터베이스 초기화
    postgres_success = await generator.init_postgres_c1_data()
    print()
    
    mysql_success = await generator.init_mysql_c1_data()
    print()
    
    elasticsearch_success = await generator.init_elasticsearch_c1_data()
    print()
    
    print("="*60)
    print("📊 C1 데이터 생성 결과:")
    print(f"  PostgreSQL: {'✅ 성공' if postgres_success else '❌ 실패'}")
    print(f"  MySQL:      {'✅ 성공' if mysql_success else '❌ 실패'}")
    print(f"  Elasticsearch: {'✅ 성공' if elasticsearch_success else '❌ 실패'}")
    
    if postgres_success and mysql_success and elasticsearch_success:
        generator.print_data_summary()
        sys.exit(0)
    else:
        print("\n❌ 일부 데이터베이스 초기화 실패!")
        print("Docker 서비스들이 실행 중인지 확인해주세요.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())