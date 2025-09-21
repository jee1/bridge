-- MySQL 초기화 스크립트
-- Bridge 개발 환경용 샘플 데이터 생성

-- bridge_dev 데이터베이스 사용
USE bridge_dev;

-- 사용자 테이블
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    age INT,
    city VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- 주문 테이블
CREATE TABLE orders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    product_name VARCHAR(100) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 제품 테이블
CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    category VARCHAR(50),
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 고객 세그먼트 테이블
CREATE TABLE customer_segments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    segment_name VARCHAR(50) NOT NULL,
    score DECIMAL(5,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 샘플 데이터 삽입
INSERT INTO users (name, email, age, city) VALUES
('김철수', 'kim@example.com', 28, '서울'),
('이영희', 'lee@example.com', 35, '부산'),
('박민수', 'park@example.com', 42, '대구'),
('최지영', 'choi@example.com', 29, '서울'),
('정수현', 'jung@example.com', 31, '인천'),
('한소영', 'han@example.com', 26, '광주'),
('윤태호', 'yoon@example.com', 38, '대전'),
('강미영', 'kang@example.com', 33, '울산');

INSERT INTO products (name, category, price, stock_quantity) VALUES
('노트북', '전자제품', 1200000.00, 50),
('마우스', '전자제품', 25000.00, 200),
('키보드', '전자제품', 80000.00, 150),
('모니터', '전자제품', 300000.00, 75),
('책상', '가구', 150000.00, 30),
('의자', '가구', 200000.00, 25),
('스마트폰', '전자제품', 800000.00, 100),
('태블릿', '전자제품', 500000.00, 60);

INSERT INTO orders (user_id, product_name, amount, status, order_date) VALUES
(1, '노트북', 1200000.00, 'completed', '2024-01-15 10:30:00'),
(1, '마우스', 25000.00, 'completed', '2024-01-15 10:35:00'),
(2, '키보드', 80000.00, 'completed', '2024-01-16 14:20:00'),
(3, '모니터', 300000.00, 'pending', '2024-01-17 09:15:00'),
(4, '책상', 150000.00, 'completed', '2024-01-18 16:45:00'),
(5, '의자', 200000.00, 'shipped', '2024-01-19 11:30:00'),
(6, '스마트폰', 800000.00, 'completed', '2024-01-20 13:20:00'),
(7, '태블릿', 500000.00, 'completed', '2024-01-21 15:10:00'),
(8, '노트북', 1200000.00, 'pending', '2024-01-22 10:00:00'),
(1, '키보드', 80000.00, 'completed', '2024-01-23 14:30:00');

INSERT INTO customer_segments (user_id, segment_name, score) VALUES
(1, 'VIP', 95.5),
(2, 'Premium', 85.2),
(3, 'Standard', 65.8),
(4, 'Premium', 88.1),
(5, 'Standard', 72.3),
(6, 'VIP', 92.7),
(7, 'Premium', 86.4),
(8, 'Standard', 68.9);

-- 인덱스 생성
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_order_date ON orders(order_date);
CREATE INDEX idx_customer_segments_user_id ON customer_segments(user_id);
CREATE INDEX idx_customer_segments_segment ON customer_segments(segment_name);

-- 통계 뷰 생성
CREATE VIEW user_order_summary AS
SELECT 
    u.id,
    u.name,
    u.email,
    u.city,
    COUNT(o.id) as total_orders,
    COALESCE(SUM(o.amount), 0) as total_spent,
    MAX(o.order_date) as last_order_date
FROM users u
LEFT JOIN orders o ON u.id = o.user_id
GROUP BY u.id, u.name, u.email, u.city;
