#!/usr/bin/env python3
"""수동 테스트 스크립트 - 의존성 없이 기본 동작 확인"""

import sys
import os
sys.path.insert(0, 'src')

def test_imports():
    """모듈 임포트 테스트"""
    print("=== 모듈 임포트 테스트 ===")
    
    try:
        import bridge
        print("✅ bridge 모듈 임포트 성공")
    except Exception as e:
        print(f"❌ bridge 모듈 임포트 실패: {e}")
        return False
    
    try:
        from bridge.connectors.base import BaseConnector
        print("✅ BaseConnector 임포트 성공")
    except Exception as e:
        print(f"❌ BaseConnector 임포트 실패: {e}")
        return False
    
    try:
        from bridge.connectors.mock import MockConnector
        print("✅ MockConnector 임포트 성공")
    except Exception as e:
        print(f"❌ MockConnector 임포트 실패: {e}")
        return False
    
    try:
        from bridge.connectors.postgres import PostgresConnector
        print("✅ PostgresConnector 임포트 성공")
    except Exception as e:
        print(f"❌ PostgresConnector 임포트 실패: {e}")
        return False
    
    # Elasticsearch는 의존성이 없어서 실패할 것으로 예상
    try:
        from bridge.connectors.elasticsearch import ElasticsearchConnector
        print("✅ ElasticsearchConnector 임포트 성공")
    except Exception as e:
        print(f"⚠️ ElasticsearchConnector 임포트 실패 (의존성 없음): {e}")
    
    return True

def test_mock_connector():
    """MockConnector 동작 테스트"""
    print("\n=== MockConnector 동작 테스트 ===")
    
    try:
        from bridge.connectors.mock import MockConnector
        
        # 기본 초기화
        connector = MockConnector()
        print(f"✅ MockConnector 초기화: name={connector.name}")
        
        # 연결 테스트
        result = connector.test_connection()
        print(f"✅ 연결 테스트: {result}")
        
        # 메타데이터 조회
        metadata = connector.get_metadata()
        print(f"✅ 메타데이터 조회: {len(metadata.get('fields', []))}개 필드")
        
        # 쿼리 실행
        results = list(connector.run_query("SELECT * FROM dummy"))
        print(f"✅ 쿼리 실행: {len(results)}개 결과")
        
        return True
    except Exception as e:
        print(f"❌ MockConnector 테스트 실패: {e}")
        return False

def test_postgres_connector():
    """PostgresConnector 초기화 테스트"""
    print("\n=== PostgresConnector 초기화 테스트 ===")
    
    try:
        from bridge.connectors.postgres import PostgresConnector
        
        # 기본 초기화 (의존성 없이도 가능)
        connector = PostgresConnector(
            name="test_postgres",
            settings={
                "host": "localhost",
                "port": 5432,
                "database": "test",
                "user": "test",
                "password": "test",
            }
        )
        print(f"✅ PostgresConnector 초기화: name={connector.name}")
        print(f"✅ 설정: host={connector.settings['host']}, port={connector.settings['port']}")
        
        # 의존성 없이 실행 시도 (ImportError 예상)
        try:
            import asyncio
            asyncio.run(connector.test_connection())
            print("⚠️ 의존성 없이도 실행됨 (예상과 다름)")
        except ImportError as e:
            print(f"✅ 예상된 ImportError: {e}")
        except Exception as e:
            print(f"⚠️ 예상과 다른 에러: {e}")
        
        return True
    except ImportError as e:
        print(f"✅ PostgresConnector 초기화 시 ImportError (예상됨): {e}")
        return True
    except Exception as e:
        print(f"❌ PostgresConnector 테스트 실패: {e}")
        return False

def test_elasticsearch_connector_structure():
    """ElasticsearchConnector 구조 테스트 (의존성 없이)"""
    print("\n=== ElasticsearchConnector 구조 테스트 ===")
    
    try:
        # 파일 직접 읽기로 구조 확인
        with open('src/bridge/connectors/elasticsearch.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 클래스 정의 확인
        if 'class ElasticsearchConnector(BaseConnector):' in content:
            print("✅ ElasticsearchConnector 클래스 정의 존재")
        else:
            print("❌ ElasticsearchConnector 클래스 정의 없음")
            return False
        
        # 주요 메서드 확인
        methods = [
            '__init__',
            '_get_client',
            'test_connection',
            'get_metadata',
            'run_query',
            'close',
            '_validate_port'
        ]
        
        for method in methods:
            if f'def {method}' in content:
                print(f"✅ {method} 메서드 존재")
            else:
                print(f"❌ {method} 메서드 없음")
                return False
        
        # 개선된 설정 검증 로직 확인
        if '필수 설정이 누락되었습니다' in content:
            print("✅ 개선된 설정 검증 로직 존재")
        else:
            print("❌ 개선된 설정 검증 로직 없음")
            return False
        
        return True
    except Exception as e:
        print(f"❌ ElasticsearchConnector 구조 테스트 실패: {e}")
        return False

def test_removed_directories():
    """삭제된 디렉토리 확인"""
    print("\n=== 삭제된 디렉토리 확인 ===")
    
    removed_dirs = [
        'src/bridge/dashboard',
        'src/bridge/ml/insights',
        'src/bridge/ml/pipelines',
        'src/bridge/ml/recommendations'
    ]
    
    for dir_path in removed_dirs:
        if not os.path.exists(dir_path):
            print(f"✅ {dir_path} 삭제됨")
        else:
            print(f"❌ {dir_path} 아직 존재")
            return False
    
    return True

def test_py_typed_files():
    """py.typed 파일 확인"""
    print("\n=== py.typed 파일 확인 ===")
    
    expected_py_typed = [
        'src/bridge/audit/py.typed',
        'src/bridge/automation/py.typed',
        'src/bridge/governance/py.typed',
        'src/bridge/orchestrator/py.typed',
        'src/bridge/semantic/py.typed',
        'src/bridge/workspaces/py.typed'
    ]
    
    for file_path in expected_py_typed:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 존재")
        else:
            print(f"❌ {file_path} 없음")
            return False
    
    return True

def test_connector_registry():
    """커넥터 레지스트리 테스트"""
    print("\n=== 커넥터 레지스트리 테스트 ===")
    
    try:
        from bridge.connectors.registry import connector_registry
        from bridge.connectors.mock import MockConnector
        
        # 등록된 커넥터 확인
        connectors = connector_registry.list()
        print(f"✅ 등록된 커넥터: {connectors}")
        
        # MockConnector 조회
        mock_connector = connector_registry.get("mock")
        if mock_connector and isinstance(mock_connector, MockConnector):
            print("✅ MockConnector 레지스트리에서 조회 성공")
        else:
            print("❌ MockConnector 레지스트리에서 조회 실패")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 커넥터 레지스트리 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🧪 Bridge 프로젝트 수동 동작 테스트")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_mock_connector,
        test_postgres_connector,
        test_elasticsearch_connector_structure,
        test_removed_directories,
        test_py_typed_files,
        test_connector_registry
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} 실패")
        except Exception as e:
            print(f"❌ {test.__name__} 예외 발생: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과!")
        return True
    else:
        print("⚠️ 일부 테스트 실패")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)