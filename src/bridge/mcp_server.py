#!/usr/bin/env python3
"""통합된 Bridge MCP 서버 - 환경 변수 기반 모드 지원"""

# 통합된 MCP 서버를 사용
from .mcp_server_unified import UnifiedBridgeMCPServer, run

# 하위 호환성을 위한 별칭
BridgeMCPServer = UnifiedBridgeMCPServer

if __name__ == "__main__":
    run()
