#!/usr/bin/env python3
"""Bridge MCP 서버 실행 스크립트."""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.bridge.mcp_server import main

if __name__ == "__main__":
    asyncio.run(main())

