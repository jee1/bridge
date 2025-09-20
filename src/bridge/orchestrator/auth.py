"""기본 인증 및 인가 모듈."""
from __future__ import annotations

import os
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# API 키 인증을 위한 보안 스키마
security = HTTPBearer()


def get_api_key() -> str:
    """환경 변수에서 API 키를 가져온다."""
    api_key = os.getenv("BRIDGE_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key not configured"
        )
    return api_key


def verify_api_key(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> str:
    """API 키를 검증한다."""
    api_key = get_api_key()
    
    if credentials.credentials != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return credentials.credentials


# 의존성 주입을 위한 타입 별칭
AuthenticatedUser = Annotated[str, Depends(verify_api_key)]
