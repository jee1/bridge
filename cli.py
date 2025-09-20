from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict

import requests

DEFAULT_BASE_URL = "http://localhost:8000"


def submit_task(base_url: str, payload: Dict[str, Any]) -> str:
    response = requests.post(f"{base_url}/tasks/plan", json=payload, timeout=10)
    response.raise_for_status()
    body = response.json()
    job_step = body["steps"][-1]
    return job_step["details"]["job_id"]


def fetch_status(base_url: str, job_id: str) -> Dict[str, Any]:
    response = requests.get(f"{base_url}/tasks/{job_id}", timeout=10)
    if response.status_code == 404:
        raise RuntimeError("작업을 찾을 수 없습니다. job_id를 확인하세요.")
    return response.json(), response.status_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge MCP Task CLI")
    parser.add_argument("intent", help="사용자 의도 설명")
    parser.add_argument("--sources", nargs="*", default=["mock"], help="사용할 데이터 소스 목록")
    parser.add_argument("--tools", nargs="*", default=["sql_executor"], help="필요한 도구 목록")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="오케스트레이터 API 기본 URL")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="상태 조회 간격(초)")
    args = parser.parse_args()

    payload = {
        "intent": args.intent,
        "sources": args.sources,
        "required_tools": args.tools,
        "context": {},
    }

    try:
        job_id = submit_task(args.base_url, payload)
    except Exception as exc:
        print(f"[ERROR] 작업 제출 실패: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"작업이 제출되었습니다. job_id={job_id}")

    while True:
        try:
            status, code = fetch_status(args.base_url, job_id)
        except Exception as exc:
            print(f"[ERROR] 상태 조회 실패: {exc}", file=sys.stderr)
            sys.exit(2)

        print(f"[STATUS {code}] {json.dumps(status, ensure_ascii=False)}")

        if status.get("ready"):
            if status.get("successful"):
                print("[SUCCESS] 작업이 완료되었습니다.")
                sys.exit(0)
            if status.get("state") == "FAILURE":
                print(f"[FAILED] {status.get('error')}", file=sys.stderr)
                sys.exit(3)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
