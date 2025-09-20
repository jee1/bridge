from __future__ import annotations

from unittest import mock

import pytest

import cli


@mock.patch("cli.requests.post")
@mock.patch("cli.requests.get")
def test_cli_success_flow(mock_get, mock_post, capsys):
    mock_post.return_value = mock.Mock(status_code=200)
    mock_post.return_value.json.return_value = {
        "steps": [
            {"name": "queue_execution", "details": {"job_id": "abc"}},
        ]
    }
    mock_post.return_value.raise_for_status.return_value = None

    mock_get.return_value = mock.Mock(status_code=200)
    mock_get.return_value.json.return_value = {
        "job_id": "abc",
        "state": "SUCCESS",
        "ready": True,
        "successful": True,
        "result": {"foo": "bar"},
        "error": None,
    }

    with mock.patch("time.sleep") as sleep_mock:
        with mock.patch("sys.exit") as exit_mock:
            sleep_mock.side_effect = RuntimeError("stop")
            exit_mock.side_effect = RuntimeError("stop")
            with pytest.raises(RuntimeError, match="stop"):
                cli.main()

    assert mock_post.called
    assert mock_get.called


@mock.patch("cli.requests.post")
def test_cli_submit_failure(mock_post, capsys):
    mock_post.side_effect = RuntimeError("boom")

    with pytest.raises(SystemExit) as excinfo:
        cli.main()

    assert excinfo.value.code == 1


@mock.patch("cli.requests.post")
@mock.patch("cli.requests.get")
def test_cli_status_failure(mock_get, mock_post):
    mock_post.return_value = mock.Mock(status_code=200)
    mock_post.return_value.json.return_value = {
        "steps": [
            {"name": "queue_execution", "details": {"job_id": "abc"}},
        ]
    }
    mock_post.return_value.raise_for_status.return_value = None

    mock_get.return_value = mock.Mock(status_code=404)
    mock_get.return_value.json.return_value = {}

    with pytest.raises(SystemExit) as excinfo:
        with mock.patch("time.sleep", return_value=None):
            cli.main()

    assert excinfo.value.code == 2
