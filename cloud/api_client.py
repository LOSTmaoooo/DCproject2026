import os
from pathlib import Path
from typing import Any

import requests

BASE_URL = os.environ.get('API_BASE_URL', 'http://localhost:8000/api/v1').rstrip('/')
TIMEOUT = 300


def _url(path: str) -> str:
    return f"{BASE_URL}/{path.lstrip('/')}"


def _envelope(code: int, msg: str, data: Any = None) -> dict[str, Any]:
    return {'code': code, 'msg': msg, 'data': data}


def _post_files(path: str, files: list[tuple[str, tuple[str, bytes, str]]], data: dict[str, str] | None = None) -> dict[str, Any]:
    try:
        response = requests.post(_url(path), files=files, data=data or {}, timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        return _envelope(500, '服务端返回格式错误', None)
    except requests.RequestException as exc:
        return _envelope(503, str(exc), None)
    except ValueError:
        return _envelope(500, '服务端返回非 JSON 数据', None)


def _get_json(path: str) -> dict[str, Any]:
    try:
        response = requests.get(_url(path), timeout=TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        return _envelope(500, '服务端返回格式错误', None)
    except requests.RequestException as exc:
        return _envelope(503, str(exc), None)
    except ValueError:
        return _envelope(500, '服务端返回非 JSON 数据', None)


def fetch_model_list() -> list[dict[str, str]]:
    payload = _get_json('/models/list')
    if payload.get('code') != 200:
        return []
    data = payload.get('data') or []
    options: list[dict[str, str]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                options.append({'label': item, 'value': item})
            elif isinstance(item, dict):
                value = item.get('value') or item.get('modelName') or item.get('name')
                label = item.get('label') or value
                if isinstance(value, str):
                    options.append({'label': str(label or value), 'value': value})
    return options


def register_part(part_name: str, image_paths: list[Path]) -> dict[str, Any]:
    files: list[tuple[str, tuple[str, bytes, str]]] = []
    for path in image_paths:
        files.append(('images', (path.name, path.read_bytes(), 'image/jpeg')))
    return _post_files('/mode1/register', files=files, data={'part_name': part_name})


def predict_single(image_path: Path, model_name: str | None = None) -> dict[str, Any]:
    files = [('image', (image_path.name, image_path.read_bytes(), 'image/jpeg'))]
    data = {'model_name': model_name or ''}
    return _post_files('/mode2/predict', files=files, data=data)


def run_batch_analysis(zip_path: Path) -> dict[str, Any]:
    files = [('file', (zip_path.name, zip_path.read_bytes(), 'application/zip'))]
    return _post_files('/mode3/batch_analysis', files=files)
