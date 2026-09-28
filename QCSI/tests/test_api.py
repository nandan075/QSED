"""
Integration tests for QCSI FastAPI service.
"""

import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from qcsi.api import app

client = TestClient(app)


def create_png_bytes(color: int = 150) -> bytes:
    img = np.zeros((150, 150, 3), dtype=np.uint8) + color
    cv2.putText(img, "TEST", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    _, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "version" in data


def test_compare_endpoint():
    bytes_A = create_png_bytes(100)
    bytes_B = create_png_bytes(100)

    response = client.post(
        "/compare",
        files={
            "file_A": ("testA.png", bytes_A, "image/png"),
            "file_B": ("testB.png", bytes_B, "image/png"),
        },
        data={"include_visualization": "true"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "calibrated_similarity_pct" in data
    assert "verdict" in data
    assert "components" in data
    assert "evidence" in data
    assert "visualization_base64" in data
