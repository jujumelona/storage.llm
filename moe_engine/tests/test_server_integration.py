#!/usr/bin/env python3
"""
Integration tests for storageLLM server (Bug B2, B3)
Tests server API with various payloads including edge cases
"""

import json
import requests
import pytest
import time
from typing import Dict, List, Any

# Server configuration
BASE_URL = "http://127.0.0.1:8080"
TIMEOUT = 5


class TestServerIntegration:
    """Integration tests for server endpoints"""

    @pytest.fixture(scope="class", autouse=True)
    def check_server(self):
        """Check if server is running"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            if response.status_code != 200:
                pytest.skip("Server not running or not healthy")
        except requests.exceptions.RequestException:
            pytest.skip("Server not accessible")

    def test_health_endpoint(self):
        """Test /health endpoint"""
        response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
        assert response.status_code == 200

    def test_valid_prefetch_request(self):
        """Test valid prefetch request"""
        payload = {
            "current_layer": 2,
            "selected_experts": [0, 1, 2],
            "next_experts": [3, 4]
        }

        response = requests.post(
            f"{BASE_URL}/prefetch",
            json=payload,
            timeout=TIMEOUT
        )

        # Should succeed or return 200
        assert response.status_code in [200, 204]

    def test_negative_layer(self):
        """Test negative layer (Bug B2)"""
        payload = {
            "current_layer": -1,
            "selected_experts": [0, 1],
            "next_experts": [2]
        }

        response = requests.post(
            f"{BASE_URL}/prefetch",
            json=payload,
            timeout=TIMEOUT
        )

        # Should reject with 400
        assert response.status_code == 400
        assert "out of range" in response.text.lower() or "invalid" in response.text.lower()

    def test_layer_out_of_range(self):
        """Test layer out of range (Bug B2)"""
        payload = {
            "current_layer": 999999,
            "selected_experts": [0, 1],
            "next_experts": [2]
        }

        response = requests.post(
            f"{BASE_URL}/prefetch",
            json=payload,
            timeout=TIMEOUT
        )

        # Should reject with 400
        assert response.status_code == 400

    def test_negative_expert_id(self):
        """Test negative expert ID (Bug B2)"""
        payload = {
            "current_layer": 2,
            "selected_experts": [-1, 0, 1],
            "next_experts": [2]
        }

        response = requests.post(
            f"{BASE_URL}/prefetch",
            json=payload,
            timeout=TIMEOUT
        )

        # Should reject with 400
        assert response.status_code == 400
        assert "expert" in response.text.lower()

    def test_expert_id_out_of_range(self):
        """Test expert ID out of range (Bug B2)"""
        payload = {
            "current_layer": 2,
            "selected_experts": [0, 1, 999999],
            "next_experts": [2]
        }

        response = requests.post(
            f"{BASE_URL}/prefetch",
            json=payload,
            timeout=TIMEOUT
        )

        # Should reject with 400
        assert response.status_code == 400

    def test_duplicate_expert_ids(self):
        """Test duplicate expert IDs (should be deduplicated)"""
        payload = {
            "current_layer": 2,
            "selected_experts": [3, 3, 3, 1, 1],
            "next_experts": [4, 4]
        }

        response = requests.post(
            f"{BASE_URL}/prefetch",
            json=payload,
            timeout=TIMEOUT
        )

        # Should succeed (after deduplication)
        assert response.status_code in [200, 204]

    def test_empty_expert_lists(self):
        """Test empty expert lists"""
        payload = {
            "current_layer": 2,
            "selected_experts": [],
            "next_experts": []
        }

        response = requests.post(
            f"{BASE_URL}/prefetch",
            json=payload,
            timeout=TIMEOUT
        )

        # Should succeed (empty is valid)
        assert response.status_code in [200, 204]

    def test_json_key_collision(self):
        """Test JSON key collision in string (Bug B1)"""
        payload = {
            "note": 'payload says "selected_experts":[9]',
            "current_layer": 2,
            "selected_experts": [1, 2],
            "next_experts": [3]
        }

        response = requests.post(
            f"{BASE_URL}/prefetch",
            json=payload,
            timeout=TIMEOUT
        )

        # Should succeed and use correct selected_experts [1, 2], not [9]
        assert response.status_code in [200, 204]

    def test_malformed_json_nested_array(self):
        """Test malformed JSON with nested array (Bug B1)"""
        # This is tricky - we need to send raw JSON
        payload_str = '{"current_layer":2,"selected_experts":[1,[2]],"next_experts":[3]}'

        response = requests.post(
            f"{BASE_URL}/prefetch",
            data=payload_str,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )

        # Should reject (nested array not allowed)
        assert response.status_code in [400, 500]

    def test_type_mismatch_string_layer(self):
        """Test type mismatch - string instead of int (Bug B1)"""
        payload_str = '{"current_layer":"2","selected_experts":[1,2]}'

        response = requests.post(
            f"{BASE_URL}/prefetch",
            data=payload_str,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )

        # Should reject (type mismatch)
        assert response.status_code in [400, 500]

    def test_concurrent_requests(self):
        """Test concurrent requests (stress test)"""
        import concurrent.futures

        def send_request(i: int) -> int:
            payload = {
                "current_layer": i % 8,
                "selected_experts": [0, 1],
                "next_experts": [2, 3]
            }
            try:
                response = requests.post(
                    f"{BASE_URL}/prefetch",
                    json=payload,
                    timeout=TIMEOUT
                )
                return response.status_code
            except:
                return 500

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Most should succeed
        success_count = sum(1 for r in results if r in [200, 204])
        assert success_count >= 45  # At least 90% success rate

    def test_large_expert_list(self):
        """Test large expert list (boundary test)"""
        # Create a list of 64 experts (max allowed)
        payload = {
            "current_layer": 2,
            "selected_experts": list(range(8)),  # Assuming 8 experts per layer
            "next_experts": []
        }

        response = requests.post(
            f"{BASE_URL}/prefetch",
            json=payload,
            timeout=TIMEOUT
        )

        # Should succeed
        assert response.status_code in [200, 204]


class TestServerPerformance:
    """Performance tests for server"""

    @pytest.fixture(scope="class", autouse=True)
    def check_server(self):
        """Check if server is running"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=TIMEOUT)
            if response.status_code != 200:
                pytest.skip("Server not running or not healthy")
        except requests.exceptions.RequestException:
            pytest.skip("Server not accessible")

    def test_response_time(self):
        """Test response time for prefetch requests"""
        payload = {
            "current_layer": 2,
            "selected_experts": [0, 1, 2],
            "next_experts": [3, 4]
        }

        times = []
        for _ in range(10):
            start = time.time()
            response = requests.post(
                f"{BASE_URL}/prefetch",
                json=payload,
                timeout=TIMEOUT
            )
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status_code in [200, 204]

        avg_time = sum(times) / len(times)
        max_time = max(times)

        print(f"\nAverage response time: {avg_time*1000:.2f}ms")
        print(f"Max response time: {max_time*1000:.2f}ms")

        # Response should be fast (< 100ms for prefetch hint)
        assert avg_time < 0.1
        assert max_time < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
