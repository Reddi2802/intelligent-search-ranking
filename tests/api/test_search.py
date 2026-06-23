"""
API endpoint tests.
Tests edge cases, validation, and expected behavior.
Requires the API to be running on localhost:8000.
"""

import pytest
import httpx

BASE_URL = "http://localhost:8000/api/v1"


@pytest.fixture(scope="module")
def client():
    return httpx.Client(base_url=BASE_URL, timeout=30.0)


# ─── Health ───────────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["bm25_index"] == "loaded"
    assert data["faiss_index"] == "loaded"
    assert data["embedding_model"] == "loaded"
    assert data["ml_ranker"] == "loaded"
    assert data["cross_encoder"] == "loaded"


# ─── Search — valid inputs ─────────────────────────────────────────────────────

def test_search_returns_results(client):
    r = client.post("/search", json={"query": "why did stalin want eastern europe"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 10  # default top_k
    assert data["total_results"] == 100


def test_search_results_are_sorted_by_rank(client):
    r = client.post("/search", json={"query": "manhattan project impact"})
    assert r.status_code == 200
    ranks = [result["rank"] for result in r.json()["results"]]
    assert ranks == sorted(ranks)


def test_search_result_has_required_fields(client):
    r = client.post("/search", json={"query": "what is quantum mechanics"})
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert "passage_id" in result
    assert "text" in result
    assert "rank" in result
    assert "scores" in result


def test_search_top_k_respected(client):
    r = client.post("/search", json={"query": "climate change", "top_k": 5})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 5


def test_search_all_modes(client):
    for mode in ["bm25", "semantic", "hybrid", "full"]:
        r = client.post("/search", json={"query": "black holes", "mode": mode})
        assert r.status_code == 200, f"Mode {mode} failed"
        assert len(r.json()["results"]) > 0


def test_search_scores_present_in_full_mode(client):
    r = client.post("/search", json={"query": "einstein theory", "mode": "full"})
    assert r.status_code == 200
    scores = r.json()["results"][0]["scores"]
    assert scores["bm25_score"] >= 0
    assert scores["semantic_score"] >= 0
    assert scores["cross_encoder_score"] != 0


def test_search_latency_present(client):
    r = client.post("/search", json={"query": "world war two"})
    assert r.status_code == 200
    assert r.json()["latency_ms"] > 0


# ─── Search — edge cases ───────────────────────────────────────────────────────

def test_empty_query_rejected(client):
    r = client.post("/search", json={"query": ""})
    assert r.status_code == 422


def test_whitespace_only_query_rejected(client):
    r = client.post("/search", json={"query": "     "})
    assert r.status_code == 422


def test_top_k_zero_rejected(client):
    r = client.post("/search", json={"query": "test", "top_k": 0})
    assert r.status_code == 422


def test_top_k_above_maximum_rejected(client):
    r = client.post("/search", json={"query": "test", "top_k": 101})
    assert r.status_code == 422


def test_invalid_mode_rejected(client):
    r = client.post("/search", json={"query": "test", "mode": "invalid"})
    assert r.status_code == 422


def test_missing_query_rejected(client):
    r = client.post("/search", json={"top_k": 5})
    assert r.status_code == 422


def test_very_long_query_rejected(client):
    long_query = "word " * 200  # 1000 characters
    r = client.post("/search", json={"query": long_query})
    assert r.status_code == 422


def test_special_characters_handled(client):
    r = client.post("/search", json={"query": "!@#$%^&*()"})
    assert r.status_code == 200


def test_single_character_query(client):
    r = client.post("/search", json={"query": "a"})
    assert r.status_code == 200


def test_query_with_numbers(client):
    r = client.post("/search", json={"query": "world war 2 1939"})
    assert r.status_code == 200
    assert len(r.json()["results"]) > 0


# ─── Analytics ────────────────────────────────────────────────────────────────

def test_analytics_returns_valid_response(client):
    r = client.get("/analytics")
    assert r.status_code == 200
    data = r.json()
    assert "total_queries" in data
    assert "avg_latency_ms" in data
    assert "p95_latency_ms" in data
    assert "mode_distribution" in data


def test_analytics_tracks_queries(client):
    # run a search first
    client.post("/search", json={"query": "test analytics tracking"})
    r = client.get("/analytics")
    assert r.json()["total_queries"] > 0