from serving.clients import (
    DeterministicClient,
    SimulatedSelfHostedClient,
    _strip_reasoning,
)


def test_strips_complete_think_block():
    assert _strip_reasoning("<think>scratch</think>The answer.") == "The answer."


def test_strips_unclosed_think_block():
    # Reasoning model hit the token cap mid-thought: no closing tag.
    assert _strip_reasoning("<think>still reasoning and out of budget") == ""


def test_keeps_plain_text():
    assert _strip_reasoning("3 updates today.") == "3 updates today."


def test_simulated_self_hosted_reports_cold_then_warm():
    client = SimulatedSelfHostedClient()
    cold = client.complete("sys", "- [x] hi")
    warm = client.complete("sys", "- [x] hi")
    assert cold.cold_start is True and warm.cold_start is False
    assert cold.latency_ms > warm.latency_ms
    assert cold.simulated is True


def test_deterministic_client_renders_updates():
    result = DeterministicClient().complete("sys", "- [household] groceries\n- [pet] fed")
    assert "2 update(s)" in result.text
    assert result.simulated is True
