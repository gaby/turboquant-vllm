"""CPU unit tests for REAP saliency accumulation (expert_pruning).

Regression: the expert hook used a single aggregate norm over the whole
routed batch — ``(Σ_t gate_t) · ‖out_all‖ / N`` — instead of the REAP
formula ``S_j = mean over active tokens of gate_t · ‖out_t‖``, which
couples all tokens and skews the pruning ranking.
"""

import pytest

torch = pytest.importorskip("torch")

from turboquant_vllm.expert_pruning import (  # noqa: E402
    _make_expert_hook,
    _make_gate_hook,
    _SaliencyCollector,
)


class TestSaliencyAccumulation:
    def test_per_token_gate_norm_pairing(self):
        collector = _SaliencyCollector(num_experts=4, device=torch.device("cpu"))
        # 3 tokens, top_k=2. Expert 0 is picked by tokens 0 and 2.
        gate_values = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.4, 0.6]])
        top_k_indices = torch.tensor([[0, 1], [1, 2], [3, 0]])
        collector.record_gate(gate_values, top_k_indices)

        # Expert 0's output for its two routed tokens, norms 2.0 and 3.0
        token_norms = torch.tensor([2.0, 3.0])
        collector.record_expert_activation(0, token_norms, num_tokens=2)

        # S_0 = (0.7·2.0 + 0.6·3.0) / 2 active tokens
        expected = (0.7 * 2.0 + 0.6 * 3.0) / 2
        saliency = collector.compute_saliency()
        assert saliency[0].item() == pytest.approx(expected, rel=1e-5)
        assert collector.active_count[0].item() == 2

    def test_mismatched_token_count_falls_back_to_aggregate(self):
        collector = _SaliencyCollector(num_experts=2, device=torch.device("cpu"))
        gate_values = torch.tensor([[1.0], [0.5]])
        top_k_indices = torch.tensor([[0], [0]])
        collector.record_gate(gate_values, top_k_indices)

        # 3 norms for 2 active tokens (capacity-style dispatch mismatch):
        # falls back to gate_sum * ‖all‖ / num_tokens
        token_norms = torch.tensor([1.0, 2.0, 2.0])
        collector.record_expert_activation(0, token_norms, num_tokens=3)
        expected = (1.0 + 0.5) * token_norms.norm().item() / 3
        assert collector.weighted_sum[0].item() == pytest.approx(expected, rel=1e-5)


class TestHooks:
    def test_expert_hook_records_per_token_norms(self):
        recorded = {}

        class _Collector:
            def record_expert_activation(self, idx, token_norms, num_tokens):
                recorded["idx"] = idx
                recorded["norms"] = token_norms
                recorded["num_tokens"] = num_tokens

        hook = _make_expert_hook(_Collector(), expert_idx=3)
        out = torch.tensor([[3.0, 4.0], [0.0, 5.0]])  # norms 5.0, 5.0
        hook(None, None, out)
        assert recorded["idx"] == 3
        torch.testing.assert_close(recorded["norms"], torch.tensor([5.0, 5.0]))
        assert recorded["num_tokens"] == 2

    def test_gate_hook_clamps_top_k_to_num_experts(self):
        # num_experts (4) < top_k (8): reshape must use the clamped k
        collector = _SaliencyCollector(num_experts=4, device=torch.device("cpu"))
        hook = _make_gate_hook(collector, top_k=8)
        logits = torch.randn(2, 3, 4)  # (batch, seq, num_experts)
        hook(None, None, logits)
        assert collector._current_gate_values.shape == (6, 4)
        assert collector._current_top_k_indices.shape == (6, 4)
