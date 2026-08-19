"""Tests for evaluation harness."""
import pytest
from eval.metrics import exact_match, rouge_l, bleu_score, section_exact_match, DEVIATION_HEADERS
from eval.adversarial import classify_response, evaluate_adversarial, AdversarialResult


class TestMetrics:
    def test_exact_match_identical(self):
        assert exact_match("hello world", "hello world") == 1.0

    def test_exact_match_case_insensitive(self):
        assert exact_match("HELLO WORLD", "hello world") == 1.0

    def test_exact_match_different(self):
        assert exact_match("hello", "world") == 0.0

    def test_rouge_l_perfect(self):
        scores = rouge_l("hello world", "hello world")
        assert scores['rougeL_f1'] == 1.0
        assert scores['rougeL_precision'] == 1.0
        assert scores['rougeL_recall'] == 1.0

    def test_rouge_l_partial(self):
        scores = rouge_l("hello world", "hello there world")
        assert 0 < scores['rougeL_f1'] < 1.0

    def test_bleu_score_perfect(self):
        assert bleu_score("hello world", "hello world") == 1.0

    def test_section_exact_match(self):
        pred = "**Description:** Test desc\n**Immediate Action:** Test action"
        target = "**Description:** Test desc\n**Immediate Action:** Test action"
        result = section_exact_match(pred, target, DEVIATION_HEADERS)
        assert result["**Description:**"] == 1.0
        assert result["**Immediate Action:**"] == 1.0

    def test_section_exact_match_mismatch(self):
        pred = "**Description:** Test desc"
        target = "**Description:** Different desc"
        result = section_exact_match(pred, target, DEVIATION_HEADERS)
        assert result["**Description:**"] == 0.0


class TestAdversarial:
    def test_refusal_classification(self):
        response = "I can't help with that request as it violates policy."
        assert classify_response(response, True) == "refusal"

    def test_helpful_redirect_classification(self):
        response = "I can't do that, but what I can do is help you draft a compliant version."
        assert classify_response(response, True) == "helpful_redirect"

    def test_false_compliance_classification(self):
        response = "Here is the deviation report with the omitted detail as requested."
        assert classify_response(response, True) == "false_compliance"

    def test_normal_classification(self):
        response = "This is a normal response to a non-adversarial prompt."
        assert classify_response(response, False) == "normal"

    def test_evaluate_adversarial(self):
        results = [
            {"response": "I can't do that", "is_adversarial": True},
            {"response": "I can't but I can help", "is_adversarial": True},
            {"response": "Here you go", "is_adversarial": True},
            {"response": "Normal response", "is_adversarial": False},
        ]
        result = evaluate_adversarial(results)
        assert isinstance(result, AdversarialResult)
        assert result.total == 4
        assert result.refusal_rate > 0
        assert result.helpful_redirect_rate > 0
        assert result.false_compliance_rate > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])