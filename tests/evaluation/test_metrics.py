from ragfold.evaluation.metrics import (
    answer_exact_match,
    answer_f1,
    average_precision,
    hit_at_k,
    map_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_recall_precision_hit_and_mrr_use_predicted_first_reference_second():
    predicted = ["a", "b", "c"]
    reference = ["b", "d"]

    assert recall_at_k(predicted, reference, k=2) == 0.5
    assert precision_at_k(predicted, reference, k=2) == 0.5
    assert hit_at_k(predicted, reference, k=2) == 1.0
    assert mean_reciprocal_rank(predicted, reference) == 0.5


def test_ndcg_and_average_precision_reward_early_relevant_results():
    predicted = ["b", "a", "d"]
    reference = ["b", "d"]

    assert ndcg_at_k(predicted, reference, k=3) > 0.9
    assert average_precision(predicted, reference, k=3) > 0.8


def test_map_averages_queries():
    predicted = [["a", "b"], ["c", "d"]]
    reference = [["a"], ["d"]]

    assert map_score(predicted, reference, k=2) == (1.0 + 0.5) / 2


def test_answer_exact_match_and_f1_normalize_text():
    assert answer_exact_match("The Eiffel Tower", "eiffel tower") == 1.0
    assert answer_f1("Paris France", "Paris") == 2 / 3
