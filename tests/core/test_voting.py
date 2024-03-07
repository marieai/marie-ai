import pytest

from marie.pipe.model import ClassificationResult
from marie.pipe.voting import get_voting_strategy


def test_model_conversion():
    raw_obj = {'page': '0', 'classification': 'misc', 'score': 0.9958, 'sub_classifier': [
        {'classifier': 'corr_longformer_X_level2_misc',
         'details': [{'page': '0', 'classification': 'misc_credentialing', 'score': 0.6762}]}],
               'classifier': 'corr_longformer_classifier_X'}

    result = ClassificationResult(**raw_obj)
    print(result)

    # serialize back to dict
    assert result.dict() == raw_obj


def test_voting_majority():
    voter = get_voting_strategy("majority")

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.5, page=0, classifier="A"),
            ClassificationResult(classification="A", score=0.5, page=0, classifier="B"),
            ClassificationResult(classification="B", score=0.5, page=0, classifier="C"),
        ]
    )

    assert result.classification == "A"
    assert len(result.classifier.split(",")) == 2
    assert result.score == 0.5


def test_voting_majority_tie_break_abstain():
    voter = get_voting_strategy("majority", "abstain")

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.5, page=0, classifier="A"),
            ClassificationResult(classification="B", score=0.5, page=0, classifier="C"),
        ]
    )
    assert result.classification is None
    assert result.score == 0
    assert result.classifier is None


def test_voting_majority_tie_break_best():
    voter = get_voting_strategy("majority", "best")

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.7, page=0, classifier="A"),
            ClassificationResult(classification="B", score=0.8, page=0, classifier="B"),
            ClassificationResult(classification="C", score=0.6, page=0, classifier="B"),
        ]
    )
    assert result.classification == "B"
    assert result.score == 0.8
    assert result.classifier == "B"


def test_voting_majority_tie_break_best_with_diff_valid_custom_diff():
    voter = get_voting_strategy("majority", "best_with_diff", silent=True, max_diff=0.25)

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.7, page=0, classifier="A"),
            ClassificationResult(classification="B", score=0.8, page=0, classifier="B"),
            ClassificationResult(classification="C", score=0.6, page=0, classifier="B"),
        ]
    )
    assert result.classification == "B"
    assert result.score == 0.8
    assert result.classifier == "B"


def test_voting_majority_tie_break_best_with_diff():
    voter = get_voting_strategy("majority", "best_with_diff", silent=True)

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.75, page=0, classifier="A"),
            ClassificationResult(classification="B", score=0.8, page=0, classifier="B"),
            ClassificationResult(classification="C", score=0.6, page=0, classifier="B"),
        ]
    )
    assert result.classification == "B"
    assert result.score == 0.8
    assert result.classifier == "B"


def test_voting_majority_tie_break_best_with_diff_invalid():
    voter = get_voting_strategy("majority", "best_with_diff")

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.2, page=0, classifier="A"),
            ClassificationResult(classification="B", score=0.8, page=0, classifier="B"),
            ClassificationResult(classification="C", score=0.1, page=0, classifier="B"),
        ]
    )
    assert result.classification is None
    assert result.score == 0
    assert result.classifier is None


def test_voting_max_score():
    voter = get_voting_strategy("max_score")

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=1.0, page=0, classifier="A"),
            ClassificationResult(classification="A", score=0.5, page=0, classifier="B"),
            ClassificationResult(classification="B", score=0.5, page=0, classifier="C"),
        ]
    )

    assert result.classification == "A"
    assert len(result.classifier.split(",")) == 1
    assert result.score == 1.0


def test_voting_max_score_tie_break_abstain():
    voter = get_voting_strategy("max_score", "abstain")

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.5, page=0, classifier="A"),
            ClassificationResult(classification="A", score=0.5, page=0, classifier="B"),
            ClassificationResult(classification="B", score=0.5, page=0, classifier="C"),
        ]
    )

    assert result.classification is None
    assert result.classifier is None


def test_voting_max_score_tie_break_best():
    voter = get_voting_strategy("max_score", "best")

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.8, page=0, classifier="A"),
            ClassificationResult(classification="B", score=0.8, page=0, classifier="B"),
            ClassificationResult(classification="C", score=0.8, page=0, classifier="B"),
        ]
    )
    assert result.classification == "A"
    assert result.score == 0.8
    assert result.classifier == "A"


def test_voting_max_score_tie_break_best_with_diff():
    voter = get_voting_strategy("max_score", "best_with_diff")

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.6, page=0, classifier="A"),
            ClassificationResult(classification="B", score=0.8, page=0, classifier="B"),
            ClassificationResult(classification="C", score=0.5, page=0, classifier="B"),
        ]
    )
    assert result.classification is None
    assert result.classifier is None


def test_voting_max_score_tie_break_best_with_diff_custom_max_diff():
    voter = get_voting_strategy("max_score", "best_with_diff", silent=True, max_diff=0.25)

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=0.6, page=0, classifier="A"),
            ClassificationResult(classification="B", score=0.8, page=0, classifier="B"),
            ClassificationResult(classification="C", score=0.5, page=0, classifier="B"),
        ]
    )
    assert result.classification == "B"
    assert result.score == 0.8
    assert result.classifier == "B"


def test_voting_max_score_with_weights():
    voter = get_voting_strategy("max_score", weights=[1, 2, 3])

    result = voter.vote(
        [
            ClassificationResult(classification="A", score=1, page=0, classifier="A"),
            ClassificationResult(classification="B", score=1, page=0, classifier="B"),
            ClassificationResult(classification="C", score=1, page=0, classifier="B"),
        ]
    )
    assert result.classification == "C"
    assert result.score == 1.0
    assert result.classifier == "B"
