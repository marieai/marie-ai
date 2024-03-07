from abc import abstractmethod
from enum import Enum
from typing import List, Union

from marie.pipe.model import ClassificationResult


class TieBreakPolicy(Enum):
    """A tie break policy"""

    ABSTAIN = "abstain"
    BEST = "best"
    BEST_WITH_DIFF = "best_with_diff"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class VotingPolicy(Enum):
    """Policy indicating what voting strategy should be used.

    MAJORITY: The voting strategy should be based on the majority of votes.
    MAX_SCORE: The voting strategy should be based on the maximum score.
    """

    MAJORITY = "majority"
    MAX_SCORE = "max_score"

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class VotingStrategy:
    """Abstract base class for voting strategies."""

    @abstractmethod
    def vote(self, results: List[ClassificationResult]) -> ClassificationResult:
        """
        Abstract method to be implemented by subclasses. This method should take a list of results and return a single result based on the voting strategy.

        :param results: A list of ClassificationResult instances.
        :return: The result of the vote.
        """
        ...

    def __call__(self, *args, **kwargs) -> ClassificationResult:
        """
        Makes the class callable. Delegates to the vote method.

        :param args: Positional arguments to be passed to the vote method.
        :param kwargs: Keyword arguments to be passed to the vote method.
        :return: The result of the vote.
        """
        return self.vote(*args, **kwargs)


class MajorityVoter(VotingStrategy):
    def __init__(
        self,
        tie_break_policy: Union[TieBreakPolicy, str] = "abstain",
        silent=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.silent = silent
        if isinstance(tie_break_policy, str):
            tie_break_policy = TieBreakPolicy(tie_break_policy)
        self.tie_break_policy = tie_break_policy
        # max_diff is only used when tie_break_policy is BEST_WITH_DIFF
        self.max_diff = kwargs.get("max_diff", 0.1)

    def vote(self, results: List[ClassificationResult]) -> ClassificationResult:
        if not results:
            if not self.silent:
                raise ValueError("No results to vote on")
            else:
                return ClassificationResult(
                    classification=None, score=0, page=0, classifier=None
                )

        if len(results) == 1:
            return results[0]

        # score bases on the number of votes for each class
        scores = {}
        for result in results:
            scores[result.classification] = scores.get(result.classification, 0) + 1

        max_score = max(scores.values())
        max_score_label = next(
            (label for label, score in scores.items() if score == max_score), None
        )

        candidates = [
            result for result in results if result.classification == max_score_label
        ]
        total_score = sum([result.score for result in candidates])

        # check for tie
        if list(scores.values()).count(max_score) > 1:
            if self.tie_break_policy == TieBreakPolicy.ABSTAIN:
                return ClassificationResult(
                    classification=None, score=0, page=results[0].page, classifier=None
                )
            elif self.tie_break_policy == TieBreakPolicy.BEST:
                best_score = 0
                best = None
                for result in results:
                    if (
                        scores[result.classification] == max_score
                        and result.score >= best_score
                    ):
                        best_score = result.score
                        best = result
                return best
            elif self.tie_break_policy == TieBreakPolicy.BEST_WITH_DIFF:
                best = [
                    result
                    for result in results
                    if scores[result.classification] == max_score
                ]
                topK = sorted(best, key=lambda x: x.score, reverse=True)[:2]
                diff = abs(topK[0].score - topK[1].score)
                if diff < self.max_diff:
                    return topK[0]
                else:
                    return ClassificationResult(
                        classification=None,
                        score=0,
                        page=results[0].page,
                        classifier=None,
                    )
            else:
                raise ValueError(f"Unknown tie break policy {self.tie_break_policy}")

        return ClassificationResult(
            classification=max_score_label,
            score=round(total_score / len(candidates), 4),
            page=candidates[0].page,
            classifier=",".join([r.classifier for r in candidates]),
            sub_classifier=candidates[0].sub_classifier,  # get first sub_classifier
        )


class MaxScoreVoter(VotingStrategy):
    def __init__(
        self,
        tie_break_policy: Union[TieBreakPolicy, str] = "abstain",
        weights: list[float] = None,
        silent=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.silent = silent
        if isinstance(tie_break_policy, str):
            tie_break_policy = TieBreakPolicy(tie_break_policy)
        self.tie_break_policy = tie_break_policy  # type: TieBreakPolicy
        # max_diff is only used when tie_break_policy is BEST_WITH_DIFF
        self.max_diff = kwargs.get("max_diff", 0.1)

    def vote(self, results: List[ClassificationResult]) -> ClassificationResult:
        if not results:
            if not self.silent:
                raise ValueError("No results to vote on")
            else:
                return ClassificationResult(
                    classification=None, score=0, page=0, classifier=None
                )

        if not self.weights:
            self.weights = [1] * len(results)

        if len(self.weights) != len(results):
            raise ValueError(
                "Number of weights must be equal to the number of results, or empty list for equal weights."
            )

        if len(results) == 1:
            return results[0]

        max_score = max([result.score * w for result, w in zip(results, self.weights)])
        max_score_result = next(
            (
                result
                for i, result in enumerate(results)
                if result.score * self.weights[i] == max_score
            ),
            None,
        )
        # check for tie

        if (
            self.tie_break_policy == TieBreakPolicy.ABSTAIN
            and len(
                [
                    result
                    for i, result in enumerate(results)
                    if result.score * self.weights[i] == max_score
                ]
            )
            > 1
        ):
            return ClassificationResult(
                classification=None, score=0, page=results[0].page, classifier=None
            )
        elif self.tie_break_policy == TieBreakPolicy.BEST_WITH_DIFF:
            tmp_results = [
                ClassificationResult(
                    classification=result.classification,
                    score=result.score * self.weights[i],
                    page=result.page,
                    classifier=result.classifier,
                    sub_classifier=result.sub_classifier,
                )
                for i, result in enumerate(results)
            ]
            topK = sorted(tmp_results, key=lambda x: x.score, reverse=True)[:2]
            diff = abs(topK[0].score - topK[1].score)
            if diff < self.max_diff:
                return topK[0]
            else:
                return ClassificationResult(
                    classification=None, score=0, page=results[0].page, classifier=None
                )

        return max_score_result


def get_voting_strategy(
    policy: Union[str | VotingPolicy],
    tie_break_policy: Union[str | TieBreakPolicy] = "abstain",
    **kwargs,
) -> VotingStrategy:
    """
    Returns a voting strategy based on the provided policy and tie break policy.

    :param policy: The voting policy to be used. This can be a string or an instance of VotingPolicy.
    :param tie_break_policy: The policy to be used when there is a tie. This can be a string or an instance of TieBreakPolicy. Default is "abstain".
    :param kwargs: Additional keyword arguments.
    :return: An instance of a voting strategy.
    :raises ValueError: If the provided policy or tie_break_policy is not recognized.

    :Example:
        .. highlight:: python
        .. code-block:: python

            from marie.pipe.voting import get_voting_strategy
            from marie.pipe.model import ClassificationResult

            voter = get_voting_strategy("majority", "best_with_diff", silent=True, max_diff=0.25)
            results = [
                ClassificationResult(classification="A", score=0.7, page=0, classifier="A"),
                ClassificationResult(classification="B", score=0.8, page=0, classifier="B"),
                ClassificationResult(classification="C", score=0.6, page=0, classifier="B"),
            ]
            result = voter.vote(results)
    """
    if isinstance(policy, str):
        policy = VotingPolicy(policy)
    if policy == VotingPolicy.MAJORITY:
        return MajorityVoter(tie_break_policy, **kwargs)
    elif policy == VotingPolicy.MAX_SCORE:
        return MaxScoreVoter(tie_break_policy, **kwargs)
    else:
        raise ValueError(f"Unknown voting policy {policy}")
