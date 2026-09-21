from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

FactValue = int | str

_POOL_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
_MAX_POOLS = 1000
_MAX_IN_VALUES = 64
_MAX_STRING_BYTES = 128
_MAX_RULE_BYTES = 16 * 1024
_CATCH_ALL_PRIORITY = 1_000_000


@dataclass(frozen=True, slots=True)
class _FactDefinition:
    value_type: type[int] | type[str]
    values: frozenset[str] | None = None


_FACTS: dict[str, _FactDefinition] = {
    'document.total_page_count': _FactDefinition(int),
    'document.requested_page_count': _FactDefinition(int),
    'document.effective_page_count': _FactDefinition(int),
    'workload.kind': _FactDefinition(
        str, frozenset({'document', 'text', 'image', 'agent'})
    ),
    'workload.mode': _FactDefinition(
        str, frozenset({'interactive', 'batch', 'backfill', 'background'})
    ),
    'pipeline.stage': _FactDefinition(
        str, frozenset({'extract', 'validate', 'enrich'})
    ),
    'request.source': _FactDefinition(
        str,
        frozenset(
            {
                'gateway-job-api',
                'studio-submission',
                'workflow',
                'operator',
            }
        ),
    ),
}


class AdmissionPolicyError(ValueError):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


class AdmissionMatchError(ValueError):
    def __init__(self, category: str) -> None:
        self.category = category
        super().__init__(category)


@dataclass(frozen=True, slots=True)
class AdmissionCondition:
    fact_name: str
    eq: FactValue | None = None
    values: tuple[FactValue, ...] = ()
    gte: int | None = None
    lte: int | None = None

    def matches(self, facts: Mapping[str, FactValue]) -> bool:
        value = facts.get(self.fact_name)
        if value is None:
            return False
        return (
            (self.eq is None or value == self.eq)
            and (not self.values or value in self.values)
            and (
                self.gte is None
                or isinstance(value, int)
                and not isinstance(value, bool)
                and value >= self.gte
            )
            and (
                self.lte is None
                or isinstance(value, int)
                and not isinstance(value, bool)
                and value <= self.lte
            )
        )

    def to_snapshot(self) -> dict[str, object]:
        result: dict[str, object] = {}
        if self.eq is not None:
            result['eq'] = self.eq
        if self.values:
            result['in'] = list(self.values)
        if self.gte is not None:
            result['gte'] = self.gte
        if self.lte is not None:
            result['lte'] = self.lte
        return result


@dataclass(frozen=True, slots=True)
class AdmissionRule:
    pool_id: str
    enabled: bool
    priority: int
    accepting: bool
    conditions: tuple[AdmissionCondition, ...]
    rule_digest: str

    def matches(self, facts: Mapping[str, FactValue]) -> bool:
        return (
            self.enabled
            and self.accepting
            and all(condition.matches(facts) for condition in self.conditions)
        )

    def to_snapshot(self) -> dict[str, object]:
        return {
            'pool_id': self.pool_id,
            'enabled': self.enabled,
            'admission': {
                'schema_version': 1,
                'priority': self.priority,
                'accepting': self.accepting,
                'match': {
                    condition.fact_name: condition.to_snapshot()
                    for condition in self.conditions
                },
            },
            'rule_digest': self.rule_digest,
        }


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    pool_id: str
    priority: int
    policy_generation: int
    policy_digest: str
    rule_digest: str
    normalized_fact_digest: str
    matched_facts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PoolEndpointBinding:
    pool_id: str
    endpoint_group_id: str
    revision: str


@dataclass(frozen=True, slots=True)
class AdmissionPolicy:
    fabric_group_id: str
    generation: int
    policy_digest: str
    rules: tuple[AdmissionRule, ...]
    endpoint_bindings: tuple[PoolEndpointBinding, ...] = ()

    @classmethod
    def from_rows(
        cls,
        fabric_group_id: str,
        generation: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> AdmissionPolicy:
        if not fabric_group_id or type(generation) is not int or generation < 1:
            raise AdmissionPolicyError('routing_policy_identity_invalid')
        if len(rows) > _MAX_POOLS:
            raise AdmissionPolicyError('routing_policy_pool_limit')

        seen: set[str] = set()
        rules: list[AdmissionRule] = []
        endpoint_bindings: list[PoolEndpointBinding] = []
        for row in rows:
            pool_id = row.get('pool_id')
            if not isinstance(pool_id, str) or not _POOL_ID.fullmatch(pool_id):
                raise AdmissionPolicyError('routing_policy_pool_invalid')
            if pool_id in seen:
                raise AdmissionPolicyError('routing_policy_duplicate_pool')
            seen.add(pool_id)

            enabled = row.get('enabled')
            if type(enabled) is not bool:
                raise AdmissionPolicyError('routing_policy_pool_invalid')
            metadata = row.get('metadata')
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, Mapping):
                raise AdmissionPolicyError('routing_policy_metadata_invalid')
            admission = metadata.get('admission')
            if admission is None:
                continue
            rules.append(_parse_rule(pool_id, enabled, admission))
            dispatch = metadata.get('llm_dispatch')
            if dispatch is not None:
                endpoint_bindings.append(_parse_endpoint_binding(pool_id, dispatch))

        normalized = tuple(sorted(rules, key=lambda rule: rule.pool_id))
        _validate_complete_rule_set(normalized)
        snapshot = _snapshot(fabric_group_id, normalized)
        return cls(
            fabric_group_id=fabric_group_id,
            generation=generation,
            policy_digest=_digest(snapshot),
            rules=normalized,
            endpoint_bindings=tuple(
                sorted(endpoint_bindings, key=lambda binding: binding.pool_id)
            ),
        )

    def to_snapshot(self) -> dict[str, object]:
        return _snapshot(self.fabric_group_id, self.rules)

    def match(self, facts: Mapping[str, FactValue]) -> RoutingDecision:
        normalized = _normalize_facts(facts)
        matches = sorted(
            (rule for rule in self.rules if rule.matches(normalized)),
            key=lambda rule: rule.priority,
        )
        if not matches:
            raise AdmissionMatchError('routing_no_match')
        if len(matches) > 1 and matches[0].priority == matches[1].priority:
            raise AdmissionMatchError('routing_ambiguous')
        rule = matches[0]
        return RoutingDecision(
            pool_id=rule.pool_id,
            priority=rule.priority,
            policy_generation=self.generation,
            policy_digest=self.policy_digest,
            rule_digest=rule.rule_digest,
            normalized_fact_digest=_digest(normalized),
            matched_facts=tuple(condition.fact_name for condition in rule.conditions),
        )

    def endpoint_binding(self, pool_id: str) -> PoolEndpointBinding:
        for binding in self.endpoint_bindings:
            if binding.pool_id == pool_id:
                return binding
        raise AdmissionMatchError('routing_endpoint_binding_missing')


def _parse_endpoint_binding(pool_id: str, value: object) -> PoolEndpointBinding:
    if not isinstance(value, Mapping):
        raise AdmissionPolicyError('routing_policy_endpoint_invalid')
    endpoint_id = value.get('endpoint_group_id') or value.get('endpoint_id')
    revision = value.get('revision')
    if (
        value.get('schema_version') != 1
        or not isinstance(endpoint_id, str)
        or not _POOL_ID.fullmatch(endpoint_id)
        or not isinstance(revision, str)
        or not _POOL_ID.fullmatch(revision)
    ):
        raise AdmissionPolicyError('routing_policy_endpoint_invalid')
    return PoolEndpointBinding(pool_id, endpoint_id, revision)


def _parse_rule(pool_id: str, enabled: bool, value: object) -> AdmissionRule:
    if not isinstance(value, Mapping):
        raise AdmissionPolicyError('routing_policy_metadata_invalid')
    if len(_json(value).encode()) > _MAX_RULE_BYTES:
        raise AdmissionPolicyError('routing_policy_metadata_too_large')
    if set(value) != {'schema_version', 'priority', 'accepting', 'match'}:
        raise AdmissionPolicyError('routing_policy_unknown_key')
    if value['schema_version'] != 1 or type(value['schema_version']) is not int:
        raise AdmissionPolicyError('routing_policy_schema_unsupported')
    priority = value['priority']
    accepting = value['accepting']
    match = value['match']
    if type(priority) is not int or not 0 <= priority <= _CATCH_ALL_PRIORITY:
        raise AdmissionPolicyError('routing_policy_priority_invalid')
    if type(accepting) is not bool or not isinstance(match, Mapping):
        raise AdmissionPolicyError('routing_policy_metadata_invalid')
    if accepting and not enabled:
        raise AdmissionPolicyError('routing_policy_pool_disabled')
    if len(match) > len(_FACTS):
        raise AdmissionPolicyError('routing_policy_unknown_fact')

    conditions = tuple(
        _parse_condition(fact_name, condition)
        for fact_name, condition in sorted(match.items())
    )
    normalized = {
        'pool_id': pool_id,
        'enabled': enabled,
        'schema_version': 1,
        'priority': priority,
        'accepting': accepting,
        'match': {
            condition.fact_name: condition.to_snapshot() for condition in conditions
        },
    }
    return AdmissionRule(
        pool_id=pool_id,
        enabled=enabled,
        priority=priority,
        accepting=accepting,
        conditions=conditions,
        rule_digest=_digest(normalized),
    )


def _parse_condition(fact_name: object, value: object) -> AdmissionCondition:
    if not isinstance(fact_name, str) or fact_name not in _FACTS:
        raise AdmissionPolicyError('routing_policy_unknown_fact')
    if not isinstance(value, Mapping) or not value:
        raise AdmissionPolicyError('routing_policy_value_invalid')
    allowed = {'eq', 'in', 'gte', 'lte'}
    if not set(value).issubset(allowed):
        raise AdmissionPolicyError('routing_policy_unknown_key')
    if 'eq' in value and 'in' in value:
        raise AdmissionPolicyError('routing_policy_operator_conflict')

    definition = _FACTS[fact_name]
    if definition.value_type is str and ({'gte', 'lte'} & set(value)):
        raise AdmissionPolicyError('routing_policy_operator_type_mismatch')

    eq = _validate_fact_value(fact_name, value['eq']) if 'eq' in value else None
    values: tuple[FactValue, ...] = ()
    if 'in' in value:
        raw_values = value['in']
        if (
            not isinstance(raw_values, list)
            or not raw_values
            or len(raw_values) > _MAX_IN_VALUES
        ):
            raise AdmissionPolicyError('routing_policy_value_invalid')
        parsed = tuple(_validate_fact_value(fact_name, item) for item in raw_values)
        if len(set(parsed)) != len(parsed):
            raise AdmissionPolicyError('routing_policy_value_invalid')
        values = tuple(sorted(parsed))

    gte = _validate_bound(value.get('gte')) if 'gte' in value else None
    lte = _validate_bound(value.get('lte')) if 'lte' in value else None
    if gte is not None and lte is not None and gte > lte:
        raise AdmissionPolicyError('routing_policy_value_invalid')
    if eq is not None and isinstance(eq, int) and not _within(eq, gte, lte):
        raise AdmissionPolicyError('routing_policy_value_invalid')
    if values and definition.value_type is int:
        values = tuple(item for item in values if _within(int(item), gte, lte))
        if not values:
            raise AdmissionPolicyError('routing_policy_value_invalid')
    return AdmissionCondition(fact_name, eq, values, gte, lte)


def _validate_bound(value: object) -> int:
    if type(value) is not int or value < 1 or value > 2**31 - 1:
        raise AdmissionPolicyError('routing_policy_value_invalid')
    return value


def _validate_fact_value(fact_name: str, value: object) -> FactValue:
    definition = _FACTS[fact_name]
    if definition.value_type is int:
        if type(value) is not int or not 1 <= value <= 2**31 - 1:
            raise AdmissionPolicyError('routing_policy_value_invalid')
        return value
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode()) > _MAX_STRING_BYTES
    ):
        raise AdmissionPolicyError('routing_policy_value_invalid')
    if definition.values is not None and value not in definition.values:
        raise AdmissionPolicyError('routing_policy_value_invalid')
    return value


def _normalize_facts(facts: Mapping[str, FactValue]) -> dict[str, FactValue]:
    if not isinstance(facts, Mapping):
        raise AdmissionMatchError('routing_facts_invalid')
    normalized: dict[str, FactValue] = {}
    try:
        for fact_name, value in sorted(facts.items()):
            if fact_name not in _FACTS:
                raise AdmissionMatchError('routing_facts_invalid')
            normalized[fact_name] = _validate_fact_value(fact_name, value)
    except AdmissionPolicyError as exc:
        raise AdmissionMatchError('routing_facts_invalid') from exc
    return normalized


def _validate_complete_rule_set(rules: tuple[AdmissionRule, ...]) -> None:
    catch_all = [
        rule
        for rule in rules
        if rule.enabled and rule.accepting and not rule.conditions
    ]
    if len(catch_all) != 1 or catch_all[0].priority != _CATCH_ALL_PRIORITY:
        raise AdmissionPolicyError('routing_policy_catch_all_required')

    accepting = [rule for rule in rules if rule.enabled and rule.accepting]
    for index, left in enumerate(accepting):
        for right in accepting[index + 1 :]:
            if left.priority == right.priority and _rules_overlap(left, right):
                raise AdmissionPolicyError('routing_policy_ambiguous')


def _rules_overlap(left: AdmissionRule, right: AdmissionRule) -> bool:
    left_conditions = {item.fact_name: item for item in left.conditions}
    right_conditions = {item.fact_name: item for item in right.conditions}
    for fact_name in left_conditions.keys() & right_conditions.keys():
        if not _conditions_overlap(
            left_conditions[fact_name], right_conditions[fact_name]
        ):
            return False
    return True


def _conditions_overlap(left: AdmissionCondition, right: AdmissionCondition) -> bool:
    left_values = _condition_values(left)
    right_values = _condition_values(right)
    if left_values is not None:
        return any(right.matches({right.fact_name: value}) for value in left_values)
    if right_values is not None:
        return any(left.matches({left.fact_name: value}) for value in right_values)
    left_low = left.gte or 1
    right_low = right.gte or 1
    left_high = left.lte or 2**31 - 1
    right_high = right.lte or 2**31 - 1
    return max(left_low, right_low) <= min(left_high, right_high)


def _condition_values(condition: AdmissionCondition) -> tuple[FactValue, ...] | None:
    if condition.eq is not None:
        return (condition.eq,)
    if condition.values:
        return condition.values
    definition = _FACTS[condition.fact_name]
    if definition.values is not None:
        return tuple(sorted(definition.values))
    return None


def _within(value: int, low: int | None, high: int | None) -> bool:
    return (low is None or value >= low) and (high is None or value <= high)


def _snapshot(
    fabric_group_id: str, rules: Sequence[AdmissionRule]
) -> dict[str, object]:
    return {
        'schema_version': 1,
        'fabric_group_id': fabric_group_id,
        'rules': [
            rule.to_snapshot() for rule in sorted(rules, key=lambda x: x.pool_id)
        ],
    }


def _json(value: object) -> str:
    try:
        return json.dumps(
            value,
            separators=(',', ':'),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AdmissionPolicyError('routing_policy_value_invalid') from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()
