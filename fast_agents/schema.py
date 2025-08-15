from __future__ import annotations

from typing import Any, List, Sequence, Tuple

from pydantic import BaseModel, ConfigDict

from fast_agents.validator_rule import ValidatorRule
from fast_agents.exceptions import ValidationRuleException
from fast_agents.helpers.path_resolver import resolve_path_expressions


class Schema(BaseModel):
    """
    Base schema with optional post-parse rule validation.

    Users declare rules via an inner Meta class:

        class MySchema(Schema):
            ...fields...
            class Meta:
                rules = [
                    Schema.Rule("$.field", []),
                ]
    """

    # Allow arbitrary types (e.g., bson.ObjectId) by default in schemas
    model_config = ConfigDict(arbitrary_types_allowed=True)

    class Rule:
        def __init__(self, path: str, validators: List[ValidatorRule]):
            self.path = path
            self.validators = validators

    class Meta:  # type: ignore[empty-body]
        rules: List['Schema.Rule'] = []

    async def validate(self, *, partial: bool = False) -> None:
        data = self.model_dump(exclude_unset=partial)

        rules: List[Schema.Rule] = getattr(self.Meta, 'rules', []) or []
        if not rules:
            return

        errors: List[dict[str, Any]] = []
        for rule in rules:
            matches: List[Tuple[Sequence[str], Any]] = resolve_path_expressions(data, rule.path)
            for loc, value in matches:
                for validator in rule.validators:
                    try:
                        await validator.validate(value=value, data=data, loc=loc)
                    except ValidationRuleException as exc:
                        if getattr(exc, 'errors', None):
                            errors.extend(exc.errors)  # type: ignore[arg-type]
                        else:
                            errors.append({
                                "loc": tuple(loc) if loc else tuple(),
                                "msg": str(exc),
                                "type": getattr(exc, 'error_type', 'rule_error'),
                            })

        if errors:
            raise ValidationRuleException(
                "schema rule validation failed",
                loc=tuple(),
                error_type="rule_error",
                errors=errors,
            )


