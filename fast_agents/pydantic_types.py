from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Annotated, Optional, Any

from bson import ObjectId
from pydantic import PlainSerializer
from pydantic.functional_validators import BeforeValidator


def _to_object_id(value: object) -> Optional[ObjectId]:
    if value is None or isinstance(value, ObjectId):
        return value  # type: ignore[return-value]
    if isinstance(value, str) and ObjectId.is_valid(value):
        return ObjectId(value)
    raise ValueError("Invalid ObjectId")


def _to_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        s = value.strip()
        try:
            return date.fromisoformat(s)
        except ValueError:
            if "T" in s or " " in s:
                s_norm = s.replace("Z", "+00:00")
                try:
                    return datetime.fromisoformat(s_norm).date()
                except ValueError:
                    pass
    raise ValueError("Invalid date")


def _to_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        s = value.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            pass
    raise ValueError("Invalid datetime")


ObjectIdField = Annotated[
    ObjectId,
    BeforeValidator(_to_object_id),
    PlainSerializer(lambda v: str(v) if v is not None else None),
]

DateField = Annotated[
    date,
    BeforeValidator(_to_date),
    PlainSerializer(lambda v: v.isoformat() if v is not None else None),
]

DateTimeField = Annotated[
    datetime,
    BeforeValidator(_to_datetime),
    PlainSerializer(lambda v: v.isoformat() if v is not None else None),
]


def _extract_json(value: Any) -> Any:
    # Accept dict/list directly, or JSON-parse strings; otherwise return as-is
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        import json

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value


JSONField = Annotated[
    Any,
    BeforeValidator(_extract_json),
]


__all__ = [
    "ObjectIdField",
    "DateField",
    "DateTimeField",
    "JSONField",
]


