import pytest

from fast_agents.helpers.path_resolver import resolve_path_expressions


class TestResolvePathExpressions:
    def test_simple_field(self):
        data = {"a": 1, "b": 2}
        results = resolve_path_expressions(data, "$.a")
        assert results == [((("a")), 1)] or results == [(("a",), 1)]

    def test_nested_fields(self):
        data = {"a": {"b": {"c": 3}}}
        results = resolve_path_expressions(data, "$.a.b.c")
        assert results == [(("a", "b", "c"), 3)]

    def test_list_wildcard(self):
        data = {"items": [10, 20, 30]}
        results = resolve_path_expressions(data, "$.items[*]")
        # Expect three entries with loc including key and index
        assert len(results) == 3
        locs = [loc for loc, _ in results]
        values = [v for _, v in results]
        assert ("items", "0") in locs and ("items", "1") in locs and ("items", "2") in locs
        assert values == [10, 20, 30]

    def test_combination_list_then_field(self):
        data = {"a": {"list": [{"b": 1}, {"b": 2}, {"b": 3}]}}
        # First resolve to list items, then we can select 'b' from each item
        intermediate = resolve_path_expressions(data, "$.a.list[*]")
        assert len(intermediate) == 3
        # Manually follow with 'b' to validate returned loc/value consistency
        b_results = []
        for loc, item in intermediate:
            if isinstance(item, dict) and "b" in item:
                b_results.append((loc + ("b",), item["b"]))
        assert b_results == [
            (("a", "list", "0", "b"), 1),
            (("a", "list", "1", "b"), 2),
            (("a", "list", "2", "b"), 3),
        ]

    def test_missing_path_returns_empty(self):
        data = {"a": {"x": 1}}
        results = resolve_path_expressions(data, "$.a.b")
        assert results == []

    def test_invalid_path_raises(self):
        data = {"a": 1}
        with pytest.raises(ValueError):
            resolve_path_expressions(data, "a.b")


