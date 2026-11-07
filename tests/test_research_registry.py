from stockresearch.research_registry import METHODS, get_method, list_methods


def test_registry_covers_exactly_the_implemented_test_ids():
    expected = {*(f"T{i:02d}" for i in range(1, 19))}
    assert set(METHODS) == expected
    assert len(METHODS) == len(expected)


def test_registry_entries_have_required_research_metadata():
    for method in list_methods():
        assert method.question.strip()
        assert method.domain.strip()
        assert method.methods
        assert method.audience_note.strip()
        assert method.limitations
        assert method.status == "implemented"


def test_domain_filter_returns_only_matching_methods():
    volatility = list_methods("volatility")
    assert volatility
    assert all(method.domain == "volatility" for method in volatility)


def test_unknown_method_has_clear_error():
    try:
        get_method("T99")
    except KeyError as exc:
        assert "Unknown research method" in str(exc)
    else:
        raise AssertionError("Unknown method should fail")
