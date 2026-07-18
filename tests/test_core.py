
from optimizer.compiler import compile_prompt
from optimizer.evaluators import extract_json
from optimizer.models import PromptModules
from optimizer.stats import paired_deltas, sign_test


def test_compiler_rejects_missing_variable():
    try:
        compile_prompt(PromptModules(role="r"), {})
    except Exception as exc:
        assert "user_input" in str(exc)
    else:
        raise AssertionError("StrictUndefined did not fail")


def test_json_extraction_rejects_trailing_text():
    assert extract_json('{"ok": true}') == '{"ok": true}'
    try:
        extract_json('{"ok": true} explanation')
    except ValueError:
        pass
    else:
        raise AssertionError("trailing content was accepted")


def test_paired_statistics_ignore_ties():
    deltas, wins, losses, ties = paired_deltas({"a": 0, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 0})
    assert deltas == [1, 0, -1]
    assert (wins, losses, ties) == (1, 1, 1)
    assert sign_test(wins, losses) == 1.0
