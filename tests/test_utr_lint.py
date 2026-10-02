# -*- coding: utf-8 -*-
"""Tests for scripts/utr-lint.py.

The suite runs in two configurations: with and without pymorphy3. Tests
that need morphology are skipped when the library is absent, so the zero
dependency promise stays testable.

Run:
    python -m pytest tests -q
    python -m pytest tests -q -k clean
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LINTER = ROOT / "scripts" / "utr-lint.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"

STRICT = "strict"
NORMAL = "normal"


def _load_linter():
    """Import the hyphenated script as a module."""
    spec = importlib.util.spec_from_file_location("utr_lint", LINTER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses needs the module to be registered before execution.
    sys.modules["utr_lint"] = module
    spec.loader.exec_module(module)
    return module


utr = _load_linter()
Morphology = utr.Morphology
MORPHOLOGY = Morphology()
HAS_MORPHOLOGY = MORPHOLOGY.available
needs_morphology = pytest.mark.skipif(
    not HAS_MORPHOLOGY, reason="нужна библиотека pymorphy3"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def lint(text: str, mode: str = STRICT):
    return utr.analyze(text, mode=mode, morphology=MORPHOLOGY).findings


def rules_of(text: str, mode: str = STRICT) -> set[str]:
    return {item.rule for item in lint(text, mode)}


def findings_of(text: str, rule: str, mode: str = STRICT) -> list:
    return [item for item in lint(text, mode) if item.rule == rule]


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def words(count: int, prefix: str = "слово") -> str:
    return " ".join("%s%d" % (prefix, index) for index in range(1, count + 1)) + "."


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------


def test_clean_prose_has_no_findings():
    text = (
        "Откройте файл `config.yaml` и найдите параметр timeout.\n"
        "Установите timeout в 30 секунд.\n"
        "Запустите сервер командой `docker compose up -d`.\n"
        "Проверьте журнал: команда `docker logs app` показывает порт 8080.\n"
        "Если порт занят, остановите процесс командой `docker compose down`.\n"
    )
    assert lint(text) == []


def test_clean_fixture_has_no_findings():
    assert lint(read_fixture("clean.md")) == []
    assert lint(read_fixture("clean.md"), NORMAL) == []


def test_empty_and_trivial_input():
    assert lint("") == []
    assert lint("\n\n\n") == []
    assert lint("## Заголовок\n") == []
    assert lint("```\nкод\n```") == []


def test_bom_is_stripped():
    assert lint("\ufeff# Заголовок\n\nПроверьте журнал.\n") == []


# ---------------------------------------------------------------------------
# Modality is content, never a violation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Запрос может завершиться ошибкой.",
    "Возможно, причина в таймауте.",
    "Вероятно, сервер перегружен.",
    "Иногда ошибка возникает после перезапуска.",
    "Клиент, вероятно, повторит запрос автоматически.",
    "Данные, возможно, устарели.",
    "Результат может быть неполным.",
    "Сервер, скорее всего, перегружен.",
])
def test_modality_is_never_flagged(text: str):
    assert lint(text) == []


def test_hedge_blocked_sentence_is_clean():
    text = (
        "Ошибка 502 может означать, что шлюз не получил ответ.\n"
        "Вероятно, причина в недоступности сервера.\n"
    )
    assert lint(text) == []


# ---------------------------------------------------------------------------
# Long sentences
# ---------------------------------------------------------------------------


def test_long_sentence_is_hard_in_strict_mode():
    findings = findings_of(words(21), "long-sentence")
    assert len(findings) == 1
    assert findings[0].level == "hard"
    assert findings[0].match == "21 слов"
    assert findings[0].suggestion == "Разделите предложение"


def test_long_sentence_is_advisory_in_normal_mode():
    findings = findings_of(words(21), "long-sentence", NORMAL)
    assert len(findings) == 1
    assert findings[0].level == "advisory"


def test_absolute_cap_is_hard_in_both_modes():
    assert findings_of(words(26), "long-sentence")[0].level == "hard"
    assert findings_of(words(26), "long-sentence", NORMAL)[0].level == "hard"


def test_sentence_at_the_limit_is_clean():
    assert findings_of(words(20), "long-sentence") == []


def test_long_sentence_column_points_at_the_sentence_start():
    text = "Проверьте журнал. " + words(22)
    findings = findings_of(text, "long-sentence")
    assert findings[0].col == 19


def test_abbreviation_dot_is_not_a_sentence_end():
    # "т. д." must not split the sentence into fragments.
    text = "Используйте сокращения т. д. и т. п. только по требованию заказчика."
    assert findings_of(text, "long-sentence") == []
    assert lint(text) == []


# ---------------------------------------------------------------------------
# Bureaucratic wording
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "В настоящее время сервис недоступен.",
    "Оператор осуществляет проверку в целях контроля.",
    "В рамках программы обновляется сервер.",
    "Представляется необходимым выполнить проверку.",
    "На данный момент ошибок нет.",
    "В связи с обновлением сервис недоступен.",
    "Проверьте данный модуль.",
    "Имеет место ошибка конфигурации.",
])
def test_bureaucratism_is_detected(text: str):
    assert "bureaucratism" in rules_of(text)


def test_bureaucratism_gives_a_replacement():
    findings = findings_of("В настоящее время сервис недоступен.", "bureaucratism")
    assert findings[0].suggestion == "сейчас"
    assert findings[0].level == "hard"


def test_noun_data_is_not_a_bureaucratic_adjective():
    assert findings_of("Передайте данные в отчёт.", "bureaucratism") == []
    assert lint("Передайте данные в отчёт.") == []


# ---------------------------------------------------------------------------
# Verb + noun pairs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,verb", [
    ("Произвести установку пакета.", "установить"),
    ("Выполнить проверку журнала.", "проверить"),
    ("Осуществить проверку состояния.", "проверить"),
    ("Произвести изменение настроек.", "изменить"),
    ("Осуществлять использование старого API запрещено.", "использовать"),
    ("Выполнить настройку окружения.", "настроить"),
    ("Провести проверку сборки.", "проверить"),
    ("Провести проверку сборки.", "проверить"),
    ("Обеспечить выполнение работ в срок.", "выполнить"),
    ("Принять решение о продолжении.", "решить"),
    ("Оказать влияние на скорость.", "повлиять"),
])
def test_verb_object_noun_replacements(text: str, verb: str):
    findings = findings_of(text, "verb-object-noun")
    assert len(findings) == 1, text
    assert findings[0].level == "hard"
    assert verb in (findings[0].suggestion or "")


def test_plain_verbs_are_not_flagged():
    text = "Установите пакет. Проверьте журнал. Настройте окружение."
    assert findings_of(text, "verb-object-noun") == []
    assert lint(text) == []


def test_verb_object_noun_survives_an_inflected_form():
    findings = findings_of("Выполните проверку состояния узла.", "verb-object-noun")
    assert len(findings) == 1
    assert findings[0].match.lower().endswith("проверку")


def test_nominalization_counts_only_real_action_nouns():
    assert findings_of("Проверка конфигурации занимает 5 мс.",
                       "nominalization") == []
    heavy = findings_of(
        "Обработка запросов, проверка ответов, сохранение результатов и "
        "синхронизация кэша выполняются в фоне.", "nominalization")
    assert len(heavy) == 1
    assert heavy[0].level == "hard"


def test_nominalization_leading_topic_noun_is_not_counted_twice():
    text = ("Настройка сервера, проверка журнала и сохранение конфигурации "
            "завершены.")
    findings = findings_of(text, "nominalization")
    assert len(findings) == 1
    assert findings[0].match == "проверка"


# ---------------------------------------------------------------------------
# Passive voice
# ---------------------------------------------------------------------------


def test_passive_voice_is_advisory():
    findings = findings_of("Файл был удалён после проверки журнала.",
                           "passive-voice")
    assert findings and all(item.level == "advisory" for item in findings)


def test_active_voice_is_not_flagged():
    assert findings_of("Пользователь удалил файл сам.", "passive-voice") == []


@pytest.mark.parametrize("text", [
    "Файл был удалён сервером.",
    "Данные были обработаны в фоне.",
    "Отчёт был сформирован автоматически.",
])
def test_short_passive_forms_are_found(text: str):
    assert "passive-voice" in rules_of(text)


# ---------------------------------------------------------------------------
# Terminology: one term per concept
# ---------------------------------------------------------------------------


def test_synonym_rotation_names_the_kept_term():
    findings = findings_of(
        "Проверьте конфигурацию. Затем валидируйте ответ сервера.",
        "synonym-rotation")
    assert len(findings) == 1
    assert "«проверить»" in findings[0].message
    assert findings[0].level == "hard"


def test_consistent_terminology_is_silent():
    assert findings_of("Проверьте конфигурацию. Проверьте ответ.",
                       "synonym-rotation") == []


def test_adjective_form_of_a_term_is_not_a_rotation():
    assert findings_of("Проверьте конфигурацию и проверочный лист.",
                       "synonym-rotation") == []


def test_synonym_rotation_is_scoped_per_file():
    findings = findings_of("Удалите файл. Затем стирайте кэш.", "synonym-rotation")
    assert len(findings) == 1


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def test_dangling_conjunction_in_list_items():
    text = (
        "- Установите целевой объект и\n"
        "* Запишите результат или  \n"
        "+ Закройте панель\n"
        "1. Запустите задачу и\n"
        "2) Остановите задачу или\n"
    )
    findings = findings_of(text, "dangling-conjunction")
    assert [item.line for item in findings] == [1, 2, 4, 5]
    assert all(item.level == "hard" for item in findings)


def test_closed_list_item_is_fine():
    assert findings_of("- Установите цель и.\n- Остановите цель или.",
                       "dangling-conjunction") == []


def test_list_continuation_lines_are_joined():
    assert findings_of("  - Установите цель и\n    запишите результат.",
                       "dangling-conjunction") == []


def test_dangling_conjunction_column():
    findings = findings_of("- Установите цель\n  и", "dangling-conjunction")
    assert len(findings) == 1
    assert findings[0].line == 2
    assert findings[0].col == 3


def test_four_space_line_is_code_not_a_list():
    assert findings_of("    - код и", "dangling-conjunction") == []


def test_blockquoted_list_is_skipped():
    assert findings_of("> - Установите цель и\n> - Запишите результат или",
                       "dangling-conjunction") == []


def test_nested_items_are_checked():
    findings = findings_of("- Родительский пункт и\n  - Вложенный пункт или",
                           "dangling-conjunction")
    assert [item.line for item in findings] == [1, 2]


def test_inline_code_is_not_a_conjunction():
    assert findings_of("- Откройте `and` как метку", "dangling-conjunction") == []
    assert findings_of("- Объедините `left` и `right`",
                       "dangling-conjunction") == []


def test_fenced_list_is_not_prose():
    assert findings_of("```text\n- код и\n```", "dangling-conjunction") == []
    assert findings_of("~~~\n- код и\n~~~", "dangling-conjunction") == []


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def test_inline_code_is_masked():
    text = "Выполните проверку; в целях контроля изменяйте `config; data`."
    findings = lint(text)
    assert all(item.match != "`config; data`" for item in findings)
    # only the real semicolon is reported
    semicolons = findings_of(text, "semicolon")
    assert len(semicolons) == 1


def test_fenced_code_is_not_analysed():
    text = (
        "```bash\n"
        "python -m build --wheel; rm -rf dist\n"
        "// This is a comment; with a semicolon\n"
        "```\n"
    )
    assert lint(text) == []


def test_tilde_fence_is_not_analysed():
    assert lint("~~~\nПроверьте; в целях контроля.\n~~~\n") == []


def test_markdown_table_cells_are_linted_independently():
    cell = words(26)
    text = "| Поле | Значение |\n| --- | --- |\n| timeout | %s |\n" % cell
    findings = findings_of(text, "long-sentence")
    assert len(findings) == 1
    assert findings[0].match == "26 слов"
    assert findings[0].line == 3


def test_table_without_leading_pipes():
    text = "Поле | Значение\n--- | ---\ntimeout | %s\n" % words(26)
    assert len(findings_of(text, "long-sentence")) == 1


def test_pipe_in_prose_is_not_a_table():
    text = "Используйте команда a | b; затем проверьте результат."
    assert any(item.rule == "semicolon" for item in lint(text))


def test_escaped_pipe_stays_in_the_cell():
    text = "| a | b |\n| --- | --- |\n| x \\| y | %s |\n" % words(3)
    assert findings_of(text, "long-sentence") == []


def test_heading_and_blockquote_columns():
    text = "> Оператор в целях контроля проверяет узел.\n"
    findings = findings_of(text, "bureaucratism")
    assert len(findings) == 1
    assert findings[0].col == 12
    assert text[findings[0].col - 1:].startswith("в целях")


# ---------------------------------------------------------------------------
# Technical literals
# ---------------------------------------------------------------------------


def test_urls_are_masked():
    text = "Откройте https://example.com/a;b?c=1#d в браузере."
    assert lint(text) == []


def test_email_is_masked():
    assert lint("Письма приходят на ops@example.com.") == []


def test_windows_paths_are_masked():
    text = "Скопируйте C:\\Users\\user\\report.xlsx в D:\\reports."
    assert lint(text) == []


def test_windows_path_with_spaces_and_semicolons():
    text = ('Откройте "C:\\Program Files\\app; old\\config.yaml" '
            "и проверьте путь.")
    assert lint(text) == []


def test_posix_paths_are_masked():
    assert lint("Соберите проект командой ./scripts/build.sh --release.") == []


def test_cli_flags_are_masked():
    text = ("Запустите " +
            " ".join("--option-%d=значение%d" % (i, i) for i in range(1, 31)))
    assert findings_of(text, "long-sentence") == []


def test_env_assignment_is_masked():
    assert lint("Установите LOG_LEVEL=DEBUG перед запуском сервера.") == []


def test_json_object_is_masked():
    assert lint('Передайте {"mode": "strict", "limit": 20} сервису.') == []


def test_russian_quoted_text_is_not_masked():
    text = 'Проверьте значение "таймаут" в настройках сервиса.'
    assert lint(text) == []
    assert utr.QUOTED_LITERAL.search(text)


def test_commands_in_prose_are_masked():
    text = "Запустите `pytest -k smoke --json` и прочитайте вывод команды."
    assert lint(text) == []


# ---------------------------------------------------------------------------
# Complex sentences and vagueness
# ---------------------------------------------------------------------------


def test_complex_sentence():
    text = ("Сервер возвращает ошибку, если ключ не найден, и клиент должен "
            "повторить запрос, когда сеть снова станет доступна, потому что "
            "очередь не была обработана, а ответ не пришёл вовсе.")
    findings = findings_of(text, "complex-sentence")
    assert len(findings) == 1
    assert findings[0].level == "advisory"


def test_very_complex_sentence_is_hard():
    text = ("Если ключ не найден, и клиент повторил запрос, когда сеть снова "
            "доступна, потому что очередь не была обработана, а ответ не "
            "пришёл, то сервер вернёт ошибку, и агент повторит попытку, но "
            "клиент сообщит о сбое, и оператор проверит журнал.")
    assert findings_of(text, "complex-sentence")[0].level == "hard"


def test_short_sentence_complexity_is_not_judged():
    assert findings_of("Проверьте журнал и сообщите о результате.",
                       "complex-sentence") == []


@pytest.mark.parametrize("text", [
    "Сделайте это как можно скорее.",
    "Проверьте результат в кратчайшие сроки.",
    "Используйте достаточно большой таймаут.",
    "Работает корректно в большинстве случаев и тому подобное.",
    "В ряде случаев ошибка возникает.",
    "Обновите конфигурацию надлежащим образом.",
])
def test_vague_formulations_are_detected(text: str):
    assert "vague-formulation" in rules_of(text)


@pytest.mark.parametrize("text", [
    "Модуль уникальный и инновационный.",
    "Решение оптимальное и современное.",
    "Инструмент мощный и масштабируемый.",
])
def test_marketing_words_are_detected(text: str):
    assert "marketing-word" in rules_of(text)


# ---------------------------------------------------------------------------
# Semicolon
# ---------------------------------------------------------------------------


def test_semicolon_is_hard_in_both_modes():
    text = "Проверьте систему; потом запустите тесты."
    assert findings_of(text, "semicolon")[0].level == "hard"
    assert findings_of(text, "semicolon", NORMAL)[0].level == "hard"


def test_semicolon_column():
    findings = findings_of("Проверьте систему; потом запустите тесты.",
                           "semicolon")
    assert findings[0].col == 18


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------


@needs_morphology
def test_participle_clause_is_detected():
    findings = findings_of("Файл, созданный сервером, удалён после проверки.",
                           "participle-clause")
    assert findings and findings[0].level == "advisory"


@needs_morphology
def test_gerund_is_detected():
    findings = findings_of("Сервер запущен, закрыв соединение с клиентом.",
                           "participle-clause")
    assert findings and "Деепричастный" in findings[0].message


@needs_morphology
def test_noun_chain_is_detected():
    findings = findings_of("Настройка серверного модуля распределения нагрузки "
                           "выполнена.", "noun-chain")
    assert findings and findings[0].level == "advisory"


@needs_morphology
def test_short_noun_chain_is_fine():
    assert findings_of("Настройка выполнена.", "noun-chain") == []


@needs_morphology
def test_preposition_is_not_a_noun_in_a_chain():
    text = "Оператор в целях контроля проверяет узел."
    assert findings_of(text, "noun-chain") == []


@needs_morphology
def test_noun_chain_does_not_cross_a_sentence_boundary():
    text = ("Запрос может завершиться ошибкой. Возможно, причина в таймауте. "
            "Вероятно, сервер перегружен.")
    assert findings_of(text, "noun-chain") == []


def test_morphology_degrades_gracefully():
    broken = Morphology()
    broken._analyzer = object()
    broken.backend = "broken"
    assert broken.analyse("файл") == (False,) * 6
    assert utr.analyze("Проверьте журнал.", morphology=broken).findings == []


def test_morphology_backend_is_reported():
    findings = utr.analyze("Проверьте журнал.")
    assert findings.words == 2
    payload = json.loads(utr.render_json(
        findings.findings, findings, STRICT, 0, [], MORPHOLOGY))
    assert payload["morphology"] in {"builtin", "pymorphy3"}


# ---------------------------------------------------------------------------
# Positions, determinism, sorting
# ---------------------------------------------------------------------------


def test_positions_point_at_the_matched_text():
    text = "# Заголовок\n\nОператор в целях контроля проверяет узел.\n"
    for item in lint(text):
        line = text.splitlines()[item.line - 1]
        assert line[item.col - 1:item.col - 1 + len(item.match)] == item.match


def test_findings_are_sorted_and_stable():
    text = ("Оператор в целях контроля; произвести установку.\n"
            "- Установите объект и\n")
    first = [item.to_dict() for item in lint(text)]
    second = [item.to_dict() for item in lint(text)]
    assert first == second
    keys = [(item["line"], item["col"], item["rule"])
            for item in first]
    assert keys == sorted(keys)


def test_multiple_files_keep_their_own_terminology_scope():
    first = utr.analyze("Проверьте конфигурацию.\n", filename="a.md",
                        morphology=MORPHOLOGY)
    second = utr.analyze("Проверьте конфигурацию.\n", filename="b.md",
                         morphology=MORPHOLOGY)
    assert [item.to_dict() for item in first.findings] == \
           [item.to_dict() for item in second.findings]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_selftest_passes(capsys):
    assert utr.main(["--selftest"]) == 0
    assert "selftest OK" in capsys.readouterr().out


def test_list_rules_mentions_every_rule(capsys):
    assert utr.main(["--list-rules"]) == 0
    out = capsys.readouterr().out
    for rule_id in utr.RULE_IDS:
        assert rule_id in out


def test_json_output_shape(capsys):
    path = str(FIXTURES / "violations.md")
    code = utr.main(["--json", path])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["mode"] == STRICT
    assert payload["baseline"] == 0
    assert payload["count"] == len(payload["violations"])
    assert payload["hard_count"] + payload["advisory_count"] == payload["count"]
    assert payload["violations"]
    for item in payload["violations"]:
        assert set(item) == {"file", "line", "col", "rule", "level", "match",
                              "message", "suggestion"}
        assert item["level"] in {"hard", "advisory"}


def test_baseline_changes_the_exit_code(capsys):
    path = str(FIXTURES / "violations.md")
    assert utr.main([path]) == 1
    capsys.readouterr()
    assert utr.main(["--baseline", "100", path]) == 0


def test_disable_removes_a_rule(capsys):
    path = str(FIXTURES / "violations.md")
    assert utr.main(["--disable", "semicolon", path]) == 1
    out = capsys.readouterr().out
    assert " semicolon:" not in out


def test_unknown_rule_is_a_usage_error():
    with pytest.raises(SystemExit) as error:
        utr.main(["--disable", "nosuchrule"])
    assert error.value.code == 2


def test_missing_file_is_a_usage_error(capsys):
    assert utr.main(["nosuchfile.md"]) == 2


def test_stdin_is_used_without_files(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _FakeStdin("Проверьте систему; потом тест."))
    assert utr.main(["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert [item["file"] for item in payload["violations"]] == ["<stdin>"]


def test_normal_mode_relaxes_the_length_rule(monkeypatch, capsys):
    text = words(21) + "\n"
    monkeypatch.setattr("sys.stdin", _FakeStdin(text))
    assert utr.main(["--json"]) == 1
    strict_payload = json.loads(capsys.readouterr().out)
    monkeypatch.setattr("sys.stdin", _FakeStdin(text))
    utr.main(["--json", "--mode", NORMAL])
    normal_payload = json.loads(capsys.readouterr().out)
    assert strict_payload["violations"][0]["level"] == "hard"
    assert normal_payload["violations"][0]["level"] == "advisory"


class _FakeStdin:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


# ---------------------------------------------------------------------------
# Repository hygiene
# ---------------------------------------------------------------------------


def test_skill_declares_russian_only_rules():
    text = (ROOT / "SKILL.md").read_text(encoding="utf-8")
    assert "УТР" in text
    assert "STRICT" in text and "NORMAL" in text


def test_reference_documents_every_rule():
    text = (ROOT / "references" / "writing-rules.md").read_text(encoding="utf-8")
    for rule_id in utr.RULE_IDS:
        assert rule_id in text, rule_id


def test_examples_reference_the_linter():
    text = (ROOT / "examples" / "before-after.md").read_text(encoding="utf-8")
    assert text.count("```text") >= 10


def test_edge_case_fixture_reports_dangling_conjunctions():
    text = read_fixture("linter-edge-cases.md")
    findings = findings_of(text, "dangling-conjunction")
    assert [item.line for item in findings] == [7, 8]


# ---------------------------------------------------------------------------
# Facts are preserved by the examples (SKILL.md rules 26 and 27)
# ---------------------------------------------------------------------------

EXAMPLE_HEADING = re.compile(r"^## \d+\. (.+)$", re.M)
BEFORE_BLOCK = re.compile(r"\*\*Было\*\*\s+```text\n(.*?)```", re.S)
AFTER_BLOCK = re.compile(r"\*\*Стало\*\*\s+```text\n(.*?)```", re.S)
FACT_TOKEN = re.compile(r"`[^`\n]+`|[A-Za-z_][\w./-]*|\d[\d.,]*")

#: Structural tokens a rewrite is allowed to introduce.
STRUCTURAL = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"}


def _facts(text: str) -> set[str]:
    """Collect comparable facts: numbers, code, Latin names, hyphenated names."""
    found = set()
    for token in FACT_TOKEN.findall(text):
        bare = token.strip("`").strip(".,;:!?()")
        if not bare:
            continue
        if bare in STRUCTURAL:
            continue
        found.add(bare.lower())
    return found


def _example_pairs() -> list[tuple[str, str, str]]:
    text = (ROOT / "examples" / "before-after.md").read_text(encoding="utf-8")
    names = EXAMPLE_HEADING.findall(text)
    bodies = EXAMPLE_HEADING.split(text)[2::2]
    pairs = []
    for name, body in zip(names, bodies):
        before = BEFORE_BLOCK.search(body)
        after = AFTER_BLOCK.search(body)
        if before and after:
            pairs.append((name, before.group(1), after.group(1)))
    return pairs


def test_examples_have_all_expected_pairs():
    pairs = _example_pairs()
    assert len(pairs) == 18, len(pairs)


def test_examples_lose_no_fact():
    for name, before, after in _example_pairs():
        lost = _facts(before) - _facts(after)
        assert not lost, "%s loses %s" % (name, sorted(lost))


def test_examples_add_no_fact():
    for name, before, after in _example_pairs():
        added = _facts(after) - _facts(before)
        assert not added, "%s adds %s" % (name, sorted(added))


#: Hedge words. If the source hedges, the rewrite must hedge too (rule 25).
HEDGE = re.compile(r"\b(?:вероятн\w*|возможно|наверн\w*|предположительн\w*|по-видимому)\b", re.I)


def test_examples_keep_hedges():
    for name, before, after in _example_pairs():
        if HEDGE.search(before):
            assert HEDGE.search(after), "%s drops a hedge" % name


# ---------------------------------------------------------------------------
# The glossary promises what the linter finds
# ---------------------------------------------------------------------------

GLOSSARY = ROOT / "references" / "glossary.md"
GLOSSARY_ROW = re.compile(r"^\| `([^`]+)` \| [^|]+\| `([a-z-]+)` \|$", re.M)


def _glossary_section(title: str) -> str:
    text = GLOSSARY.read_text(encoding="utf-8")
    return text.split(title, 1)[1].split("\n## ", 1)[0]


def test_glossary_entries_are_detected():
    rows = []
    for title in ("## Замена канцеляризмов", "## Замена глагола"):
        rows += GLOSSARY_ROW.findall(_glossary_section(title))
    assert len(rows) >= 30
    for phrase, rule in rows:
        sentence = "Оператор должен %s узла сегодня." % phrase
        assert rule in rules_of(sentence), phrase


def test_glossary_synonyms_match_the_linter():
    groups = {name: members for name, members in utr.SYNONYM_GROUPS}
    section = _glossary_section("## Одно действие")
    rows = re.findall(r"^\| ([а-яё]+) \| ([^|]+) \|$", section, re.M)
    assert rows
    for concept, words in rows:
        members = groups[concept]
        names = [name for name, _, _ in members]
        for word in (item.strip().strip("`") for item in words.split(",")):
            assert word in names, (concept, word)
        for name, pattern, _ in members:
            assert re.search(pattern, name, re.I), (concept, name)


# ---------------------------------------------------------------------------
# Published documents carry no placeholders
# ---------------------------------------------------------------------------


def test_no_placeholders_in_documents():
    for path in ROOT.glob("**/*.md"):
        text = path.read_text(encoding="utf-8")
        assert "your-account" not in text, path


def test_directory_argument_is_expanded(tmp_path, capsys):
    (tmp_path / "a.md").write_text("Проверьте систему; потом тест.\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("Проверьте систему.\n", encoding="utf-8")
    (tmp_path / "skip.py").write_text("x = 1; y = 2\n", encoding="utf-8")
    assert utr.main(["--json", str(tmp_path)]) == 1
    payload = json.loads(capsys.readouterr().out)
    files = {Path(item["file"]).name for item in payload["violations"]}
    assert files == {"a.md"}
