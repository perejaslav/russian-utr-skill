#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deterministic linter for UTR (Russian Simplified Technical Language).

The linter only *analyses* text. It never rewrites it, never writes files,
never opens sockets and never reports telemetry. Every finding is
reproducible: the same input always produces the same output, in the same
order, with the same positions.

What is checked mechanically here is the part of UTR that has a surface
form: punctuation, sentence length, official bureaucratic constructions,
verb + noun pairs, likely passive voice, terminology drift, dangling
conjunctions, vague wording and marketing claims.

What is NOT checked here (it belongs to the model, not to a regex):
ambiguity of pronouns, broken logic, lost meaning, bad information order.
See SKILL.md, section "Checks the model performs".

Usage:
    utr-lint.py FILE [FILE ...]
    echo "текст" | utr-lint.py [--json]
    utr-lint.py --baseline 5 FILE          # pass unless hard findings exceed 5
    utr-lint.py --disable semicolon,bureaucratism FILE
    utr-lint.py --mode normal FILE         # softer thresholds for prose
    utr-lint.py --list-rules
    utr-lint.py --selftest

Exit codes:
    0  no hard findings above the baseline
    1  hard findings exceed the baseline
    2  bad command line

Two levels of severity:
    hard      breaks a rule that changes how reliably the text can be read
    advisory  worth a look, but a competent author may keep it

Optional dependency:
    pymorphy3 adds two rules (participle-clause, noun-chain) and removes
    false positives from passive-voice. Without it the linter falls back to
    a built-in lexicon and regexes. Results are reported either way.

Requires Python 3.11+. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Iterator, Sequence

__version__ = "1.0.0"

STRICT = "strict"
NORMAL = "normal"
MODES = (STRICT, NORMAL)

HARD = "hard"
ADVISORY = "advisory"

#: Target length for an instruction. Above it, the rule is broken.
MAX_WORDS_INSTRUCTION = 20
#: Absolute cap. Above it, even prose is too dense to parse at a glance.
MAX_WORDS_HARD = 25
#: A noun chain of this many words in a row is reported (morphology only).
MAX_NOUN_CHAIN = 4
#: Conjunction count that turns a sentence from "long" into "complex".
COMPLEX_SENTENCE_ADVISORY = 5
COMPLEX_SENTENCE_HARD = 8
#: Action nouns per sentence before nominalisation is reported.
NOMINALIZATION_ADVISORY = 2
NOMINALIZATION_HARD = 3
#: Minimum words in a sentence before its complexity is judged.
COMPLEX_SENTENCE_MIN_WORDS = 10

# Cooperatives and other function words that the morphological analyser may
# read as a gerund or a participle. They are never a clausal construction.
FUNCTION_WORDS = frozenset({
    "хотя", "если", "когда", "пока", "чтобы", "как", "так", "ли", "же",
    "не", "ни", "бы", "то", "за", "на", "до", "из", "по", "за", "у", "к",
    "о", "об", "от", "со", "во", "с", "а", "и", "в",
})

HEDGE_NOTE = (
    "Модальность («может», «вероятно», «возможно») не проверяется: "
    "уверенность автора — это содержание, а не стиль."
)


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Bureaucratic constructions. Each entry is (pattern, replacement hint).
# The hint says what to write instead; None means "no fixed replacement,
# rewrite by hand".
BUREAUCRACY_PHRASES: tuple[tuple[str, str | None], ...] = (
    (r"\bв\s+рамках\b", "для"),
    (r"\bв\s+целях\b", "для"),
    (r"\bс\s+целью\b", "чтобы"),
    (r"\bв\s+связи\s+с\b", "из-за / потому что"),
    (r"\bв\s+этой\s+связи\b", None),
    (r"\bна\s+основане\b", "по"),
    (r"\bв\s+настоящее\s+время\b", "сейчас"),
    (r"\bв\s+настоящий\s+момент\b", "сейчас"),
    (r"\bна\s+сегодняшний\s+день\b", "сейчас"),
    (r"\bна\s+данный\s+момент\b", "сейчас"),
    (r"\bв\s+данном\s+случае\b", "здесь"),
    (r"\bявляет(?:ся|ься)\b", "назовите действие глаголом"),
    (r"\bиме(?:ет|ют)\s+место\b", "напишите, что именно произошло"),
    (r"\bследует\s+отметить\b", None),
    (r"\bнеобходимо\s+отметить\b", None),
    (r"\bважно\s+отметить\b", None),
    (r"\bследует\s+подчеркнуть\b", None),
    (r"\bследует\s+учитывать\b", None),
    (r"\bпредставляется\s+необходимым\b", "нужно"),
    (r"\bпредставляется\s+возможным\b", "можно"),
    (r"\bявляется\s+целесообразным\b", "нужно"),
    (r"\bимеющ(?:ий|ая|ее)\b", None),
    (r"\bявляющ(?:ий|ая|ее)\b", None),
    (r"\bв\s+силу\b", "из-за"),
    (r"\bдо\s+сих\s+пор\b", None),
    (r"\bтаким\s+образом\b", None),
    (r"\bотмечается,\s+что\b", None),
    (r"\bв\s+результате\s+чего\b", None),
    (r"\bв\s+связи\s+с\s+изложенным\b", None),
    (r"\bс\s+учетом\s+того,\s+что\b", None),
    (r"\bне\s+представляется\s+возможным\b", "нельзя"),
    (r"\bданн(?:ый|ая|ое|ые|ого|ом)\b", "назовите предмет прямо"),
)

# Forms that must stay out of the "данн..." rule: the plural «данные» is an
# ordinary noun ("data"), not a bureaucratic adjective.
BUREAUCRACY_EXCLUSIONS: dict[str, str] = {
    "bureaucratism": r"^данные?$",
}

# Verbs that normally should not stand in front of a noun. Each alternative
# is a prefix that covers the whole paradigm of one verb ("производ" covers
# производить, производит, производится).
KANCER_VERB_PATTERNS: tuple[str, ...] = (
    r"произвед\w*",
    r"произвест\w*",
    r"производ\w*",
    r"выполн\w*",
    r"осуществ\w*",
    r"провед\w*",
    r"провес\w*",
    r"провод\w*",
    r"предостав\w*",
    r"обеспеч\w*",
    r"оказа\w*",
    r"оказыва\w*",
    r"приня\w*",
    r"принима\w*",
    r"соверш\w*",
    r"введ\w*",
    r"ввод\w*",
    r"сдела\w*",
)

# Action noun -> plain verb. The noun forms are derived into regexes.
ACTION_NOUN_VERBS: dict[str, str] = {
    "установка": "установить",
    "установление": "установить",
    "настройка": "настроить",
    "проверка": "проверить",
    "тестирование": "протестировать",
    "контроль": "контролировать",
    "мониторинг": "отслеживать",
    "изменение": "изменить",
    "замена": "заменить",
    "редактирование": "изменить",
    "создание": "создать",
    "добавление": "добавить",
    "удаление": "удалить",
    "запуск": "запустить",
    "перезапуск": "перезапустить",
    "остановка": "остановить",
    "открытие": "открыть",
    "закрытие": "закрыть",
    "подключение": "подключить",
    "отключение": "отключить",
    "загрузка": "загрузить",
    "выгрузка": "выгрузить",
    "скачивание": "скачать",
    "копирование": "скопировать",
    "чтение": "прочитать",
    "запись": "записать",
    "сохранение": "сохранить",
    "хранение": "хранить",
    "отправка": "отправить",
    "получение": "получить",
    "извлечение": "извлечь",
    "обработка": "обработать",
    "преобразование": "преобразовать",
    "синхронизация": "синхронизировать",
    "инициализация": "инициализировать",
    "актуализация": "актуализировать",
    "валидация": "валидировать",
    "верификация": "верифицировать",
    "аутентификация": "аутентифицировать",
    "авторизация": "авторизовать",
    "диагностика": "диагностировать",
    "сборка": "собрать",
    "разборка": "разобрать",
    "монтаж": "монтировать",
    "демонтаж": "демонтировать",
    "восстановление": "восстановить",
    "обновление": "обновить",
    "резервирование": "зарезервировать",
    "развертывание": "развернуть",
    "сопровождение": "сопровождать",
    "обслуживание": "обслуживать",
    "логирование": "записать в журнал",
    "трассировка": "трассировать",
    "кэширование": "кэшировать",
    "выявление": "выявить",
    "обнаружение": "обнаружить",
    "определение": "определить",
    "измерение": "измерить",
    "сравнение": "сравнить",
    "вычисление": "вычислить",
    "завершение": "завершить",
    "выполнение": "выполнить",
    "предоставление": "предоставить",
    "обеспечение": "обеспечить",
    "использование": "использовать",
    "применение": "применить",
    "реализация": "реализовать",
    "анализ": "анализировать",
    "поиск": "найти",
    "вызов": "вызвать",
    "блокировка": "заблокировать",
    "разблокировка": "разблокировать",
    "передача": "передать",
    "выдача": "выдать",
    "проведение": "провести",
    "разрешение": "разрешить",
    "подтверждение": "подтвердить",
}

# Nouns that are not action nouns but still read better with a verb.
EXTRA_NOUN_VERBS: dict[str, str] = {
    "решение": "решить",
    "влияние": "повлиять",
    "помощь": "помочь",
    "указание": "указать",
    "эксплуатацию": "запустить",
}

# Words that share a stem with a listed noun but mean something else.
ACTION_NOUN_EXCLUSIONS: dict[str, str] = {
    "анализ": r"анализатор",
    "поиск": r"поисковик",
    "контроль": r"контроллер",
    "мониторинг": r"мониторингов",
    "сборка": r"сборщик",
    "настройка": r"настроение",
}

# Stems that cannot be derived mechanically ("настройка" contains "й").
ACTION_NOUN_STEM_OVERRIDES: dict[str, tuple[str, ...]] = {
    "настройка": ("настро", "настройк"),
}

# Probable passive voice: an auxiliary followed by a past passive participle
# from this list. Prefixes cover the whole paradigm.
PASSIVE_PARTICIPLE_STEMS: tuple[str, ...] = (
    r"установ\w*",
    r"удал\w*",
    r"создан\w*",
    r"собран\w*",
    r"разобран\w*",
    r"получен\w*",
    r"найден\w*",
    r"обнаружен\w*",
    r"выявлен\w*",
    r"добавлен\w*",
    r"измен[её]н\w*",
    r"выполнен\w*",
    r"провед[её]н\w*",
    r"сохран[её]н\w*",
    r"отправлен\w*",
    r"прочитан\w*",
    r"записан\w*",
    r"загружен\w*",
    r"выгружен\w*",
    r"открыт\w*",
    r"закрыт\w*",
    r"включ[её]н\w*",
    r"выключен\w*",
    r"запущен\w*",
    r"остановлен\w*",
    r"перезапущен\w*",
    r"выбран\w*",
    r"указан\w*",
    r"перед[аа]н\w*",
    r"принят\w*",
    r"выдан\w*",
    r"заблокирован\w*",
    r"разблокирован\w*",
    r"разреш[её]н\w*",
    r"запрещ[её]н\w*",
    r"подтвержд[её]н\w*",
    r"отмен[её]н\w*",
    r"обработан\w*",
    r"настроен\w*",
    r"возвращ[её]н\w*",
    r"восстановлен\w*",
    r"обновл[её]н\w*",
    r"синхронизирован\w*",
    r"подключ[её]н\w*",
    r"отключ[её]н\w*",
    r"скопирован\w*",
    r"перемещ[её]н\w*",
    r"разв[её]рнут\w*",
    r"сформирован\w*",
    r"сгенерирован\w*",
    r"примен[её]н\w*",
    r"использован\w*",
    r"пересчитан\w*",
    r"запрошен\w*",
    r"заверш[её]н\w*",
    r"очищен\w*",
    r"зашифрован\w*",
    r"расшифрован\w*",
    r"подписан\w*",
    r"доставлен\w*",
    r"вычислен\w*",
    r"переведен\w*",
)

PASSIVE_AUXILIARIES = r"(?:был|была|было|были|будет|будут|есть|стал|стала|стало|стали)"

# Vague wording: the reader cannot act on it.
VAGUE_PHRASES: tuple[tuple[str, str | None], ...] = (
    (r"\bкак\s+можно\s+(?:скорее|быстрее)\b", "укажите срок"),
    (r"\bв\s+ближайшее\s+время\b", "укажите срок"),
    (r"\bв\s+кратчайшие\s+сроки\b", "укажите срок"),
    (r"\bв\s+ряде\s+случаев\b", "каких именно случаях"),
    (r"\bв\s+некоторых\s+случаях\b", "каких именно случаях"),
    (r"\bв\s+отдельных\s+случаях\b", "каких именно случаях"),
    (r"\bнекоторое\s+время\b", "сколько секунд или минут"),
    (r"\bнекоторый\s+период\b", "сколько секунд или минут"),
    (r"\bкаким-либо\s+образом\b", "как именно"),
    (r"\bкаким\s+бы\s+то\s+ни\s+было\s+образом\b", "как именно"),
    (r"\bлюбым\s+способом\b", "назовите способ"),
    (r"\bкак\s+угодно\b", "назовите способ"),
    (r"\bв\s+разумных\s+пределах\b", "назовите предел"),
    (r"\bдостаточно\s+(?:большой|маленький|длинный|короткий|высокий|низкий"
     r"|быстрый|медленный|широкий|узкий|глубокий|л[её]гкий|т[яё]жёлый"
     r"|сложный|простой|частый|редкий|близкий|дал[её]кий|хороший|плохой"
     r"|слабый|сильный|ж[её]сткий|мягкий|строгий|новый|старый)\b",
     "назовите число"),
    (r"\bсоответствующ(им|его)\s+образом\b", "назовите способ"),
    (r"\bнадлежащ(им|его)\s+образом\b", "назовите способ"),
    (r"\bоптимальным\s+образом\b", "назовите критерий"),
    (r"\bпри\s+необходимости\b", "назовите условие"),
    (r"\bв\s+случае\s+необходимости\b", "назовите условие"),
    (r"\bкак\s+правило\b", "как именно"),
    (r"\bпо\s+возможности\b", "назовите условие"),
    (r"\bнасколько\s+возможно\b", "назовите предел"),
    (r"\bи\s+так\s+далее\b", "перечислите или поставьте многоточие"),
    (r"\bи\s+т\.\s?д\.\b", "перечислите или поставьте многоточие"),
    (r"\bи\s+т\.\s?п\.\b", "перечислите или поставьте многоточие"),
    (r"\bи\s+тому\s+подобное\b", "перечислите"),
    (r"\bработа(?:ет|ют|ать)\s+корректно\b", "назовите ожидаемый результат"),
    (r"\bне\s+работает\s+как\s+следует\b", "назовите ожидаемый результат"),
)

# Marketing adjectives: a quality claim with nothing to measure.
MARKETING_WORDS: tuple[str, ...] = (
    r"уникальн\w*",
    r"инновационн\w*",
    r"революционн\w*",
    r"оптимальн\w*",
    r"идеальн\w*",
    r"лучш\w*",
    r"мощн\w*",
    r"высокопроизводительн\w*",
    r"масштабируем\w*",
    r"интуитивн\w*",
    r"бесшовн\w*",
    r"впечатляющ\w*",
    r"флагманск\w*",
    r"передов\w*",
    r"эффективн\w*",
    r"ультрасовременн\w*",
)

# One term per concept. Each group is a set of alternatives for the same
# action. A group fires only when two different alternatives appear in the
# same file.
SynonymMember = tuple[str, str, str | None]

SYNONYM_GROUPS: tuple[tuple[str, tuple[SynonymMember, ...]], ...] = (
    ("проверка", (
        ("проверить", r"\bпровер\w*\b", r"проверочн"),
        ("валидировать", r"\bвалидир\w*\b", None),
        ("верифицировать", r"\bверифиц\w*\b", None),
        ("контролировать", r"\bконтролир\w*\b", None),
    )),
    ("удаление", (
        ("удалить", r"\bудал\w*\b", None),
        ("стирать", r"\bстира\w*\b", None),
        ("стереть", r"\bстер\w*\b", None),
    )),
    ("запуск", (
        ("запустить", r"\b(?:запуст\w*|запуск\w*)\b", None),
        ("стартовать", r"\bстарт\w*\b", None),
    )),
    ("остановка", (
        ("остановить", r"\bостанов\w*\b", None),
        ("прекратить", r"\bпрекращ\w*\b", None),
    )),
    ("показ", (
        ("показать", r"\b(?:покаж\w*|показ\w*)\b", r"показател|показательн"),
        ("отобразить", r"\bотображ\w*|\bотобра\w*\b", None),
    )),
    ("использование", (
        ("использовать", r"\bиспольз\w*\b", None),
        ("применять", r"\bпримен\w*\b", None),
        ("задействовать", r"\bзадейств\w*\b", None),
    )),
    ("исправление", (
        ("исправить", r"\bисправ\w*\b", None),
        ("устранить", r"\bустран\w*\b", None),
        ("починить", r"\bпочин\w*\b", None),
        ("чинить", r"\bчини\w*\b", None),
    )),
    ("отправка", (
        ("отправить", r"\bотправ\w*\b", None),
        ("передать", r"\bпереда\w*\b", None),
        ("отсылать", r"\bотсыл\w*\b", None),
    )),
    ("получение", (
        ("получить", r"\bполуч\w*\b", None),
        ("извлечь", r"\bизвлеч\w*\b", None),
    )),
    ("изменение", (
        ("изменить", r"\bизмен\w*\b", None),
        ("модифицировать", r"\bмодифиц\w*\b", None),
        ("править", r"\bправк\w*\b", None),
    )),
    ("уведомление", (
        ("сообщить", r"\bсообщ\w*\b", None),
        ("уведомить", r"\bуведом\w*\b", None),
    )),
    ("настройка", (
        ("настроить", r"\bнастро\w*\b", r"настроение"),
        ("конфигурировать", r"\bконфигурир\w*\b", None),
    )),
    ("подключение", (
        ("подключить", r"\bподключ\w*\b", None),
        ("присоединить", r"\bприсоедин\w*\b", None),
    )),
    ("поиск", (
        ("найти", r"\b(?:найд\w*|наход\w*)\b", None),
        ("обнаружить", r"\bобнаруж\w*\b", None),
    )),
)

# Connectives that make a sentence hard to parse.
COMPLEX_SENTENCE_MARKERS = (
    r"и", r"или", r"а", r"но", r"же", r"ли", r"бы", r"то\s+есть", r"а\s+также",
    r"который", r"которая", r"которое", r"которые", r"которых", r"которым",
    r"которого", r"которой", r"если", r"когда", r"пока", r"поскольку", r"хотя",
    r"чтобы", r"в\s+случае", r"после\s+того", r"также", r"следовательно",
    r"поэтому", r"иначе", r"однако", r"потому\s+что", r"несмотря\s+на",
)

# Coordinators that must not end a list item.
DANGLING_CONJUNCTIONS = r"(?:и|или|либо|а|но|а\s+также|и/или)"


# ---------------------------------------------------------------------------
# Morphology (optional)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\s]+")
_WORD_RE = re.compile(r"[^\W\d_][^\W\d_'-]*", re.UNICODE)
_HAS_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


class Morphology:
    """Thin wrapper over pymorphy3 with a hard fallback.

    Every method returns a safe answer when the library is missing or the
    word is unknown, so callers never have to know which backend is active.
    """

    def __init__(self) -> None:
        self.backend = "builtin"
        self._analyzer = None
        self._cache: dict[str, tuple[bool, bool, bool, bool, bool, bool]] = {}
        try:  # pragma: no cover - depends on the environment
            import pymorphy3  # type: ignore
        except Exception:  # pragma: no cover - the documented no-dependency path
            return
        try:  # pragma: no cover
            self._analyzer = pymorphy3.MorphAnalyzer()
            self.backend = "pymorphy3"
        except Exception:  # pragma: no cover
            self._analyzer = None

    @property
    def available(self) -> bool:
        return self._analyzer is not None

    def analyse(self, word: str) -> tuple[bool, bool, bool, bool, bool, bool]:
        """Return (noun, verb, adjective, participle, passive, gerund)."""
        key = word.lower()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._analyse_uncached(key)
        self._cache[key] = result
        return result

    def _analyse_uncached(
        self, word: str
    ) -> tuple[bool, bool, bool, bool, bool, bool]:
        if self._analyzer is None or not _HAS_CYRILLIC.search(word):
            return (False,) * 6
        try:  # pragma: no cover - exercised only with pymorphy3 installed
            candidates = self._analyzer.parse(word)
            tags = [str(candidate.tag).upper() for candidate in candidates]
            if not tags:
                return (False,) * 6
            # The first candidate is the best reading. Homographs such as
            # «в» (preposition) and «в» (a letter) must not be read as a
            # noun just because a weaker reading says so.
            primary = tags[0]
            is_noun = "NOUN" in primary
            is_verb = "VERB" in primary or "INFN" in primary
            is_adjective = "ADJF" in primary or "ADJS" in primary
            # Participles are secondary readings («установленный» is parsed
            # as an adjective first). A noun reading anywhere in the list
            # wins, so «данные» stays a noun.
            is_participle = any("PRTF" in tag for tag in tags) and not any(
                "NOUN" in tag for tag in tags
            )
            is_passive = is_participle and any(
                "PSSV" in tag and "PRTF" in tag for tag in tags
            )
            is_gerund = any("GRND" in tag for tag in tags) and not any(
                "NOUN" in tag for tag in tags
            )
            return (is_noun, is_verb, is_adjective, is_participle, is_passive,
                    is_gerund)
        except Exception:  # pragma: no cover - never break the linter
            return (False,) * 6

    def is_noun(self, word: str) -> bool:
        return self.analyse(_clean_word(word))[0]

    def is_adjective(self, word: str) -> bool:
        return self.analyse(_clean_word(word))[2]

    def is_noun_like(self, word: str) -> bool:
        """True for nouns and for adjectives that qualify a noun.

        "серверного модуля" is one link in a noun chain even though
        "серверного" is parsed as an adjective.
        """
        cleaned = _clean_word(word)
        nouns, _, adjectives, _, _, _ = self.analyse(cleaned)
        return nouns or adjectives

    def is_participle(self, word: str) -> bool:
        cleaned = _clean_word(word)
        if cleaned.lower() in FUNCTION_WORDS:
            return False
        return self.analyse(cleaned)[3]

    def is_passive_participle(self, word: str) -> bool:
        return self.analyse(_clean_word(word))[4]

    def is_gerund(self, word: str) -> bool:
        cleaned = _clean_word(word)
        # A Russian gerund always ends in "-в" or "-вшись". This guard keeps
        # homographs such as «для» and «хотя», which the analyser also lists
        # as gerunds, out of the rule.
        if not cleaned.lower().endswith(("в", "вшись")):
            return False
        if cleaned.lower() in FUNCTION_WORDS:
            return False
        return self.analyse(cleaned)[5]


def _clean_word(token: str) -> str:
    """Strip punctuation from the edges of a token."""
    match = _WORD_RE.search(token)
    return match.group(0) if match else ""


# ---------------------------------------------------------------------------
# Findings and rule registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    file: str
    line: int
    col: int
    rule: str
    level: str
    match: str
    message: str
    suggestion: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "col": self.col,
            "rule": self.rule,
            "level": self.level,
            "match": self.match,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class RuleInfo:
    """Static description of a rule, used by --list-rules and the docs."""

    id: str
    title: str
    description: str
    needs_morphology: bool = False
    strict_level: str = HARD
    normal_level: str = HARD

    def level(self, mode: str) -> str:
        return self.strict_level if mode == STRICT else self.normal_level


RULES: tuple[RuleInfo, ...] = (
    RuleInfo(
        "long-sentence",
        "Длинное предложение",
        f"Больше {MAX_WORDS_INSTRUCTION} слов в инструкции, больше "
        f"{MAX_WORDS_HARD} в описании. Разделите предложение.",
    ),
    RuleInfo(
        "semicolon",
        "Точка с запятой",
        "В УТР точка с запятой не используется. Напишите два предложения.",
    ),
    RuleInfo(
        "bureaucratism",
        "Канцеляризм",
        "Оборот, который можно заменить обычной конструкцией.",
    ),
    RuleInfo(
        "verb-object-noun",
        "Глагол и отглагольное существительное",
        "«произвести установку» → «установить». Глагол называет исполнителя.",
    ),
    RuleInfo(
        "nominalization",
        "Избыточные отглагольные существительные",
        "Действие заменено существительным. Верните глагол.",
        strict_level=HARD,
        normal_level=ADVISORY,
    ),
    RuleInfo(
        "passive-voice",
        "Вероятный пассивный залог",
        "Исполнитель не назван. Назовите его и верните действительный залог.",
        strict_level=ADVISORY,
        normal_level=ADVISORY,
    ),
    RuleInfo(
        "synonym-rotation",
        "Один термин — одно обозначение",
        "Одно действие названо разными словами в одном файле.",
    ),
    RuleInfo(
        "complex-sentence",
        "Слишком сложное предложение",
        "Много союзов и придаточных частей. Разделите предложение.",
        strict_level=HARD,
        normal_level=HARD,
    ),
    RuleInfo(
        "dangling-conjunction",
        "Элемент списка оборван союзом",
        "Элемент списка заканчивается союзом «и», «или», «а», «но».",
    ),
    RuleInfo(
        "vague-formulation",
        "Расплывчатая формулировка",
        "Формулировку нельзя проверить и выполнить. Укажите число, срок или признак.",
    ),
    RuleInfo(
        "marketing-word",
        "Маркетинговое слово",
        "Обещание качества без измеримого признака.",
    ),
    RuleInfo(
        "participle-clause",
        "Причастный или деепричастный оборот",
        "Действие спрятано в причастии. Верните глагол.",
        needs_morphology=True,
        strict_level=ADVISORY,
        normal_level=ADVISORY,
    ),
    RuleInfo(
        "noun-chain",
        "Длинная цепочка существительных",
        f"Не меньше {MAX_NOUN_CHAIN} существительных подряд. Разделите их.",
        needs_morphology=True,
        strict_level=ADVISORY,
        normal_level=ADVISORY,
    ),
)

RULE_INDEX: dict[str, RuleInfo] = {rule.id: rule for rule in RULES}
RULE_IDS: tuple[str, ...] = tuple(rule.id for rule in RULES)


# ---------------------------------------------------------------------------
# Markdown structure
# ---------------------------------------------------------------------------

CODE_FENCE = re.compile(r"^(?:```|~~~)")
FRONTMATTER_FENCE = re.compile(r"^---\s*$")
INLINE_CODE = re.compile(r"`+[^`\n]*`+")
LIST_ITEM_START = re.compile(
    r"^(?P<indent> {0,3})(?P<marker>[-*+]|[0-9]+[.)])(?P<gap> +)(?P<body>.*)$"
)
TABLE_SEPARATOR_CELL = re.compile(r"^:?-{3,}:?$")
BLOCKQUOTE_PREFIX = re.compile(r"^(?P<prefix>\s*(?:>\s?)+)")

# Technical literals. They are masked before any rule runs, so a URL, a
# Windows path, a CLI flag or a JSON blob is never read as Russian prose.
URL_LITERAL = re.compile(r"(?:https?|ftp|ssh|git)://[^\s)\]»\"']+")
MAIL_LITERAL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
WINDOWS_PATH_LITERAL = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s)\]»\"'«]+")
POSIX_PATH_LITERAL = re.compile(
    r"(?:(?<=[\s(\[(‘«])|^)(?:~|\.{1,2})/[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*"
)
CLI_FLAG_LITERAL = re.compile(r"(?:(?<=[\s(\[,])|^)--?[A-Za-z][\w-]*(?:=\S*)?")
ENV_ASSIGNMENT_LITERAL = re.compile(r"(?:(?<=[\s(\[,])|^)[A-Z][A-Z0-9_]{2,}=\S*")
JSON_LITERAL = re.compile(r"[\[{][^\]}\n]*[\]}]")
QUOTED_LITERAL = re.compile(r'"[^"\n]*"')

MASK_TARGETS = (
    INLINE_CODE,
    URL_LITERAL,
    MAIL_LITERAL,
    WINDOWS_PATH_LITERAL,
    POSIX_PATH_LITERAL,
    CLI_FLAG_LITERAL,
    JSON_LITERAL,
    ENV_ASSIGNMENT_LITERAL,
)

ABBREVIATION_DOTS = re.compile(
    r"\b(?:т\.\s?[деп]|и\.\s?о|с\.\s?[мп]|см|рис|др|стр|пп)\.(?=\s|$)",
    re.IGNORECASE,
)
ABBREVIATION_PLACEHOLDER = "\x01"

SENTENCE_TERMINATOR = re.compile(r"[.!?…]+(?=\s|$)")
SENTENCE_END = re.compile(r"[.!?…]")
CHAIN_BREAK = re.compile(r"[,:;]")
EMPTY_WORD = re.compile(r"[^\W\d_]", re.UNICODE)


def _blank(match: re.Match[str]) -> str:
    return " " * len(match.group(0))


def _mask_quoted(match: re.Match[str]) -> str:
    """Mask a straight-quoted string only when it is not Russian prose."""
    body = match.group(0)
    return _blank(match) if not _HAS_CYRILLIC.search(body) else body


def mask_technical_literals(text: str) -> str:
    """Replace code, URLs, paths, flags and JSON with spaces.

    Lengths are preserved, so every column reported later still points at the
    right place in the original line.
    """
    masked = INLINE_CODE.sub(_blank, text)
    for pattern in MASK_TARGETS[1:]:
        masked = pattern.sub(_blank, masked)
    masked = QUOTED_LITERAL.sub(_mask_quoted, masked)
    return masked


def _protect_abbreviations(text: str) -> str:
    """Hide abbreviation dots so they do not look like sentence ends."""
    return ABBREVIATION_DOTS.sub(
        lambda match: match.group(0)[:-1] + ABBREVIATION_PLACEHOLDER,
        text,
    )


def _restore_abbreviations(text: str) -> str:
    return text.replace(ABBREVIATION_PLACEHOLDER, ".")


def strip_blockquote(line: str) -> tuple[int, str]:
    """Return (column offset, content) with blockquote markers removed."""
    match = BLOCKQUOTE_PREFIX.match(line)
    if match is None:
        return 0, line
    return len(match.group("prefix")), line[match.end():]


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def split_table_row(line: str) -> list[tuple[str, int]] | None:
    """Return trimmed table cells with their zero-based source columns.

    A pipe must separate at least two cells. Escaped pipes stay inside their
    cell. Only the ordinary Markdown table shape is implemented, which is
    enough to tell a table apart from prose that happens to contain a pipe.
    """
    left = _leading_spaces(line)
    right = len(line.rstrip())
    content = line[left:right]
    if "|" not in content:
        return None
    if content.startswith("|"):
        content = content[1:]
        left += 1
    if content.endswith("|"):
        content = content[:-1]
    raw_cells = re.split(r"(?<!\\)\|", content)
    if len(raw_cells) < 2:
        return None

    cells: list[tuple[str, int]] = []
    column = left
    for raw_cell in raw_cells:
        leading = len(raw_cell) - len(raw_cell.lstrip())
        cells.append((raw_cell.strip(), column + leading))
        column += len(raw_cell) + 1
    return cells


def markdown_table_cells(lines: Sequence[str]) -> dict[int, list[tuple[str, int]]]:
    """Map ordinary Markdown table rows to their prose cells.

    The separator row anchors detection, so prose that merely contains a
    pipe is not treated as a table. Both the leading-pipe and the
    no-leading-pipe styles are accepted.
    """
    table_cells: dict[int, list[tuple[str, int]]] = {}
    index = 1
    while index < len(lines):
        separator = split_table_row(lines[index])
        header = split_table_row(lines[index - 1])
        if (
            not separator
            or not header
            or len(separator) != len(header)
            or not all(
                TABLE_SEPARATOR_CELL.fullmatch(cell) for cell, _ in separator
            )
        ):
            index += 1
            continue
        table_cells[index - 1] = header
        table_cells[index] = []
        index += 1
        while index < len(lines):
            row = split_table_row(lines[index])
            if not row or len(row) != len(separator):
                break
            table_cells[index] = row
            index += 1
    return table_cells


def iter_prose_units(lines: Sequence[str]) -> Iterator[tuple[int, int, str]]:
    """Yield (line number, zero-based source column, masked prose).

    Fenced code and YAML front matter are skipped. Table rows are split into
    cells so table syntax is never read as prose. Blockquote markers are
    removed and reported as a column offset.
    """
    table_cells = markdown_table_cells(lines)
    in_fence = False
    in_frontmatter = bool(lines) and FRONTMATTER_FENCE.match(lines[0].strip())
    for index, raw_line in enumerate(lines):
        if in_frontmatter:
            if index and FRONTMATTER_FENCE.match(raw_line.strip()):
                in_frontmatter = False
            continue
        if CODE_FENCE.match(raw_line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if index in table_cells:
            for cell, column in table_cells[index]:
                offset, content = strip_blockquote(cell)
                yield index + 1, column + offset, mask_technical_literals(content)
            continue
        offset, content = strip_blockquote(raw_line)
        yield index + 1, offset, mask_technical_literals(content)


def count_words(text: str) -> int:
    """Count tokens that contain at least one letter or digit."""
    return sum(1 for token in text.split() if EMPTY_WORD.search(token))


def iter_sentences(text: str) -> Iterator[tuple[str, int]]:
    """Yield (sentence, start column) for one prose unit."""
    protected = _protect_abbreviations(text)
    start = 0
    for match in SENTENCE_TERMINATOR.finditer(protected):
        end = match.end()
        chunk = protected[start:end]
        if count_words(chunk):
            yield _restore_abbreviations(chunk), start
        start = end
        while start < len(protected) and protected[start].isspace():
            start += 1
    tail = protected[start:]
    if count_words(tail):
        yield _restore_abbreviations(tail), start


# ---------------------------------------------------------------------------
# Derived lexicons
# ---------------------------------------------------------------------------


def _noun_stem(noun: str) -> str:
    if noun.endswith(("ие", "ние", "ание", "ение", "ование")):
        return noun[:-2]
    if noun.endswith(("а", "я")):
        return noun[:-1]
    return noun


def _noun_stems(noun: str) -> tuple[str, ...]:
    return ACTION_NOUN_STEM_OVERRIDES.get(noun, (_noun_stem(noun),))


def _noun_alternation(noun: str) -> str:
    """One group per stem so word boundaries apply to every alternative."""
    return "|".join(
        "(?:%s\\w*)" % re.escape(stem) for stem in _noun_stems(noun)
    )


def _all_noun_alternation() -> str:
    return "|".join(
        _noun_alternation(noun)
        for noun in {**ACTION_NOUN_VERBS, **EXTRA_NOUN_VERBS}
    )


ACTION_NOUN_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = tuple(
    (
        noun,
        re.compile(r"\b" + _noun_alternation(noun) + r"\b", re.IGNORECASE),
        verb,
    )
    for noun, verb in {**ACTION_NOUN_VERBS, **EXTRA_NOUN_VERBS}.items()
)

ACTION_NOUN_EXCLUSION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in ACTION_NOUN_EXCLUSIONS.values()
)

VERB_OBJECT_NOUN = re.compile(
    r"\b(?:" + "|".join(KANCER_VERB_PATTERNS) + r")\s+"
    r"(?:в\s+|во\s+|на\s+)?"
    r"\b(?:" + _all_noun_alternation() + r")\b",
    re.IGNORECASE,
)

PASSIVE_AUXILIARY = re.compile(
    r"\b" + PASSIVE_AUXILIARIES + r"\s+"
    r"(?:" + "|".join(PASSIVE_PARTICIPLE_STEMS) + r")\b",
    re.IGNORECASE,
)

BUREAUCRACY_GROUP = re.compile(
    "|".join(pattern for pattern, _ in BUREAUCRACY_PHRASES), re.IGNORECASE
)
BUREAUCRACY_HINTS: dict[str, str] = {}
for _pattern, _hint in BUREAUCRACY_PHRASES:
    BUREAUCRACY_HINTS.setdefault(_pattern, _hint or "")

VAGUE_GROUP = re.compile(
    "|".join(pattern for pattern, _ in VAGUE_PHRASES), re.IGNORECASE
)
VAGUE_HINTS: dict[str, str] = {}
for _pattern, _hint in VAGUE_PHRASES:
    VAGUE_HINTS.setdefault(_pattern, _hint or "")

MARKETING_GROUP = re.compile(
    r"\b(?:" + "|".join(MARKETING_WORDS) + r")", re.IGNORECASE
)

COMPLEX_MARKER_GROUP = re.compile(
    r"\b(?:" + "|".join(COMPLEX_SENTENCE_MARKERS) + r")\b", re.IGNORECASE
)

DANGLING_CONJUNCTION = re.compile(
    r"\b" + DANGLING_CONJUNCTIONS + r"\s*,?\s*$", re.IGNORECASE
)

SYNONYM_MEMBERS: tuple[tuple[str, tuple[tuple[str, re.Pattern[str],
                                                    re.Pattern[str] | None], ...]],
                       ...] = tuple(
    (
        group,
        tuple(
            (label, re.compile(pattern, re.IGNORECASE),
             re.compile(exclude, re.IGNORECASE) if exclude else None)
            for label, pattern, exclude in members
        ),
    )
    for group, members in SYNONYM_GROUPS
)


def _noun_for_surface(surface: str) -> tuple[str, str] | None:
    """Map an inflected noun to (dictionary form, plain verb)."""
    lowered = surface.lower()
    for noun, pattern, verb in ACTION_NOUN_PATTERNS:
        if not pattern.fullmatch(surface) and not pattern.match(surface):
            continue
        for exclusion in ACTION_NOUN_EXCLUSION_PATTERNS:
            if exclusion.fullmatch(lowered):
                return None
        return noun, verb
    return None


def _excluded_by(rule_id: str, matched: str) -> bool:
    pattern = BUREAUCRACY_EXCLUSIONS.get(rule_id)
    if not pattern:
        return False
    return bool(re.fullmatch(pattern, matched.lower(), re.IGNORECASE))


def _hint_for(hints: dict[str, str], matched: str) -> str | None:
    lowered = matched.lower()
    best = ""
    best_key = ""
    for pattern, hint in hints.items():
        if re.search(pattern, matched, re.IGNORECASE) and len(pattern) > len(best_key):
            best, best_key = hint, pattern
    if not best_key:
        return None
    return best or None


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _level(rule_id: str, mode: str) -> str:
    return RULE_INDEX[rule_id].level(mode)


def _finding(
    filename: str,
    line: int,
    col: int,
    rule_id: str,
    mode: str,
    matched: str,
    message: str,
    suggestion: str | None = None,
    level: str | None = None,
) -> Finding:
    return Finding(
        file=filename,
        line=line,
        col=col,
        rule=rule_id,
        level=level or _level(rule_id, mode),
        match=matched,
        message=message,
        suggestion=suggestion,
    )


def check_long_sentence(
    sentence: str, start: int, line: int, col: int, filename: str, mode: str
) -> list[Finding]:
    words = count_words(sentence)
    if words <= MAX_WORDS_INSTRUCTION:
        return []
    if words > MAX_WORDS_HARD:
        level = HARD
        message = (f"Предложение из {words} слов превышает жёсткий предел "
                   f"{MAX_WORDS_HARD} слов. Разделите его на два.")
    elif mode == STRICT:
        level = HARD
        message = (f"Предложение из {words} слов. Ориентир для инструкций — "
                   f"{MAX_WORDS_INSTRUCTION} слов.")
    else:
        level = ADVISORY
        message = (f"Предложение из {words} слов. Ориентир — "
                   f"{MAX_WORDS_INSTRUCTION} слов.")
    return [_finding(filename, line, col + start + 1, "long-sentence", mode,
                     f"{words} слов", message, "Разделите предложение", level)]


def check_semicolon(
    unit: str, line: int, col: int, filename: str, mode: str
) -> list[Finding]:
    return [
        _finding(filename, line, col + match.start() + 1, "semicolon", mode,
                 match.group(0),
                 "В УТР точка с запятой не используется.",
                 "Напишите два предложения")
        for match in re.finditer(";", unit)
    ]


def check_bureaucratism(
    unit: str, line: int, col: int, filename: str, mode: str
) -> list[Finding]:
    findings = []
    for match in BUREAUCRACY_GROUP.finditer(unit):
        if _excluded_by("bureaucratism", match.group(0)):
            continue
        hint = _hint_for(BUREAUCRACY_HINTS, match.group(0))
        message = "Канцелярский оборот: «%s»." % match.group(0)
        findings.append(_finding(filename, line, col + match.start() + 1,
                                 "bureaucratism", mode, match.group(0), message,
                                 hint or "Перепишите обычной конструкцией"))
    return findings


def check_vague_formulation(
    unit: str, line: int, col: int, filename: str, mode: str
) -> list[Finding]:
    findings = []
    for match in VAGUE_GROUP.finditer(unit):
        hint = _hint_for(VAGUE_HINTS, match.group(0))
        findings.append(_finding(
            filename, line, col + match.start() + 1, "vague-formulation", mode,
            match.group(0),
            "Расплывчатая формулировка: «%s»." % match.group(0),
            hint or "Укажите число, срок или признак",
        ))
    return findings


def check_marketing_word(
    unit: str, line: int, col: int, filename: str, mode: str
) -> list[Finding]:
    return [
        _finding(filename, line, col + match.start() + 1, "marketing-word", mode,
                 match.group(0),
                 "Маркетинговое слово «%s» без измеримого признака."
                 % match.group(0),
                 "Замените измеримым значением или удалите")
        for match in MARKETING_GROUP.finditer(unit)
    ]


def check_verb_object_noun(
    unit: str, line: int, col: int, filename: str, mode: str
) -> list[Finding]:
    findings = []
    for match in VERB_OBJECT_NOUN.finditer(unit):
        noun_surface = match.group(0).split()[-1]
        resolved = _noun_for_surface(noun_surface)
        if resolved is None:
            continue
        noun, verb = resolved
        findings.append(_finding(
            filename, line, col + match.start() + 1, "verb-object-noun", mode,
            match.group(0),
            "Канцелярская пара «%s». Действие называет глагол." % match.group(0),
            "%s → %s" % (match.group(0), verb),
        ))
    return findings


def check_nominalization(
    sentence: str, start: int, line: int, col: int, filename: str, mode: str,
    morphology: Morphology,
) -> list[Finding]:
    hits: list[tuple[int, str]] = []
    for token_match in _TOKEN_RE.finditer(sentence):
        surface = token_match.group(0)
        noun = _clean_word(surface)
        if not noun:
            continue
        resolved = _noun_for_surface(noun)
        if resolved is None:
            continue
        if morphology.available and not morphology.is_noun(noun):
            continue
        hits.append((token_match.start(), resolved[0]))

    if not hits:
        return []
    # A sentence may open with its topic as a noun ("Обработка запросов...").
    # That is a normal Russian label, so the first token is not counted.
    counted = hits[1:] if hits[0][0] == 0 else hits
    total = len(counted)
    if total < NOMINALIZATION_ADVISORY:
        return []
    level = HARD if total >= NOMINALIZATION_HARD else ADVISORY
    first_offset, first_noun = counted[0]
    message = ("Отглагольных существительных в предложении: %d. "
               "Замените их глаголами." % total)
    if mode == NORMAL and level == HARD:
        level = ADVISORY
    return [_finding(filename, line, col + start + first_offset + 1,
                     "nominalization", mode, first_noun, message,
                     "Например: «%s» → «%s»" % (first_noun,
                     _verb_for(first_noun)), level)]


def _verb_for(noun: str) -> str:
    for dictionary_form, _, verb in ACTION_NOUN_PATTERNS:
        if dictionary_form == noun:
            return verb
    return "назовите действие глаголом"


def check_passive_voice(
    sentence: str, start: int, line: int, col: int, filename: str, mode: str,
    morphology: Morphology,
) -> list[Finding]:
    findings = []
    seen: set[int] = set()
    for match in PASSIVE_AUXILIARY.finditer(sentence):
        seen.add(match.start())
        findings.append(_finding(
            filename, line, col + start + match.start() + 1, "passive-voice",
            mode, match.group(0),
            "Вероятный пассивный залог. Назовите исполнителя.",
            "Напишите: кто выполняет действие",
        ))
    if morphology.available:
        for token_match in _TOKEN_RE.finditer(sentence):
            offset = token_match.start()
            if offset in seen:
                continue
            word = _clean_word(token_match.group(0))
            if not word or not morphology.is_passive_participle(word):
                continue
            findings.append(_finding(
                filename, line, col + start + offset + 1, "passive-voice", mode,
                token_match.group(0),
                "Отглагольное причастие. Назовите исполнителя.",
                "Напишите: кто выполняет действие",
            ))
    return findings


def check_complex_sentence(
    sentence: str, start: int, line: int, col: int, filename: str, mode: str
) -> list[Finding]:
    if count_words(sentence) < COMPLEX_SENTENCE_MIN_WORDS:
        return []
    markers = COMPLEX_MARKER_GROUP.findall(sentence)
    total = len(markers)
    if total < COMPLEX_SENTENCE_ADVISORY:
        return []
    level = HARD if total >= COMPLEX_SENTENCE_HARD else ADVISORY
    return [_finding(
        filename, line, col + start + 1, "complex-sentence", mode,
        "%d союзов" % total,
        "Союзов и придаточных частей: %d. Разделите предложение." % total,
        "Оставьте одну мысль в одном предложении", level,
    )]


def check_participle_clause(
    sentence: str, start: int, line: int, col: int, filename: str, mode: str,
    morphology: Morphology,
) -> list[Finding]:
    if not morphology.available:
        return []
    for token_match in _TOKEN_RE.finditer(sentence):
        word = _clean_word(token_match.group(0))
        if not word:
            continue
        if morphology.is_gerund(word):
            return [_finding(
                filename, line, col + start + token_match.start() + 1,
                "participle-clause", mode, token_match.group(0),
                "Деепричастный оборот. Верните действие в глагол.",
                "Напишите отдельное предложение",
            )]
        if morphology.is_participle(word):
            return [_finding(
                filename, line, col + start + token_match.start() + 1,
                "participle-clause", mode, token_match.group(0),
                "Причастный оборот. Верните действие в глагол.",
                "Напишите отдельное предложение",
            )]
    return []


def check_noun_chain(
    unit: str, line: int, col: int, filename: str, mode: str,
    morphology: Morphology,
) -> list[Finding]:
    if not morphology.available:
        return []
    run: list[tuple[int, str]] = []
    pending: list[tuple[int, str]] = []
    findings: list[Finding] = []

    def flush() -> None:
        nonlocal run
        if len(run) >= MAX_NOUN_CHAIN:
            text = " ".join(word for _, word in run)
            findings.append(_finding(
                filename, line, col + run[0][0] + 1, "noun-chain", mode, text,
                "Существительных подряд: %d. Разделите цепочку." % len(run),
                "Добавьте глагол или предлог между словами",
            ))
        run = []

    for token_match in _TOKEN_RE.finditer(unit):
        token = token_match.group(0)
        word = _clean_word(token)
        # A chain means words standing next to each other. A colon or a comma
        # starts a list, not a chain.
        separated = bool(CHAIN_BREAK.search(token))
        ends_sentence = bool(SENTENCE_END.search(token))
        if not separated and word and len(word) >= 3 and morphology.is_noun(word):
            if run:
                run.extend(pending)
            pending = []
            run.append((token_match.start(), token))
        elif not separated and word and len(word) >= 3 \
                and morphology.is_adjective(word):
            # An adjective joins the chain only when a noun follows it.
            pending.append((token_match.start(), token))
        else:
            flush()
            pending = []
        if ends_sentence:
            flush()
            pending = []
    flush()
    return findings


def check_synonym_rotation(
    units: Sequence[tuple[int, int, str]], filename: str, mode: str
) -> list[Finding]:
    """One term per concept, scoped to a single file."""
    findings = []
    for group, members in SYNONYM_MEMBERS:
        present: list[tuple[int, int, str, str]] = []
        for line, column, unit in units:
            for label, pattern, exclusion in members:
                for match in pattern.finditer(unit):
                    if exclusion is not None and exclusion.search(match.group(0)):
                        continue
                    present.append((line, column + match.start() + 1,
                                    match.group(0), label))
                    break
            if present:
                break
        if len(present) < 2:
            continue
        present.sort()
        keeper = present[0]
        for line, column, matched, label in present[1:]:
            findings.append(_finding(
                filename, line, column, "synonym-rotation", mode, matched,
                "«%s» и «%s» обозначают одно действие." % (label, keeper[3]),
                "Оставьте «%s» и используйте его везде" % keeper[3],
            ))
    return findings


def _list_line_content(line: str) -> str:
    """Readable text of a list line, used to tell content from blank space.

    Inline code becomes a neutral word so an item that ends with a code span
    is not mistaken for an item that ends with a coordinator.
    """
    return mask_technical_literals(INLINE_CODE.sub(" КОД ", line)).strip()


def _list_line_mask(line: str) -> str:
    """Length-preserving mask, used to locate a coordinator by column."""
    return mask_technical_literals(line)


def _is_list_continuation(line: str, content_indent: int) -> bool:
    if not line.strip():
        return True
    if LIST_ITEM_START.match(line):
        return False
    return _leading_spaces(line) >= content_indent


def check_dangling_conjunction(
    lines: Sequence[str], filename: str, mode: str
) -> list[Finding]:
    """A list item must not end with a coordinator.

    Only plain Markdown list shapes are recognised: markers at the start of
    the line with up to three leading spaces, followed by spaces. Fenced
    code and blockquoted lines are skipped.
    """
    findings = []
    in_fence = False
    in_frontmatter = bool(lines) and FRONTMATTER_FENCE.match(lines[0].strip())
    index = 0
    while index < len(lines):
        line = lines[index]
        if in_frontmatter:
            if index and FRONTMATTER_FENCE.match(line.strip()):
                in_frontmatter = False
            index += 1
            continue
        if CODE_FENCE.match(line.strip()):
            in_fence = not in_fence
            index += 1
            continue
        if in_fence or BLOCKQUOTE_PREFIX.match(line):
            index += 1
            continue

        start = LIST_ITEM_START.match(line)
        if not start:
            index += 1
            continue
        content_indent = (
            len(start.group("indent"))
            + len(start.group("marker"))
            + len(start.group("gap"))
        )
        item_lines = [(index, start.group("body"))]
        next_index = index + 1
        item_fence = False
        while next_index < len(lines):
            candidate = lines[next_index]
            if CODE_FENCE.match(candidate.strip()):
                item_fence = not item_fence
                next_index += 1
                continue
            if item_fence or not _is_list_continuation(candidate, content_indent):
                break
            item_lines.append((next_index, candidate))
            next_index += 1

        meaningful = [
            (line_index, item_line)
            for line_index, item_line in item_lines
            if _list_line_content(item_line)
        ]
        if meaningful:
            end_index, end_line = meaningful[-1]
            # The decision uses the readable form (inline code becomes a
            # neutral word), so an item ending with a code span is not read
            # as an item ending with a coordinator. The column is then taken
            # from the length-preserving mask of the same line.
            if DANGLING_CONJUNCTION.search(_list_line_content(end_line)):
                conjunction = DANGLING_CONJUNCTION.search(
                    _list_line_mask(end_line))
                if conjunction:
                    findings.append(_finding(
                        filename, end_index + 1, conjunction.start() + 1,
                        "dangling-conjunction", mode, conjunction.group(0).strip(),
                        "Элемент списка заканчивается союзом.",
                        "Закончите элемент или соедините его со следующим",
                    ))
        index = next_index
    return findings


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


@dataclass
class Analysis:
    findings: list[Finding] = field(default_factory=list)
    words: int = 0


def analyze(text: str, filename: str = "<stdin>", mode: str = STRICT,
            morphology: Morphology | None = None) -> Analysis:
    """Run every enabled rule over one text."""
    if mode not in MODES:
        raise ValueError("unknown mode: %r" % mode)
    analyser = morphology if morphology is not None else Morphology()

    text = text.replace("\ufeff", "")
    lines = text.splitlines()
    units = list(iter_prose_units(lines))
    analysis = Analysis()
    analysis.words = sum(count_words(unit) for _, _, unit in units)

    for line, column, unit in units:
        analysis.findings.extend(check_semicolon(unit, line, column, filename, mode))
        analysis.findings.extend(check_bureaucratism(unit, line, column, filename, mode))
        analysis.findings.extend(check_vague_formulation(unit, line, column,
                                                        filename, mode))
        analysis.findings.extend(check_marketing_word(unit, line, column,
                                                      filename, mode))
        analysis.findings.extend(check_verb_object_noun(unit, line, column,
                                                       filename, mode))
        analysis.findings.extend(check_noun_chain(unit, line, column, filename,
                                                  mode, analyser))
        for sentence, start in iter_sentences(unit):
            analysis.findings.extend(
                check_long_sentence(sentence, start, line, column, filename, mode)
            )
            analysis.findings.extend(
                check_nominalization(sentence, start, line, column, filename, mode,
                                     analyser)
            )
            analysis.findings.extend(
                check_passive_voice(sentence, start, line, column, filename, mode,
                                    analyser)
            )
            analysis.findings.extend(
                check_complex_sentence(sentence, start, line, column, filename, mode)
            )
            analysis.findings.extend(
                check_participle_clause(sentence, start, line, column, filename,
                                        mode, analyser)
            )

    analysis.findings.extend(check_synonym_rotation(units, filename, mode))
    analysis.findings.extend(check_dangling_conjunction(lines, filename, mode))
    analysis.findings.sort(
        key=lambda item: (item.line, item.col, item.rule, item.match)
    )
    return analysis


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def filter_findings(
    findings: Sequence[Finding], disabled: Sequence[str]
) -> list[Finding]:
    blocked = set(disabled)
    return [item for item in findings if item.rule not in blocked]


def render_text(
    findings: Sequence[Finding], analysis: Analysis, mode: str,
    baseline: int, disabled: Sequence[str], morphology: Morphology,
) -> str:
    hard_count = sum(1 for item in findings if item.level == HARD)
    advisory_count = sum(1 for item in findings if item.level == ADVISORY)
    rate = (round(len(findings) * 100 / analysis.words, 1)
            if analysis.words else 0.0)
    lines = []
    for item in findings:
        location = "%s:%d:%d" % (item.file, item.line, item.col)
        lines.append("%s %s [%s]: %s [%s]" % (
            location, item.rule, item.level, item.message, item.match))
        if item.suggestion:
            lines.append("    → %s" % item.suggestion)
    lines.append("")
    lines.append(
        "Найдено: %d (жёстких %d, рекомендаций %d, baseline %d)"
        % (len(findings), hard_count, advisory_count, baseline)
    )
    lines.append("Слов: %d (%s на 100 слов)" % (analysis.words, rate))
    lines.append("Режим: %s. Морфология: %s." % (mode, morphology.backend))
    if disabled:
        lines.append("Отключено: %s." % ", ".join(disabled))
    if not morphology.available:
        lines.append(
            "Правила participle-clause и noun-chain доступны только с pymorphy3."
        )
    lines.append(HEDGE_NOTE)
    return "\n".join(lines)


def render_json(
    findings: Sequence[Finding], analysis: Analysis, mode: str, baseline: int,
    disabled: Sequence[str], morphology: Morphology,
) -> str:
    hard_count = sum(1 for item in findings if item.level == HARD)
    payload = {
        "version": __version__,
        "mode": mode,
        "baseline": baseline,
        "disabled": list(disabled),
        "morphology": morphology.backend,
        "count": len(findings),
        "hard_count": hard_count,
        "advisory_count": len(findings) - hard_count,
        "words": analysis.words,
        "per_100_words": (round(len(findings) * 100 / analysis.words, 1)
                          if analysis.words else 0.0),
        "violations": [item.to_dict() for item in findings],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_rule_list(mode: str) -> str:
    lines = ["Правила линтера УТР (режим %s):" % mode, ""]
    for rule in RULES:
        lines.append("%-20s %-9s %s%s" % (
            rule.id,
            rule.level(mode),
            rule.title,
            " [нужна pymorphy3]" if rule.needs_morphology else "",
        ))
        lines.append("%-20s %s" % ("", rule.description))
    lines.append("")
    lines.append(HEDGE_NOTE)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _rules_of(text: str, mode: str = STRICT) -> set[str]:
    return {item.rule for item in analyze(text, mode=mode).findings}


def _findings_of(text: str, rule: str, mode: str = STRICT) -> list[Finding]:
    return [item for item in analyze(text, mode=mode).findings
            if item.rule == rule]


def selftest() -> int:
    checks = 0

    def expect(condition: bool, description: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError("selftest failed: " + description)

    # 1. Every rule fires on a known sample.
    cases = (
        ("Проверьте систему; потом запустите тесты.",
         "semicolon"),
        ("Оператор в целях контроля осуществляет проверку состояния узла "
         "перед началом работы оборудования.", "bureaucratism"),
        ("Произвести установку пакета.", "verb-object-noun"),
        ("Выполните установку и проверку соединения.", "nominalization"),
        ("Файл был удалён сервером.", "passive-voice"),
        ("Проверьте конфигурацию. Затем валидируйте выходные данные.",
         "synonym-rotation"),
        ("Сервер возвращает ошибку, если ключ не найден, и клиент должен "
         "повторить запрос, когда сеть снова станет доступна, потому что "
         "очередь не была обработана, а ответ не пришёл вовсе.", "complex-sentence"),
        ("- Установите целевой объект и\n", "dangling-conjunction"),
        ("Сделайте это в кратчайшие сроки.", "vague-formulation"),
        ("Модуль уникальный и инновационный.", "marketing-word"),
    )
    for text, rule in cases:
        expect(rule in _rules_of(text), "rule %s must fire" % rule)

    expect(
        "nominalization" in _rules_of(
            "Обработка запросов, проверка ответов и сохранение результатов "
            "выполняются в фоне."
        ),
        "nominalization must fire on a nominalised sentence",
    )
    heavy = _findings_of(
        "Обработка запросов, проверка ответов, сохранение результатов и "
        "синхронизация кэша выполняются в фоне.", "nominalization")
    expect(len(heavy) == 1 and heavy[0].level == HARD,
           "three action nouns make a hard finding")
    expect(_findings_of(
        "Обработка запросов, проверка ответов и сохранение результатов "
        "выполняются в фоне.", "nominalization")[0].level == ADVISORY,
        "two action nouns stay advisory")
    topic = ("Настройка сервера, проверка журнала и сохранение конфигурации "
             "завершены.")
    topic_findings = _findings_of(topic, "nominalization")
    expect(len(topic_findings) == 1 and topic_findings[0].match == "проверка",
           "a leading topic noun is not counted again")
    expect("nominalization" not in _rules_of(
        "Проверка конфигурации занимает 5 мс."
    ), "a single action noun is not nominalisation")

    # 2. Clean Russian text produces nothing.
    clean = (
        "Откройте файл `config.yaml` и найдите параметр timeout.\n"
        "Установите timeout в 30 секунд.\n"
        "Запустите сервер командой `docker compose up -d`.\n"
        "Проверьте журнал: команда `docker logs app` показывает порт 8080.\n"
        "Если порт занят, остановите процесс командой "
        "`docker compose down`.\n"
    )
    expect(analyze(clean).findings == [], "clean Russian text must be silent")
    expect(
        analyze(clean, mode=NORMAL).findings == [],
        "clean Russian text must be silent in normal mode too",
    )

    # 3. Modality is never a finding.
    hedges = (
        "Запрос может завершиться ошибкой. Возможно, причина в таймауте.\n"
        "Вероятно, сервер перегружен.\n"
        "Иногда ошибка возникает после перезапуска.\n"
        "Данные, возможно, устарели.\n"
    )
    expect(analyze(hedges).findings == [], "modality must never be flagged")

    # 4. Long sentences.
    long_sentence = " ".join("слово%d" % index for index in range(1, 22)) + "."
    findings = _findings_of(long_sentence, "long-sentence")
    expect(len(findings) == 1, "21 words must be one long-sentence finding")
    expect(findings[0].level == HARD, "strict long sentence is hard")
    expect(findings[0].suggestion == "Разделите предложение", "hint present")
    normal = _findings_of(long_sentence, "long-sentence", NORMAL)
    expect(normal[0].level == ADVISORY, "normal long sentence is advisory")
    very_long = " ".join("слово%d" % index for index in range(1, 27)) + "."
    expect(_findings_of(very_long, "long-sentence", NORMAL)[0].level == HARD,
           "26 words is hard even in normal mode")
    short = "Проверьте журнал."
    expect("long-sentence" not in _rules_of(short), "short sentence is fine")

    # 5. Bureaucracy and its exclusions.
    expect("bureaucratism" in _rules_of("В настоящее время сервис недоступен."),
           "vague time phrase")
    expect("bureaucratism" in _rules_of("Проверьте данный модуль."),
           "«данный» as an adjective is a bureaucratic construction")
    expect(_findings_of("Передайте данные в отчёт.", "bureaucratism") == [],
           "the noun «данные» is never a violation")

    # 6. Verb + noun pairs and their replacements.
    replacements = {
        "Произвести установку пакета.": "установить",
        "Выполнить проверку журнала.": "проверить",
        "Осуществить проверку состояния.": "проверить",
        "Произвести изменение настроек.": "изменить",
        "Осуществлять использование устаревшего API запрещено.": "использовать",
        "Представляется необходимым выполнить настройку.": "настроить",
    }
    for text, verb in replacements.items():
        findings = _findings_of(text, "verb-object-noun")
        expect(len(findings) == 1, "verb-object-noun fires for %r" % text)
        expect(verb in (findings[0].suggestion or ""),
               "replacement %r for %r" % (verb, text))
    expect(_findings_of("Установите пакет и проверьте журнал.", "verb-object-noun") == [],
           "plain verbs are not flagged")

    # 7. Passive voice.
    expect("passive-voice" in _rules_of("Файл был удалён после проверки."),
           "auxiliary plus participle")
    expect(_findings_of("Пользователь удалил файл сам.", "passive-voice") == [],
           "active voice is silent")
    expect(_findings_of("Новый файл был создан.", "passive-voice") != [],
           "short passive is reported")

    # 8. Synonym rotation.
    findings = _findings_of(
        "Проверьте конфигурацию. Затем валидируйте ответ сервера.", "synonym-rotation")
    expect(len(findings) == 1, "one rotation finding per extra term")
    expect("«проверить»" in findings[0].message, "keeper is named")
    expect(_findings_of("Проверьте конфигурацию. Проверьте ответ.", "synonym-rotation") == [],
           "consistent terms are silent")

    # 9. Dangling conjunction in list items.
    text = (
        "- Установите целевой объект и\n"
        "* Запишите результат или  \n"
        "+ Закройте панель\n"
        "1. Запустите задачу и\n"
        "2) Остановите задачу или\n"
    )
    findings = _findings_of(text, "dangling-conjunction")
    expect([item.line for item in findings] == [1, 2, 4, 5], "four dangling items")
    expect(all(item.level == HARD for item in findings), "dangling is hard")
    expect(_findings_of("  - Установите цель и\n    запишите результат.",
                        "dangling-conjunction") == [],
           "continuation lines are joined")
    expect(_findings_of("- Установите цель\n  и", "dangling-conjunction")[0].col == 3,
           "column points at the conjunction")
    expect(_findings_of("    - код и", "dangling-conjunction") == [],
           "four leading spaces is code, not a list")
    expect(_findings_of("> - Установите цель и", "dangling-conjunction") == [],
           "blockquotes are skipped")
    expect(_findings_of("```text\n- код и\n```", "dangling-conjunction") == [],
           "fenced code is skipped")
    expect(_findings_of("~~~text\n- код и\n~~~", "dangling-conjunction") == [],
           "tilde fences are skipped")
    expect(_findings_of("- Установите цель и.\n- Остановите цель или.",
                        "dangling-conjunction") == [],
           "an item closed with a full stop is not dangling")
    expect(_findings_of("- Установите цель и.\n- Остановите цель или,",
                        "dangling-conjunction") != [],
           "a trailing comma keeps the item unfinished")
    expect(_findings_of("- Установите цель\n- Остановите цель или",
                        "dangling-conjunction") != [],
           "a bare coordinator at the end is unfinished")
    expect(_findings_of("- Задайте цель\n  - И запишите результат.",
                        "dangling-conjunction") == [],
           "a nested item closes its parent")
    expect(_findings_of("- Родительский пункт и\n  - Вложенный пункт или",
                        "dangling-conjunction") != [],
           "nested items are checked")
    expect(_findings_of("- Откройте `and` как метку", "dangling-conjunction") == [],
           "inline code is not a conjunction")
    expect(_findings_of("- Объедините `left` и `right`", "dangling-conjunction") == [],
           "an item that ends with inline code is complete")
    expect(_findings_of("- Откройте `a` и", "dangling-conjunction")[0].col == 14,
           "a real coordinator after inline code is reported in place")

    # 10. Markdown structure is not prose.
    long_cell = " ".join("ячейка%d" % index for index in range(1, 27)) + "."
    table = ("| Поле | Значение |\n| --- | --- |\n| timeout | 30 |\n")
    analysis = analyze(table)
    expect(analysis.findings == [], "clean table is silent")
    expect(analysis.words == 3, "table cells are counted as prose, digits are not")
    table_long = ("| Поле | Значение |\n| --- | --- |\n| timeout | " + long_cell + " |\n")
    findings = _findings_of(table_long, "long-sentence")
    expect(len(findings) == 1 and findings[0].match == "26 слов",
           "a long table cell is one long sentence")
    table_plain = ("Поле | Значение\n--- | ---\ntimeout | " + long_cell + "\n")
    expect(len(_findings_of(table_plain, "long-sentence")) == 1,
           "table without leading pipes is a table")

    # 11. Technical literals are masked.
    literal_text = (
        "Откройте https://example.com/a;b?c=1#d в браузере.\n"
        "Скопируйте C:\\Users\\user\\report.xlsx в D:\\reports.\n"
        "Запустите `pytest -k smoke --json` и прочитайте вывод.\n"
        "Передайте {\"mode\": \"strict\", \"limit\": 20} сервису.\n"
        "Письма приходят на ops@example.com.\n"
        "Установите LOG_LEVEL=DEBUG перед запуском.\n"
        "Соберите проект командой ./scripts/build.sh.\n"
    )
    expect(analyze(literal_text).findings == [],
           "technical literals are not prose: %s"
           % analyze(literal_text).findings)
    expect("semicolon" not in _rules_of("Смотрите https://example.com/a;b."),
           "semicolon inside a URL is not a rule break")
    expect("long-sentence" not in _rules_of(
        "Запустите " + " ".join("--option-%d=значение%d" % (i, i) for i in range(30))
    ), "long CLI invocation is not prose")

    # 12. Russian prose is analysed, not skipped.
    expect("bureaucratism" in _rules_of("Оператор осуществляет проверку в целях контроля."),
           "Russian prose reaches the rules")
    expect("long-sentence" in _rules_of(
        " ".join("тест%d" % index for index in range(1, 27)) + "."),
        "long Russian sentence")

    # 13. Modes and levels.
    strict = _findings_of("Сервер обрабатывает запрос и возвращает ответ.",
                          "long-sentence", STRICT)
    expect(strict == [], "short sentence is fine in strict mode")
    expect(RULE_INDEX["passive-voice"].level(STRICT) == ADVISORY,
           "passive voice is advisory in strict mode")
    expect(RULE_INDEX["passive-voice"].level(NORMAL) == ADVISORY,
           "passive voice is advisory in normal mode")
    expect(RULE_INDEX["semicolon"].level(NORMAL) == HARD,
           "semicolon is hard in normal mode")

    # 14. Morphology adds two rules and stays optional.
    with_morphology = Morphology()
    if with_morphology.available:  # pragma: no cover - depends on environment
        expect("participle-clause" in _rules_of(
            "Файл, созданный сервером, удалён после проверки."),
            "participle-clause needs pymorphy3")
        expect("noun-chain" in _rules_of(
            "Настройка серверного модуля распределения нагрузки выполнена."),
            "noun-chain needs pymorphy3")
        expect("noun-chain" not in _rules_of("Настройка выполнена."),
               "short noun chain is fine")
    broken = Morphology()
    broken._analyzer = object()  # type: ignore[assignment]
    broken.backend = "broken"
    expect(broken.analyse("файл") == (False,) * 6,
           "a broken analyser degrades to the safe answer")
    expect(analyze("Проверьте журнал.", morphology=broken).findings == [],
           "a broken analyser does not break the linter")

    # 15. Determinism.
    sample = (
        "Выполните проверку; в целях контроля произведите изменение.\n"
        "- Установите целевой объект и\n"
    )
    first = [item.to_dict() for item in analyze(sample, filename="a.md").findings]
    second = [item.to_dict() for item in analyze(sample, filename="a.md").findings]
    expect(first == second, "the same input gives the same findings")

    # 16. Positions are exact.
    text = "# Заголовок\n\nОператор в целях контроля проверяет узел.\n"
    findings = _findings_of(text, "bureaucratism")
    expect(len(findings) == 1, "one bureaucracy finding")
    expect(text.splitlines()[findings[0].line - 1][findings[0].col - 1
                                                    :findings[0].col - 1 + len(
                                                        findings[0].match)]
           == findings[0].match, "the column points at the matched text")
    quoted = "> Оператор в целях контроля проверяет узел.\n"
    findings = _findings_of(quoted, "bureaucratism")
    expect(len(findings) == 1 and findings[0].col == 12,
           "blockquote prefix does not shift the column")

    # 17. CLI contract.
    findings = filter_findings(analyze(sample).findings, ["semicolon"])
    expect(_findings_of(sample, "semicolon"), "the sample has a semicolon")
    expect(not any(item.rule == "semicolon" for item in findings),
           "--disable removes a rule")
    payload = json.loads(render_json(
        filter_findings(analyze(sample).findings, []), analyze(sample),
        STRICT, 0, [], Morphology()))
    expect(payload["count"] == payload["hard_count"] + payload["advisory_count"],
           "counts add up")
    expect(payload["mode"] == STRICT, "mode is reported")

    # 18. An empty input is not a violation.
    expect(analyze("").findings == [], "empty text")
    expect(analyze("```\nкод\n```").findings == [], "fenced code only")
    expect(analyze("## Заголовок\n").findings == [], "a heading alone")
    expect(analyze("---\nname: x\ndescription: \"Проверьте; в целях контроля\"\n---\n"
                   "Проверьте журнал.\n").findings == [],
           "front matter is metadata, not prose")
    expect(_findings_of("---\ndescription: \"Проверьте; в целях контроля\"\n---\n"
                        "- Установите цель и\n", "dangling-conjunction")[0].line == 4,
           "front matter is skipped by the list check too")

    print("selftest OK: %d проверок, %d правил, морфология: %s"
          % (checks, len(RULES), Morphology().backend))
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.read()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="utr-lint.py",
        description="Линтер упрощённого технического русского языка (УТР). "
                    "Только анализ: текст не изменяется.",
    )
    parser.add_argument("files", nargs="*",
                        help="файлы для проверки; без файлов читается stdin")
    parser.add_argument("--json", action="store_true",
                        help="вывести результат в формате JSON")
    parser.add_argument("--baseline", type=int, default=0, metavar="N",
                        help="допустимое число жёстких нарушений (по умолчанию 0)")
    parser.add_argument("--disable", default="", metavar="R1,R2",
                        help="отключить правила по идентификаторам")
    parser.add_argument("--mode", choices=MODES, default=STRICT,
                        help="режим проверки (по умолчанию strict)")
    parser.add_argument("--list-rules", action="store_true",
                        help="показать список правил и выйти")
    parser.add_argument("--selftest", action="store_true",
                        help="запустить встроенную проверку и выйти")
    parser.add_argument("--version", action="version",
                        version="utr-lint.py %s" % __version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.list_rules:
        print(render_rule_list(args.mode))
        return 0

    unknown = [
        name for name in args.disable.split(",") if name and name not in RULE_INDEX
    ]
    if unknown:
        parser.error("неизвестные правила: %s" % ", ".join(sorted(unknown)))
    disabled = [name for name in args.disable.split(",") if name]

    morphology = Morphology()
    findings: list[Finding] = []
    analysis = Analysis()
    if args.files:
        for path in args.files:
            try:
                text = read_text(path)
            except OSError as error:
                print("utr-lint.py: %s" % error, file=sys.stderr)
                return 2
            part = analyze(text, filename=path, mode=args.mode,
                           morphology=morphology)
            findings.extend(part.findings)
            analysis.words += part.words
    else:
        text = sys.stdin.read()
        part = analyze(text, filename="<stdin>", mode=args.mode,
                       morphology=morphology)
        findings.extend(part.findings)
        analysis.words += part.words

    findings = filter_findings(findings, disabled)
    findings.sort(key=lambda item: (item.file, item.line, item.col, item.rule,
                                    item.match))
    analysis.findings = findings

    if args.json:
        print(render_json(findings, analysis, args.mode, args.baseline, disabled,
                          morphology))
    else:
        print(render_text(findings, analysis, args.mode, args.baseline, disabled,
                          morphology))

    hard_count = sum(1 for item in findings if item.level == HARD)
    return 1 if hard_count > args.baseline else 0


if __name__ == "__main__":
    sys.exit(main())