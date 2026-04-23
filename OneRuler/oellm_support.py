"""OpenEuroLLM language support helpers for OneRuler.

The original OneRuler release ships native resources for 26 languages.  This
module adds support for 21 additional OpenEuroLLM languages (bg, hr, et, el,
ga, lv, lt, mt, ro, sk, sl, sq, eu, bs, ca, gl, is, lb, mk, tr, cy).

Each language has:
  - Translated NIAH and CWE prompt files in data/prompt/{lang}/
  - 100-noun reference translations in data/vocab/100_noun_list_translated.tsv
  - A POS word list in data/vocab/dictionaries/{Language}/{Language}.csv

When a book haystack is not available for a language, synthetic sentences are
generated as a fallback via synthetic_book_sentences().
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

_DATA_DIR = Path(__file__).parent / "data"


OELLM_LANGUAGES = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "ga": "Irish",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "en": "English",
    "sq": "Albanian",
    "eu": "Basque",
    "bs": "Bosnian",
    "ca": "Catalan",
    "gl": "Galician",
    "is": "Icelandic",
    "lb": "Luxembourgish",
    "mk": "Macedonian",
    "no": "Norwegian",
    "ru": "Russian",
    "sr": "Serbian",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "cy": "Welsh",
}

NATIVE_ONERULER_LANGUAGES = {
    "cs",
    "da",
    "de",
    "en",
    "es",
    "fa",
    "fi",
    "fr",
    "hi",
    "hu",
    "it",
    "ja",
    "ko",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sr",
    "st",
    "sv",
    "sw",
    "ta",
    "uk",
    "vi",
    "zh",
}

MISSING_OELLM_LANGUAGES = tuple(
    code for code in OELLM_LANGUAGES if code not in NATIVE_ONERULER_LANGUAGES
)

NONE_WORDS = {
    "bg": ["няма"],
    "hr": ["nema"],
    "et": ["puudub"],
    "el": ["κανένα"],
    "ga": ["níl"],
    "lv": ["nav"],
    "lt": ["nėra"],
    "mt": ["xejn"],
    "ro": ["niciunul"],
    "sk": ["žiadne"],
    "sl": ["nič"],
    "sq": ["asnjë"],
    "eu": ["bat ere ez"],
    "bs": ["nema"],
    "ca": ["cap"],
    "gl": ["ningún"],
    "is": ["ekkert"],
    "lb": ["keen"],
    "mk": ["нема"],
    "tr": ["yok"],
    "cy": ["dim"],
}

_ALPHABETS = {
    "bg": "абвгдежзиклмнопрстуфхцчшя",
    "el": "αβγδεζηθικλμνξοπρστυφχψω",
    "mk": "абвгдежзиклмнопрстуфхцчшја",
}

_LATIN_ALPHABET = "abcdefghijklmnoprstuvwyz"


def is_oellm_language(lang: str) -> bool:
    return lang.lower() in OELLM_LANGUAGES


def has_native_oneruler_resources(lang: str) -> bool:
    return lang.lower() in NATIVE_ONERULER_LANGUAGES


def sentence_ending(lang: str) -> str:
    if lang in {"zh", "ja"}:
        return "。"
    if lang == "fa":
        return "۔"
    if is_oellm_language(lang) or has_native_oneruler_resources(lang):
        return "."
    raise ValueError(f"Unsupported language code: {lang}")


def none_words(lang: str) -> list[str]:
    return NONE_WORDS.get(lang.lower(), ["none"])


def _alphabet(lang: str) -> str:
    return _ALPHABETS.get(lang.lower(), _LATIN_ALPHABET)


def _word_from_index(lang: str, index: int, prefix: str) -> str:
    alphabet = _alphabet(lang)
    base = len(alphabet)
    chars = []
    value = index
    while True:
        chars.append(alphabet[value % base])
        value //= base
        if value == 0:
            break
    stem = "".join(reversed(chars)).ljust(3, alphabet[index % base])
    if lang in _ALPHABETS:
        return stem + _ALPHABETS[lang][(index + len(prefix)) % len(_ALPHABETS[lang])]
    return f"{prefix}{stem}"


def synthetic_nouns(lang: str, count: int = 100) -> list[str]:
    _require_missing_or_oellm(lang)
    return [_word_from_index(lang, index, "nom") for index in range(count)]


def synthetic_pos_words(lang: str, count: int = 5000) -> tuple[list[str], list[str], list[str]]:
    _require_missing_or_oellm(lang)
    nouns = [_word_from_index(lang, index, "nom") for index in range(count)]
    adjectives = [_word_from_index(lang, index, "adj") for index in range(count)]
    verbs = [_word_from_index(lang, index, "verb") for index in range(count)]
    return nouns, adjectives, verbs


def synthetic_book_sentences(lang: str, count: int = 2048) -> list[str]:
    _require_missing_or_oellm(lang)
    words = synthetic_nouns(lang, count=512)
    ending = sentence_ending(lang)
    sentences = []
    for index in range(count):
        window = [words[(index + offset) % len(words)] for offset in range(12)]
        sentences.append(" ".join(window).capitalize() + ending)
    return sentences


def niah_prompt_dict(lang: str) -> dict[str, str]:
    prompt_file = _DATA_DIR / "prompt" / lang.lower() / "niah.txt"
    if prompt_file.exists():
        return json.loads(prompt_file.read_text(encoding="utf-8"))
    none = none_words(lang)[0]
    return {
        "task": "Please read and memorize the text below. I will ask you about it later.\n\n<text>\n{context}\n</text>\n\n",
        "needle_words": 'The special magic word for "{key}" is: {value} ',
        "needle_numbers": 'The special magic number for "{key}" is: {value} ',
        "question_single_numbers": '<Question> What special magic numbers associated with "{query1}" are mentioned in the provided text?',
        "question_single_words": '<Question> What special magic words associated with "{query1}" are mentioned in the provided text?',
        "question_multi_numbers": '<Question> What special magic numbers associated with "{query1}" and "{query2}" are mentioned in the provided text?',
        "question_multi_words": '<Question> What special magic words associated with "{query1}" and "{query2}" are mentioned in the provided text?',
        "please_list": "Please list all that apply.",
        "if_no_numbers": f'If no such numbers exist, please answer "{none}".</Question>\n\n\n',
        "if_no_words": f'If no such words exist, please answer "{none}".</Question>\n\n\n',
        "answer_prefix": "Please provide your answer in the following format:\n",
        "answer_words": "<Answer> List all words here </Answer>",
        "answer_numbers": "<Answer> List all numbers here </Answer>",
        "none": none,
    }


def cwe_prompt(lang: str) -> str:
    _require_missing_or_oellm(lang)
    prompt_file = _DATA_DIR / "prompt" / lang.lower() / "cwe.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    return (
        "Below is a numbered list of words. In these words, some appear more "
        "often than others. Memorize the ones that appear most often.\n"
        "<List> {context} </List>\n"
        "<Question> What are the 10 most common words in the above list? </Question>\n"
        "Please provide your answer in the following format:\n"
        "<Answer> List the words here </Answer>"
    )


def nouns_for_language(lang: str, noun_df=None) -> list[str]:
    lang = lang.lower()
    if noun_df is not None and lang in noun_df.columns:
        return [noun.strip() for noun in noun_df[lang].dropna().tolist()]
    if is_oellm_language(lang):
        return synthetic_nouns(lang)
    raise ValueError(f"No noun resources for language code: {lang}")


def translate_noun(query: str, source_lang: str, target_lang: str, noun_df=None) -> str:
    source_nouns = nouns_for_language(source_lang, noun_df)
    target_nouns = nouns_for_language(target_lang, noun_df)
    try:
        index = source_nouns.index(query.strip())
    except ValueError:
        return query
    if index >= len(target_nouns):
        return query
    return target_nouns[index]


def read_text_if_exists(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _require_missing_or_oellm(lang: str) -> None:
    if not is_oellm_language(lang):
        raise ValueError(f"Unsupported OELLM language code: {lang}")


def language_names(codes: Iterable[str]) -> dict[str, str]:
    return {code: OELLM_LANGUAGES[code] for code in codes}
