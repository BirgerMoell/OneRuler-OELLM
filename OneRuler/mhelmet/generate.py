"""Generate a multilingual HELMET-style benchmark over OneRuler-OELLM languages.

The generated samples follow OneRuler's JSONL convention:

    {"index": 0, "input": "...", "outputs": ["..."], "length": 123, ...}

HELMET evaluates broad long-context capabilities rather than only synthetic
needle retrieval.  This module builds deterministic multilingual analogues for
its seven categories: recall, RAG, reranking, citation, long QA, summarization,
and in-context learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from oellm_support import OELLM_LANGUAGES, synthetic_book_sentences
from references import prompt_for, reference_pack


TASKS = ("recall", "rag", "rerank", "cite", "longqa", "summ", "icl")
DEFAULT_LENGTHS = (8192, 16384, 32768)


class WhitespaceTokenizer:
    def text_to_tokens(self, text: str) -> list[str]:
        return text.split()


def load_tokenizer(tokenizer_type: str, tokenizer_path: str):
    if tokenizer_type == "whitespace":
        return WhitespaceTokenizer()
    if tokenizer_type == "openai":
        try:
            import tiktoken
        except ImportError:
            print("tiktoken is not installed; falling back to whitespace token counts.")
            return WhitespaceTokenizer()
        encoding = tiktoken.get_encoding(tokenizer_path)

        class OpenAITokenizer:
            def text_to_tokens(self, text: str) -> list[int]:
                return encoding.encode(text)

        return OpenAITokenizer()
    if tokenizer_type == "hf":
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print("transformers is not installed; falling back to whitespace token counts.")
            return WhitespaceTokenizer()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        class HFTokenizer:
            def text_to_tokens(self, text: str) -> list[str]:
                return tokenizer.tokenize(text)

        return HFTokenizer()
    raise ValueError(f"Unknown tokenizer_type {tokenizer_type}")


def stable_code(*parts: object, width: int = 6) -> str:
    digest = hashlib.sha1("::".join(map(str, parts)).encode("utf-8")).hexdigest()
    return str(int(digest[:10], 16))[-width:].zfill(width)


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer.text_to_tokens(text))


def fit_context(tokenizer, template: str, context_units: list[str], max_seq_length: int, reserve: int, **fields) -> str:
    if not context_units:
        return ""

    lo, hi = 1, len(context_units)
    best = context_units[:1]
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = "\n".join(context_units[:mid])
        rendered = template.format(context=candidate, **fields)
        if token_len(tokenizer, rendered) + reserve <= max_seq_length:
            best = context_units[:mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return "\n".join(best)


def fit_context_with_required(
    tokenizer,
    template: str,
    context_units: list[str],
    required_units: list[str],
    max_seq_length: int,
    reserve: int,
    **fields,
) -> str:
    """Fit context while guaranteeing that gold-supporting units remain present."""
    required = list(dict.fromkeys(required_units))
    selected = required.copy()
    required_set = set(required)
    for unit in context_units:
        if unit in required_set:
            continue
        candidate = selected + [unit]
        rendered = template.format(context="\n".join(candidate), **fields)
        if token_len(tokenizer, rendered) + reserve > max_seq_length:
            continue
        selected = candidate
    rendered = template.format(context="\n".join(selected), **fields)
    if token_len(tokenizer, rendered) + reserve > max_seq_length:
        raise ValueError("Required evidence does not fit in the requested context length")
    return "\n".join(selected)


def language_units(lang: str, seed: int, minimum: int = 4096) -> list[str]:
    rng = random.Random(seed)
    units = synthetic_book_sentences(lang, count=max(512, min(minimum, 8192)))
    rng.shuffle(units)
    while len(units) < minimum:
        units.extend(units)
    return units[:minimum]


def reference_terms(lang: str, multiplier: int = 64) -> list[str]:
    pack = reference_pack(lang)
    base = pack["terms"] + pack["cities"] + pack["outcomes"]
    return [f"{term}-{idx}" for idx in range(multiplier) for term in base]


def non_target_term(terms: list[str], index: int, target: str) -> str:
    term = terms[index % len(terms)]
    if term == target:
        term = terms[(index + 1) % len(terms)]
    return term


def reference_sentence(lang: str, kind: str, **values: str) -> str:
    fragments = {
        "recall": {
            "bg": "Регистрова бележка: {key} има присвоен код {answer}.",
            "de": "Registervermerk: {key} hat den zugewiesenen Code {answer}.",
            "en": "Registry note: {key} has assigned code {answer}.",
            "es": "Nota de registro: {key} tiene asignado el código {answer}.",
            "fr": "Note de registre : {key} reçoit le code {answer}.",
            "it": "Nota di registro: {key} ha il codice assegnato {answer}.",
            "pt": "Nota de registo: {key} tem o código atribuído {answer}.",
            "sv": "Registeranteckning: {key} har tilldelad kod {answer}.",
        },
        "meeting": {
            "bg": "Запис за среща: домакин на {key} беше {city}.",
            "de": "Sitzungsprotokoll: Gastgeberstadt für {key} war {city}.",
            "en": "Meeting record: the host city for {key} was {city}.",
            "es": "Registro de reunión: la ciudad anfitriona de {key} fue {city}.",
            "fr": "Compte rendu : la ville hôte de {key} était {city}.",
            "it": "Verbale della riunione: la città ospite per {key} era {city}.",
            "pt": "Registo da reunião: a cidade anfitriã de {key} foi {city}.",
            "sv": "Mötesprotokoll: värdstaden för {key} var {city}.",
        },
        "registry": {
            "bg": "{key} има регистрова стойност {value}.",
            "de": "{key} hat den Registerwert {value}.",
            "en": "{key} registry value is {value}.",
            "es": "El valor de registro de {key} es {value}.",
            "fr": "La valeur de registre de {key} est {value}.",
            "it": "Il valore di registro di {key} è {value}.",
            "pt": "O valor de registo de {key} é {value}.",
            "sv": "Registervärdet för {key} är {value}.",
        },
        "verify": {
            "bg": "Проверочен запис: {key} е свързан с фразата {phrase}.",
            "de": "Verifizierungsvermerk: {key} ist mit der Phrase {phrase} verknüpft.",
            "en": "Verification record: {key} is linked to phrase {phrase}.",
            "es": "Registro de verificación: {key} está vinculado a la frase {phrase}.",
            "fr": "Notice de vérification : {key} est lié à la phrase {phrase}.",
            "it": "Record di verifica: {key} è collegato alla frase {phrase}.",
            "pt": "Registo de verificação: {key} está ligado à frase {phrase}.",
            "sv": "Verifieringspost: {key} är kopplad till frasen {phrase}.",
        },
        "summary": {
            "bg": "Проект {key} приключи с резултат {outcome} {value}.",
            "de": "Projekt {key} schloss mit Ergebnis {outcome} {value}.",
            "en": "Project {key} closed with outcome {outcome} {value}.",
            "es": "El proyecto {key} cerró con resultado {outcome} {value}.",
            "fr": "Le projet {key} s'est clos avec le résultat {outcome} {value}.",
            "it": "Il progetto {key} si è chiuso con esito {outcome} {value}.",
            "pt": "O projeto {key} fechou com resultado {outcome} {value}.",
            "sv": "Projekt {key} avslutades med resultat {outcome} {value}.",
        },
    }
    template = fragments[kind].get(lang, fragments[kind]["en"])
    return template.format(**values)


def make_recall(lang: str, index: int, rng: random.Random, tokenizer, max_seq_length: int) -> dict:
    nouns = reference_terms(lang)
    key = nouns[index % len(nouns)]
    answer = stable_code(lang, "recall", index)
    fact = reference_sentence(lang, "recall", key=key, answer=answer)
    units = language_units(lang, rng.randint(0, 10**9))
    insert_at = rng.randrange(max(1, len(units)))
    units.insert(insert_at, fact)
    template = prompt_for(lang, "recall")
    context = fit_context_with_required(tokenizer, template, units, [fact], max_seq_length, 32, query=key)
    return sample(index, lang, "recall", template.format(context=context, query=key), [answer], tokenizer)


def make_rag(lang: str, index: int, rng: random.Random, tokenizer, max_seq_length: int) -> dict:
    pack = reference_pack(lang)
    nouns = reference_terms(lang)
    key = nouns[(index * 7) % len(nouns)]
    city = pack["cities"][(index * 7 + 3) % len(pack["cities"])]
    passages = []
    required = ""
    for rank in range(256):
        label = f"P{rank + 1}"
        if rank == 127:
            body = reference_sentence(lang, "meeting", key=key, city=city)
            required = f"[{label}] {body}"
        else:
            distractor = non_target_term(nouns, index + rank * 13, key)
            filler = " ".join(nouns[(rank + offset) % len(nouns)] for offset in range(8))
            body = (
                f"{reference_sentence(lang, 'meeting', key=distractor, city=pack['cities'][(rank * 19) % len(pack['cities'])])} "
                f"{filler}."
            )
        passage = f"[{label}] {body}"
        passages.append(passage)
    template = prompt_for(lang, "rag")
    context = fit_context_with_required(tokenizer, template, passages, [required], max_seq_length, 32, query=key)
    return sample(index, lang, "rag", template.format(context=context, query=key), [city], tokenizer)


def make_rerank(lang: str, index: int, rng: random.Random, tokenizer, max_seq_length: int) -> dict:
    nouns = reference_terms(lang)
    key = nouns[(index * 11) % len(nouns)]
    labels = ["A", "B", "C", "D", "E"]
    strengths = [80, 20, 100, 0, 60]
    relevant = [
        f"[{label}] {key} relevance {score}. {reference_sentence(lang, 'registry', key=key, value=str(score))}"
        for label, score in zip(labels, strengths)
    ]
    distractors = []
    for rank in range(160):
        distractor = non_target_term(nouns, index + rank * 17, key)
        filler = " ".join(nouns[(rank + offset) % len(nouns)] for offset in range(10))
        distractors.append(f"Background note {rank + 1}: {distractor}. {filler}.")
    rng.shuffle(relevant)
    passages = relevant + distractors
    template = (
        "Rank exactly these five candidate passages by relevance score for the archive key {query}.\n"
        "A larger numeric relevance score means a more relevant passage.\n"
        "Use only candidate labels [A], [B], [C], [D], and [E]. Ignore background notes.\n\n"
        "<Candidates>\n{context}\n</Candidates>\n\n"
        "Return all five labels in descending relevance-score order, separated by commas."
    )
    context = fit_context_with_required(tokenizer, template, passages, relevant, max_seq_length, 48, query=key)
    expected = [label for _, label in sorted(zip(strengths, labels), reverse=True)]
    input_text = template.format(context=context, query=key)
    input_text += (
        "\n\nRanking candidates: [A], [B], [C], [D], [E]. "
        "Ignore all background notes. Return all five candidate labels sorted by relevance score, highest first."
    )
    return sample(index, lang, "rerank", input_text, [", ".join(expected)], tokenizer)


def make_cite(lang: str, index: int, rng: random.Random, tokenizer, max_seq_length: int) -> dict:
    nouns = reference_terms(lang)
    key = nouns[(index * 17) % len(nouns)]
    value = stable_code(lang, "cite", index, width=5)
    support = "S3"
    passages = []
    required = ""
    for rank in range(220):
        label = f"S{rank + 1}"
        if label == support:
            required = f"[{support}] {reference_sentence(lang, 'registry', key=key, value=value)}"
            passages.append(required)
        else:
            distractor = non_target_term(nouns, index + rank * 23, key)
            filler = " ".join(nouns[(rank + offset) % len(nouns)] for offset in range(8))
            passages.append(
                f"[{label}] {reference_sentence(lang, 'registry', key=distractor, value=stable_code(lang, index, rank, width=5))} {filler}."
            )
    template = prompt_for(lang, "cite")
    context = fit_context_with_required(tokenizer, template, passages, [required], max_seq_length, 48, query=key)
    return sample(index, lang, "cite", template.format(context=context, query=key), [f"{value} [{support}]"], tokenizer)


def make_longqa(lang: str, index: int, rng: random.Random, tokenizer, max_seq_length: int) -> dict:
    nouns = reference_terms(lang)
    key = nouns[(index * 23) % len(nouns)]
    phrase = f"{nouns[(index + 4) % len(nouns)]}-{stable_code(lang, 'qa', index, width=4)}"
    units = language_units(lang, rng.randint(0, 10**9))
    required = reference_sentence(lang, "verify", key=key, phrase=phrase)
    units.insert(len(units) // 2, required)
    template = prompt_for(lang, "longqa")
    context = fit_context_with_required(tokenizer, template, units, [required], max_seq_length, 64, query=key)
    return sample(index, lang, "longqa", template.format(context=context, query=key), [phrase], tokenizer)


def make_summ(lang: str, index: int, rng: random.Random, tokenizer, max_seq_length: int) -> dict:
    pack = reference_pack(lang)
    nouns = reference_terms(lang)
    keys = [nouns[(index + offset * 9) % len(nouns)] for offset in range(3)]
    values = [stable_code(lang, "summ", index, offset, width=4) for offset in range(3)]
    facts = [
        reference_sentence(
            lang,
            "summary",
            key=key,
            outcome=pack["outcomes"][offset % len(pack["outcomes"])],
            value=value,
        )
        for offset, (key, value) in enumerate(zip(keys, values))
    ]
    units = language_units(lang, rng.randint(0, 10**9))
    for pos, fact in zip((30, len(units) // 2, len(units) - 30), facts):
        units.insert(max(0, min(pos, len(units))), fact)
    template = prompt_for(lang, "summ")
    context = fit_context_with_required(tokenizer, template, units, facts, max_seq_length, 160, query="")
    return sample(index, lang, "summ", template.format(context=context), facts, tokenizer)


def make_icl(lang: str, index: int, rng: random.Random, tokenizer, max_seq_length: int) -> dict:
    nouns = reference_terms(lang)
    examples = []
    for offset in range(400):
        item = nouns[(index * 5 + offset) % len(nouns)]
        label = f"L{(offset * 7 + index) % 13:02d}"
        examples.append(f"Item: {item} -> Label: {label}")
    query_offset = 233
    query = nouns[(index * 5 + query_offset) % len(nouns)]
    answer = f"L{(query_offset * 7 + index) % 13:02d}"
    required = f"Item: {query} -> Label: {answer}"
    examples.append(required)
    rng.shuffle(examples)
    template = prompt_for(lang, "icl")
    context = fit_context_with_required(tokenizer, template, examples, [required], max_seq_length, 32, query=query)
    return sample(index, lang, "icl", template.format(context=context, query=query), [answer], tokenizer)


BUILDERS = {
    "recall": make_recall,
    "rag": make_rag,
    "rerank": make_rerank,
    "cite": make_cite,
    "longqa": make_longqa,
    "summ": make_summ,
    "icl": make_icl,
}


def sample(index: int, lang: str, task: str, input_text: str, outputs: list[str], tokenizer) -> dict:
    return {
        "index": index,
        "language": lang,
        "language_name": OELLM_LANGUAGES[lang],
        "task": task,
        "input": input_text,
        "outputs": outputs,
        "length": token_len(tokenizer, input_text),
    }


def parse_csv(value: str, choices: tuple[str, ...] | None = None) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if choices:
        unknown = sorted(set(items) - set(choices))
        if unknown:
            raise argparse.ArgumentTypeError(f"Unknown values: {', '.join(unknown)}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multilingual HELMET-style JSONL data.")
    parser.add_argument("--save_dir", type=Path, default=Path("dataset/mhelmet"))
    parser.add_argument("--languages", default=",".join(OELLM_LANGUAGES.keys()))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)))
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--tokenizer_type", default="hf", choices=["hf", "openai", "whitespace"])
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    languages = parse_csv(args.languages)
    unknown_langs = sorted(set(languages) - set(OELLM_LANGUAGES))
    if unknown_langs:
        raise SystemExit(f"Unknown OpenEuroLLM language codes: {', '.join(unknown_langs)}")
    tasks = parse_csv(args.tasks, TASKS)
    lengths = [int(length) for length in parse_csv(args.lengths)]

    tokenizer = load_tokenizer(args.tokenizer_type, args.tokenizer_path)

    for lang in languages:
        for max_seq_length in lengths:
            for task in tasks:
                rng = random.Random(f"{args.random_seed}:{lang}:{max_seq_length}:{task}")
                builder = BUILDERS[task]
                out_file = args.save_dir / lang / str(max_seq_length) / task / "validation.jsonl"
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with out_file.open("w", encoding="utf-8") as handle:
                    for index in range(args.num_samples):
                        item = builder(lang, index, rng, tokenizer, max_seq_length)
                        json.dump(item, handle, ensure_ascii=False)
                        handle.write("\n")
                print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
