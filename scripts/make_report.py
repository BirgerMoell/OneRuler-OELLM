#!/usr/bin/env python3
"""Generate eval_results/oellm_report.pdf — a self-contained evaluation report."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

REPO_ROOT  = Path(__file__).resolve().parents[1]
EVAL_ROOT  = REPO_ROOT / "eval_results"
FIGURE_PATH = EVAL_ROOT / "mini_eval" / "oellm_eval_results.png"
OUT_PATH   = EVAL_ROOT / "oellm_report.pdf"

# ── colour palette ────────────────────────────────────────────────────────────
DARK_BG   = colors.HexColor("#0D1117")
PANEL_BG  = colors.HexColor("#161B22")
BORDER    = colors.HexColor("#30363D")
GOLD      = colors.HexColor("#F0C060")
ORANGE    = colors.HexColor("#FF9900")
RED_SOFT  = colors.HexColor("#FF6B6B")
GREEN     = colors.HexColor("#3FB950")
YELLOW    = colors.HexColor("#FFD700")
BLUE_SOFT = colors.HexColor("#9EB8D0")
WHITE     = colors.white
GREY_LIGHT= colors.HexColor("#D0D0D0")
GREY_MED  = colors.HexColor("#808080")

EU_OFFICIAL = {
    "bg","hr","cs","da","nl","et","fi","fr","de","el","hu","ga",
    "it","lv","lt","mt","pl","pt","ro","sk","sl","es","sv","en",
}
LANG_NAMES = {
    "bg":"Bulgarian","hr":"Croatian","cs":"Czech","da":"Danish","nl":"Dutch",
    "et":"Estonian","fi":"Finnish","fr":"French","de":"German","el":"Greek",
    "hu":"Hungarian","ga":"Irish","it":"Italian","lv":"Latvian","lt":"Lithuanian",
    "mt":"Maltese","pl":"Polish","pt":"Portuguese","ro":"Romanian","sk":"Slovak",
    "sl":"Slovenian","es":"Spanish","sv":"Swedish","en":"English",
    "sq":"Albanian","eu":"Basque","bs":"Bosnian","ca":"Catalan","gl":"Galician",
    "is":"Icelandic","lb":"Luxembourgish","mk":"Macedonian","no":"Norwegian",
    "ru":"Russian","sr":"Serbian","tr":"Turkish","uk":"Ukrainian","cy":"Welsh",
}

# All benchmarked models: (display_label, csv_path, context_words, num_predict, backend, notes)
MODELS_META = [
    ("Qwen2 0.5B",          EVAL_ROOT/"mini_eval"/"qwen2-0.5b"/"summary.csv",   200,  128, "Ollama", "easy baseline"),
    ("Qwen2 1.5B",          EVAL_ROOT/"mini_eval"/"qwen2-1.5b"/"summary.csv",   200,  128, "Ollama", "easy baseline"),
    ("Gemma 4 E4B (easy)",  EVAL_ROOT/"mini_eval"/"gemma4"/"summary.csv",       200,  512, "Ollama", "token budget limited"),
    ("Gemma 3 4B",          EVAL_ROOT/"full_eval"/"gemma3-4b"/"summary.csv",   2000, 1024, "Ollama", "harder benchmark"),
    ("Gemma 4 E2B",         EVAL_ROOT/"full_eval"/"gemma4-e2b"/"summary.csv",  2000, 1024, "Ollama", "harder benchmark"),
    ("Gemma 4 E4B",         EVAL_ROOT/"full_eval"/"gemma4"/"summary.csv",      2000, 1024, "Ollama", "harder benchmark"),
]
# OpenEuroLLM datamix models — add if CSV exists
DATAMIX_MODELS = [
    ("OE-LLM 2B en-only",  "datamix-2b-en",    "100% English data"),
    ("OE-LLM 2B 50/50",    "datamix-2b-50-50", "50% EU / 50% English"),
    ("OE-LLM 2B 60/40",    "datamix-2b-60-40", "60% EU / 40% English"),
    ("OE-LLM 2B 70/30",    "datamix-2b-70-30", "70% EU / 30% English"),
    ("OE-LLM 2B 80/20",    "datamix-2b-80-20", "80% EU / 20% English"),
    ("OE-LLM 2B 90/10",    "datamix-2b-90-10", "90% EU / 10% English"),
]


def load_csv(path: Path) -> dict[str, dict] | None:
    if not path.exists():
        return None
    rows = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["lang"]] = row
    return rows


def avg_accuracy(scores: dict[str, dict]) -> float:
    vals = [float(r["accuracy"]) for r in scores.values()]
    return sum(vals) / len(vals) if vals else 0.0


def build_styles() -> dict:
    s = {}
    def add(name, **kw):
        s[name] = ParagraphStyle(name, **kw)

    add("title",     fontSize=22, leading=28, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6)
    add("subtitle",  fontSize=11, leading=15, textColor=GREY_MED,
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=4)
    add("authors",   fontSize=10, leading=14, textColor=GREY_LIGHT,
        fontName="Helvetica", alignment=TA_CENTER, spaceAfter=2)
    add("section",   fontSize=13, leading=18, textColor=GOLD,
        fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=6)
    add("subsection",fontSize=10.5, leading=15, textColor=ORANGE,
        fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4)
    add("body",      fontSize=9.5, leading=14, textColor=GREY_LIGHT,
        fontName="Helvetica", alignment=TA_JUSTIFY, spaceAfter=5)
    add("bullet",    fontSize=9.5, leading=13, textColor=GREY_LIGHT,
        fontName="Helvetica", leftIndent=14, spaceAfter=3, bulletIndent=4)
    add("caption",   fontSize=8,   leading=11, textColor=GREY_MED,
        fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=8)
    add("th",        fontSize=8.5, leading=11, textColor=GOLD,
        fontName="Helvetica-Bold", alignment=TA_CENTER)
    add("th_left",   fontSize=8.5, leading=11, textColor=GOLD,
        fontName="Helvetica-Bold", alignment=TA_LEFT)
    add("td",        fontSize=8,   leading=11, textColor=GREY_LIGHT,
        fontName="Helvetica", alignment=TA_CENTER)
    add("td_left",   fontSize=8,   leading=11, textColor=GREY_LIGHT,
        fontName="Helvetica", alignment=TA_LEFT)
    add("code",      fontSize=8.5, leading=13, textColor=colors.HexColor("#A8D8A8"),
        fontName="Courier", backColor=PANEL_BG, borderColor=BORDER,
        borderWidth=0.5, borderPadding=8, spaceAfter=6)
    add("footer",    fontSize=7.5, textColor=GREY_MED,
        fontName="Helvetica", alignment=TA_CENTER)
    return s


def _p(text, style):
    return Paragraph(text, style)


def _tbl(rows, col_widths, extra_style=None):
    base = [
        ("BACKGROUND",   (0, 0), (-1,  0), PANEL_BG),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [DARK_BG, PANEL_BG]),
        ("GRID",         (0, 0), (-1, -1), 0.4, BORDER),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]
    if extra_style:
        base.extend(extra_style)
    return Table(rows, colWidths=col_widths, style=TableStyle(base))


def _hrule():
    return Table([[""]], colWidths=[17 * cm],
                 style=TableStyle([
                     ("LINEABOVE",     (0, 0), (-1, -1), 0.8, GOLD),
                     ("TOPPADDING",    (0, 0), (-1, -1), 0),
                     ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                 ]))


def _hrule_thin():
    return Table([[""]], colWidths=[17 * cm],
                 style=TableStyle([
                     ("LINEABOVE",     (0, 0), (-1, -1), 0.4, BORDER),
                     ("TOPPADDING",    (0, 0), (-1, -1), 0),
                     ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                 ]))


def acc_color(acc: float) -> colors.Color:
    if acc >= 1.0:
        return GREEN
    if acc >= 0.75:
        return ORANGE
    if acc > 0.0:
        return RED_SOFT
    return colors.HexColor("#F85149")


def acc_cell(acc_str: str, style) -> Paragraph:
    acc = float(acc_str)
    st = ParagraphStyle("ac", fontSize=8, leading=11, fontName="Helvetica-Bold",
                        alignment=TA_CENTER, textColor=acc_color(acc))
    return Paragraph(f"{acc:.0%}", st)


# ─────────────────────────────────────────────────────────────────────────────

def build_doc(story, s):

    # ── Title page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * cm))
    story.append(_p("OELLM Evaluation Report", s["title"]))
    story.append(_p(
        "Needle-in-a-Haystack across 38 European Languages", s["subtitle"]))
    story.append(_p(
        "OneRuler benchmark · OpenEuroLLM language extension · "
        f"{date.today().strftime('%B %d, %Y')}", s["authors"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(_hrule())
    story.append(Spacer(1, 0.5 * cm))

    # ── Abstract ─────────────────────────────────────────────────────────────
    story.append(_p("Abstract", s["section"]))
    story.append(_p(
        "We benchmark a range of language models on a Needle-in-a-Haystack (NIAH) "
        "retrieval task across all 38 languages targeted by the OpenEuroLLM tokenizer. "
        "The evaluation extends the OneRuler multilingual benchmark with resources for "
        "21 additional European languages. "
        "We compare two benchmark difficulties: an <i>easy</i> setting with a short "
        "noun-only haystack (~200 words, no distractors), and a <i>harder</i> setting "
        "with a ~2 000-word real-book haystack, ~10 distracting 7-digit numbers "
        "embedded throughout, and needle position varied from 10% to 90% depth. "
        "Among instruction-following models, <b>Gemma 4 E4B scores 91%</b> and "
        "<b>Gemma 4 E2B scores 86%</b> on the harder benchmark. "
        "We additionally evaluate six OpenEuroLLM 2B base models with varying "
        "European/English data mixtures; as base models without instruction tuning "
        "they do not follow the structured NIAH format, but the results provide "
        "a reference point for future instruction-tuned variants.",
        s["body"],
    ))

    # ── 1. Background ─────────────────────────────────────────────────────────
    story.append(_p("1  Background", s["section"]))
    story.append(_p(
        "<b>OneRuler</b> (Kim et al., 2025) is a multilingual long-context benchmark "
        "covering 26 languages with seven synthetic tasks including several NIAH variants. "
        "This fork adds runtime support for the 38 languages targeted by the "
        "OpenEuroLLM tokenizer, contributing translated prompt files, noun vocabulary "
        "lists, and POS word lists for 21 additional languages. "
        "The 26 languages already in OneRuler have real novel-text haystacks "
        "(<code>data/books/{lang}/</code>); the 12 newly added languages "
        "fall back to synthetic noun-sentence haystacks.",
        s["body"],
    ))
    story.append(_p(
        "<b>OpenEuroLLM</b> is an open initiative training multilingual LLMs for "
        "European languages. The datamix-2b series of models explores how varying "
        "the ratio of European to English training data affects multilingual capability. "
        "All datamix-2b models are base (non-instruction-tuned) LlamaForCausalLM "
        "models with a 262 272-token vocabulary and a 2 048-token context window.",
        s["body"],
    ))

    # ── 2. Task ───────────────────────────────────────────────────────────────
    story.append(_p("2  Task: Needle-in-a-Haystack (NIAH)", s["section"]))

    story.append(_p("2.1  Construction", s["subsection"]))
    for bullet in [
        "<b>Haystack.</b> A passage of text in the target language of approximately "
        "the target word count. For languages with a real novel in "
        "<code>data/books/{lang}/</code> (26 of 38 languages) the haystack is drawn "
        "from the actual book text, starting at a random sentence and continuing until "
        "the word target is reached. For the remaining 12 languages the haystack is "
        "built from 10-word synthetic sentences constructed by cycling through a list "
        "of 100 translated nouns.",
        "<b>Distractors.</b> On the harder benchmark, "
        "approximately <i>max(5, context_words ÷ 200)</i> sentences containing a "
        "random 7-digit number (1 000 000 – 9 999 999) are inserted at random positions "
        "in the haystack. This ensures the needle value is not the only multi-digit "
        "number in the context, requiring the model to identify the correct key.",
        "<b>Needle.</b> A single sentence stating: <i>\"The special magic number for "
        "&lt;key&gt; is: &lt;value&gt;\"</i>, translated into the target language using "
        "the prompt template in <code>data/prompt/{lang}/niah.txt</code>. "
        "The key is a unique string (e.g. <i>oellmbg00</i>) and the value is an "
        "8-digit number derived from the language index and question index. "
        "The needle is inserted at fractional depth <i>d</i> through the sentence list, "
        "where <i>d</i> is cycled through [0.10, 0.30, 0.50, 0.70, 0.90] across "
        "successive questions.",
        "<b>Question.</b> After the haystack, the model is asked (in the target "
        "language): <i>\"What special magic numbers associated with &lt;key&gt; are "
        "mentioned in the text? Please list all that apply. If no such numbers exist, "
        "please answer &lt;none-word&gt;.\"</i> followed by an answer-format prefix.",
    ]:
        story.append(_p(f"• {bullet}", s["bullet"]))

    story.append(_p("2.2  Scoring", s["subsection"]))
    story.append(_p(
        "Responses are evaluated by <code>evaluate.py::compare_numbers()</code>. "
        "The function extracts all multi-digit numbers (length &gt; 1) from the model "
        "response, deduplicates them, and requires:",
        s["body"],
    ))
    for bullet in [
        "The response is non-empty.",
        "No language-specific 'none' word is present (which would indicate the model "
        "believes no needle exists).",
        "The count of extracted numbers equals the count of expected answers (1 for "
        "niah_single).",
        "The set of extracted numbers exactly matches the set of expected answers.",
    ]:
        story.append(_p(f"• {bullet}", s["bullet"]))
    story.append(_p(
        "This means a response is marked <b>incorrect</b> if it contains the right "
        "answer but also includes any extra number (e.g. a distractor). "
        "Accuracy per language is the mean over all questions; the reported "
        "average accuracy is the mean over all 38 languages.",
        s["body"],
    ))

    # ── 3. Models ─────────────────────────────────────────────────────────────
    story.append(_p("3  Models", s["section"]))

    story.append(_p("3.1  Instruction-following models (Ollama backend)", s["subsection"]))
    story.append(_p(
        "Instruction-tuned models are served via a local Ollama instance "
        "(http://127.0.0.1:11434). Generation uses greedy decoding "
        "(temperature=0, top_p=1). All models use the same prompt template "
        "translated into each target language.",
        s["body"],
    ))
    hdr = [_p(h, s["th"]) for h in
           ["Model", "HuggingFace ID", "Type", "Context (words)", "Max new tokens"]]
    rows = [hdr] + [
        [_p(a, s["td_left"]), _p(b, s["td_left"]), _p(c, s["td"]), _p(d, s["td"]), _p(e, s["td"])]
        for a, b, c, d, e in [
            ("Qwen2 0.5B",   "Qwen/Qwen2-0.5B-Instruct", "Instruct",      "~200", "128"),
            ("Qwen2 1.5B",   "Qwen/Qwen2-1.5B-Instruct", "Instruct",      "~200", "128"),
            ("Gemma 4 E4B",  "google/gemma-4-E4B (Q4)",  "Think+Instruct","~2000","1024"),
            ("Gemma 4 E2B",  "google/gemma-4-E2B (Q4)",  "Think+Instruct","~2000","1024"),
            ("Gemma 3 4B",   "google/gemma-3-4b-it (Q4)","Instruct",      "~2000","1024"),
        ]
    ]
    story.append(_tbl(rows, [3.5*cm, 4.5*cm, 2.8*cm, 3.2*cm, 3.2*cm]))
    story.append(Spacer(1, 0.3 * cm))

    story.append(_p("3.2  OpenEuroLLM 2B datamix models (HuggingFace backend)", s["subsection"]))
    story.append(_p(
        "The OpenEuroLLM datamix-2b series are base (non-instruction-tuned) "
        "LlamaForCausalLM models exploring different European/English data ratios. "
        "They are loaded directly via the HuggingFace <code>transformers</code> library "
        "on an NVIDIA L4 GPU (23.7 GB VRAM) in float16. "
        "Because their maximum positional embedding is 2 048 tokens, and the "
        "~2 000-word harder benchmark prompts would exceed this after tokenisation "
        "(approximately 1.3 tokens per word), the context is reduced to "
        "<b>~500 words</b> for these models (~650 content tokens, leaving ~1 400 tokens "
        "for the prompt overhead and answer). "
        "Inputs are hard-truncated to <i>max_position_embeddings − max_new_tokens</i> "
        "if they still exceed the limit. "
        "Generation uses greedy decoding (do_sample=False) with up to 64 new tokens. "
        "Distractors are reduced proportionally (5 per ~500-word context). "
        "As base models, they do not follow the structured NIAH format and are "
        "expected to produce low scores; the results are reported for completeness.",
        s["body"],
    ))
    hdr2 = [_p(h, s["th"]) for h in
            ["Model", "EU data", "EN data", "Context (words)", "Max new tokens"]]
    rows2 = [hdr2] + [
        [_p(a, s["td_left"]), _p(b, s["td"]), _p(c, s["td"]), _p("~500", s["td"]), _p("64", s["td"])]
        for a, b, c in [
            ("openeurollm/datamix-2b-en",    "0%",  "100%"),
            ("openeurollm/datamix-2b-50-50",  "50%", "50%"),
            ("openeurollm/datamix-2b-60-40",  "60%", "40%"),
            ("openeurollm/datamix-2b-70-30",  "70%", "30%"),
            ("openeurollm/datamix-2b-80-20",  "80%", "20%"),
            ("openeurollm/datamix-2b-90-10",  "90%", "10%"),
        ]
    ]
    story.append(_tbl(rows2, [6.5*cm, 2*cm, 2*cm, 3.3*cm, 3.3*cm]))
    story.append(Spacer(1, 0.3 * cm))

    # ── 4. Results ────────────────────────────────────────────────────────────
    story.append(_p("4  Results", s["section"]))

    story.append(_p("4.1  Summary across all models", s["subsection"]))

    # Build summary table from available CSVs
    hdr = [_p(h, s["th"]) for h in
           ["Model", "Benchmark", "Context", "Tok. budget", "Avg acc.", "100%", "50%", "0%"]]
    summary_rows = [hdr]
    for label, csv_path, ctx_words, num_predict, backend, note in MODELS_META:
        data = load_csv(csv_path)
        if data is None:
            continue
        avg = avg_accuracy(data)
        n100 = sum(1 for r in data.values() if float(r["accuracy"]) >= 1.0)
        n50  = sum(1 for r in data.values() if 0 < float(r["accuracy"]) < 1.0)
        n0   = sum(1 for r in data.values() if float(r["accuracy"]) == 0.0)
        bench = "easy" if ctx_words <= 300 else "harder"
        acc_st = ParagraphStyle("as", fontSize=8, leading=11, fontName="Helvetica-Bold",
                                alignment=TA_CENTER, textColor=acc_color(avg))
        summary_rows.append([
            _p(label, s["td_left"]),
            _p(bench, s["td"]),
            _p(f"~{ctx_words}w", s["td"]),
            _p(str(num_predict), s["td"]),
            Paragraph(f"{avg:.0%}", acc_st),
            _p(str(n100), s["td"]),
            _p(str(n50),  s["td"]),
            _p(str(n0),   s["td"]),
        ])
    # Add datamix rows
    for label, slug, note in DATAMIX_MODELS:
        csv_path = EVAL_ROOT / "full_eval" / f"openeurollm-{slug}" / "summary.csv"
        data = load_csv(csv_path)
        if data is None:
            summary_rows.append([
                _p(label, s["td_left"]),
                _p("harder*", s["td"]),
                _p("~500w", s["td"]),
                _p("64", s["td"]),
                _p("pending", ParagraphStyle("p", fontSize=8, leading=11,
                   fontName="Helvetica-Oblique", alignment=TA_CENTER, textColor=GREY_MED)),
                _p("—", s["td"]), _p("—", s["td"]), _p("—", s["td"]),
            ])
        else:
            avg = avg_accuracy(data)
            n100 = sum(1 for r in data.values() if float(r["accuracy"]) >= 1.0)
            n50  = sum(1 for r in data.values() if 0 < float(r["accuracy"]) < 1.0)
            n0   = sum(1 for r in data.values() if float(r["accuracy"]) == 0.0)
            acc_st = ParagraphStyle("as2", fontSize=8, leading=11, fontName="Helvetica-Bold",
                                    alignment=TA_CENTER, textColor=acc_color(avg))
            summary_rows.append([
                _p(label, s["td_left"]),
                _p("harder*", s["td"]),
                _p("~500w", s["td"]),
                _p("64", s["td"]),
                Paragraph(f"{avg:.0%}", acc_st),
                _p(str(n100), s["td"]),
                _p(str(n50),  s["td"]),
                _p(str(n0),   s["td"]),
            ])
    story.append(_tbl(summary_rows, [4.0*cm, 2.0*cm, 1.8*cm, 2.2*cm, 1.8*cm, 1.4*cm, 1.4*cm, 1.4*cm]))
    story.append(_p(
        "* harder = ~2000-word context for Ollama models, ~500-word context for "
        "HuggingFace datamix models (2048-token context window constraint). "
        "100%/50%/0% = number of languages at that accuracy level.",
        s["caption"],
    ))

    # ── 4.2 Figure ────────────────────────────────────────────────────────────
    story.append(_p("4.2  Per-language figure (instruction-tuned models)", s["subsection"]))
    story.append(_p(
        "Figure 1 shows per-language NIAH accuracy for instruction-tuned models. "
        "Gold labels denote EU official languages; blue labels denote additional "
        "European languages added in this fork. "
        "The datamix base models are omitted from this figure as they score near 0% "
        "across all languages.",
        s["body"],
    ))
    if FIGURE_PATH.exists():
        story.append(Image(str(FIGURE_PATH), width=16.5 * cm, height=19.5 * cm))
        story.append(_p(
            "Figure 1 — NIAH single-needle accuracy for five instruction-tuned "
            "model/benchmark combinations. Each horizontal bar represents one language. "
            "Orange = Gemma 4 E4B (harder), yellow = Gemma 4 E2B (harder), "
            "green = Gemma 3 4B (harder).",
            s["caption"],
        ))

    story.append(PageBreak())

    # ── 4.3 Per-language table (Gemma 4 E4B) ─────────────────────────────────
    story.append(_p("4.3  Per-language results — Gemma 4 E4B (harder benchmark)", s["subsection"]))
    story.append(_p(
        "Table 2 shows per-language accuracy for the best-performing model "
        "(Gemma 4 E4B, ~2000-word context, 1024-token budget). "
        "Haystack source: <b>book</b> = real novel text, <b>synth</b> = synthetic "
        "noun sentences.",
        s["body"],
    ))

    e4b = load_csv(EVAL_ROOT / "full_eval" / "gemma4" / "summary.csv")
    e2b = load_csv(EVAL_ROOT / "full_eval" / "gemma4-e2b" / "summary.csv")
    g3  = load_csv(EVAL_ROOT / "full_eval" / "gemma3-4b" / "summary.csv")

    if e4b:
        eu_langs  = sorted((l for l in e4b if l in EU_OFFICIAL),     key=lambda l: LANG_NAMES[l])
        ext_langs = sorted((l for l in e4b if l not in EU_OFFICIAL),  key=lambda l: LANG_NAMES[l])

        hdr = [_p(h, s["th"]) for h in
               ["Language", "Code", "Group", "Haystack",
                "E4B acc.", "E2B acc.", "G3-4B acc."]]

        def lang_rows(lang_list):
            out = []
            for lang in lang_list:
                r = e4b[lang]
                group = "EU" if lang in EU_OFFICIAL else "extra"
                out.append([
                    _p(LANG_NAMES[lang], s["td_left"]),
                    _p(lang, s["td"]),
                    _p(group, s["td"]),
                    _p(r["haystack"][:5], s["td"]),
                    acc_cell(r["accuracy"], s),
                    acc_cell(e2b[lang]["accuracy"], s) if e2b and lang in e2b else _p("—", s["td"]),
                    acc_cell(g3[lang]["accuracy"], s)  if g3 and lang in g3   else _p("—", s["td"]),
                ])
            return out

        tbl_rows = [hdr] + lang_rows(eu_langs) + lang_rows(ext_langs)
        story.append(_tbl(tbl_rows, [4.2*cm, 1.3*cm, 1.5*cm, 1.8*cm, 2.2*cm, 2.2*cm, 2.2*cm]))
        story.append(_p(
            "Table 2 — Green = 100%, orange = 50%, red = 0%. "
            "E4B = Gemma 4 E4B, E2B = Gemma 4 E2B, G3-4B = Gemma 3 4B.",
            s["caption"],
        ))

    story.append(PageBreak())

    # ── 5. Analysis ───────────────────────────────────────────────────────────
    story.append(_p("5  Analysis", s["section"]))

    story.append(_p("5.1  MoE architecture advantage", s["subsection"]))
    story.append(_p(
        "Gemma 4 E4B (91%) and E2B (86%) both outperform the dense Gemma 3 4B (64%) "
        "by a significant margin despite having similar or smaller active parameter "
        "counts. The E4B and E2B are Mixture-of-Experts models where the active "
        "parameter count during inference is 4B and 2B respectively, but the total "
        "model capacity is much larger. This suggests MoE architecture is particularly "
        "beneficial for multilingual retrieval tasks.",
        s["body"],
    ))

    story.append(_p("5.2  Failure modes in instruction-tuned models", s["subsection"]))
    for bullet in [
        "<b>Distractor confusion (Norwegian, Serbian — Gemma 4 E4B).</b> "
        "The model correctly retrieves the needle number but also includes one of "
        "the distractor numbers in its answer. The evaluator requires exactly one "
        "number in the response; an extra number causes the question to fail. "
        "This demonstrates the distractors are effective at confusing the model.",
        "<b>Token-budget exhaustion (Irish, Basque, Welsh — Gemma 4 E4B and E2B).</b> "
        "Gemma 4 models perform internal chain-of-thought reasoning before outputting "
        "the answer. At 1 024 max new tokens, some prompts exhaust the budget during "
        "reasoning, producing an empty response. "
        "No wrong answers (wrong number given) were observed for any language with "
        "either Gemma 4 model — indicating adequate training coverage for all 38 "
        "OELLM languages.",
        "<b>Format errors (Gemma 3 4B — multiple languages).</b> "
        "The smaller dense model makes more errors, including returning multiple "
        "numbers or malformed answers, especially for lower-resource languages "
        "(Greek, Lithuanian, Romanian, Macedonian, Serbian — all 0%).",
    ]:
        story.append(_p(f"• {bullet}", s["bullet"]))

    story.append(_p("5.3  OpenEuroLLM datamix base models", s["subsection"]))
    story.append(_p(
        "The datamix-2b models are trained for text completion, not instruction "
        "following. When presented with the structured NIAH prompt they generate "
        "repetitive or incoherent continuations rather than extracting and reporting "
        "the needle value. "
        "The evaluation is run with a shorter ~500-word context to respect the "
        "2 048-token positional embedding limit; even so, scores are near 0% across "
        "all data mixture ratios. "
        "This is consistent with the general behaviour of base LLMs on instruction "
        "tasks: without supervised fine-tuning or RLHF the models cannot reliably "
        "follow structured prompts. "
        "The datamix results are included as a baseline for comparison once "
        "instruction-tuned variants of these models become available.",
        s["body"],
    ))

    story.append(_p("5.4  Haystack source effect", s["subsection"]))
    story.append(_p(
        "For Gemma 4 E4B, there is no clear accuracy gap between languages using "
        "real book haystacks and those using synthetic noun-sentence haystacks. "
        "All 12 synthetic-haystack languages that score below 100% do so due to "
        "distractor confusion or token-budget issues, not haystack quality. "
        "This validates the synthetic fallback as an adequate substitute when "
        "real text is unavailable.",
        s["body"],
    ))

    # ── 6. Reproducing ────────────────────────────────────────────────────────
    story.append(_p("6  Reproducing the Results", s["section"]))
    story.append(_p(
        "All code is in the <code>main</code> branch of "
        "<code>BirgerMoell/OneRuler-OELLM</code>. "
        "Install <a href='https://ollama.com'>Ollama</a> and pull the desired model, "
        "then run:",
        s["body"],
    ))
    story.append(_p(
        "# Harder benchmark — Gemma 4 E4B (recommended)<br/>"
        "ollama pull gemma4<br/>"
        "python scripts/run_oellm_mini_eval.py \\ <br/>"
        "&nbsp;&nbsp;--model gemma4 --num-predict 1024 \\ <br/>"
        "&nbsp;&nbsp;--output-dir eval_results/full_eval<br/><br/>"
        "# Gemma 4 E2B<br/>"
        "ollama pull gemma4:e2b<br/>"
        "python scripts/run_oellm_mini_eval.py \\ <br/>"
        "&nbsp;&nbsp;--model gemma4:e2b --num-predict 1024 \\ <br/>"
        "&nbsp;&nbsp;--output-dir eval_results/full_eval<br/><br/>"
        "# OpenEuroLLM datamix models (HuggingFace, GPU required)<br/>"
        "python scripts/run_oellm_mini_eval.py \\ <br/>"
        "&nbsp;&nbsp;--backend huggingface \\ <br/>"
        "&nbsp;&nbsp;--model openeurollm/datamix-2b-80-20 \\ <br/>"
        "&nbsp;&nbsp;--context-words 500 --num-predict 64 \\ <br/>"
        "&nbsp;&nbsp;--output-dir eval_results/full_eval<br/><br/>"
        "# Oracle smoke-test (no model needed, ~30 s)<br/>"
        "python scripts/run_oellm_mini_eval.py --backend oracle --questions 2",
        s["code"],
    ))
    story.append(_p(
        "Regenerate the figure and report after new results are available:",
        s["body"],
    ))
    story.append(_p(
        "uv run --with matplotlib python3 scripts/make_eval_figure.py<br/>"
        "uv run --with reportlab python3 scripts/make_report.py",
        s["code"],
    ))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5 * cm))
    story.append(_hrule_thin())
    story.append(Spacer(1, 0.2 * cm))
    story.append(_p(
        "Generated by <code>scripts/make_report.py</code> · "
        "OneRuler: Kim et al., 2025 (arXiv:2503.01996) · "
        "OpenEuroLLM: huggingface.co/openeurollm",
        s["footer"],
    ))


def main():
    story: list = []
    s = build_styles()
    build_doc(story, s)

    def page_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(DARK_BG)
        canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
        canvas.restoreState()

    frame = Frame(1.5 * cm, 1.5 * cm, 18 * cm, 26.7 * cm, id="main")
    tmpl  = PageTemplate(id="dark", frames=[frame], onPage=page_bg)
    doc   = BaseDocTemplate(
        str(OUT_PATH), pagesize=A4, pageTemplates=[tmpl],
        leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm,  bottomMargin=1.5*cm,
    )
    doc.build(story)
    print(f"Saved {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
