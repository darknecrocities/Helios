"""Adversarial stress-testing suite for Helios Byte-Level BPE and Unigram tokenizers.

Validates lossless roundtrips across tricky Unicode, complex emojis, raw bytes,
boundary strings, performance scaling, zero OOV guarantee, and Viterbi optimality.
"""

from __future__ import annotations

import random
import time

import pytest

from helios.tokenizer.base import BaseTokenizer
from helios.tokenizer.bpe import ByteLevelBPETokenizer
from helios.tokenizer.unigram import UnigramTokenizer


@pytest.fixture(scope="module")
def adversarial_corpus() -> list[str]:
    """Rich multilingual and code corpus for training adversarial test tokenizers."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Nagpapakapagod ang mga manggagawa sa buong kapuluan ng Pilipinas.",
        "Nakikipagkapwa-tao at nagpapakadalubhasa upang maglingkod sa bayan.",
        "def compute_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:\n    return out\n",
        "你好世界！ 这是一个开源的基础大语言模型，具有创新的分层记忆架构。",
        "مرحبا بالعالم! يدعم نموذج هيليوس الذاكرة الهرمية واسترجاع المعرفة بدقة.",
        "नमस्ते दुनिया! हीलियोस ओपन-वेट लैंग्वेज मॉडल में आपका स्वागत है।",
        "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์และตัวอักษรที่ซับซ้อน เช่น น้ำ ผู้หญิง",
        "Языковая модель Гелиос с многоуровневой архитектурой памяти и защитой.",
        "Helios 🚀✨: 👨‍👩‍👧‍👦 + 🇵🇭 + 1️⃣ = ❤️! 👍🏽 👩🏻‍🔬 🧙🏿‍♂️ 🏳️‍🌈 🏴󠁧󠁢󠁳󠁣󠁴󠁿",
        "Special symbols: !@#$%^&*()_+-=[]{}|;:,.<>/?`~ \t\r\n\x00\x1f\x7f",
    ]


@pytest.fixture(
    scope="module",
    params=["base_bpe", "trained_bpe", "base_unigram", "trained_unigram"],
)
def target_tokenizer(
    request: pytest.FixtureRequest, adversarial_corpus: list[str]
) -> BaseTokenizer:
    """Fixture providing base and trained variants of both BPE and Unigram tokenizers."""
    param = request.param
    if param == "base_bpe":
        return ByteLevelBPETokenizer()
    elif param == "trained_bpe":
        tok = ByteLevelBPETokenizer()
        tok.train(adversarial_corpus, vocab_size=350, min_frequency=1)
        return tok
    elif param == "base_unigram":
        return UnigramTokenizer()
    elif param == "trained_unigram":
        tok = UnigramTokenizer()
        tok.train(adversarial_corpus, vocab_size=350, min_frequency=1)
        return tok
    raise ValueError(f"Unknown tokenizer parameter: {param}")


# ============================================================================
# 1. Tricky Unicode Invariant Tests
# ============================================================================

TRICKY_UNICODE_CASES: dict[str, str] = {
    "cjk_simplified": "氦模型是一个开源的基础大语言模型，具有创新的分层记忆架构。",
    "cjk_traditional": "氦模型是一個開源的基礎大語言模型，具有創新的分層記憶架構。",
    "cjk_japanese": "日本語のテスト：ひらがな、カタカナ、漢字（東京、京都、大阪）。",
    "cjk_korean_hangul": "한국어 텍스트: 한글 음절과 낱자 테스트입니다. 안녕하세요!",
    "arabic_rtl_sentence": "نموذج هيليوس للذكاء الاصطناعي يدعم الذاكرة الهرمية واسترجاع المعرفة.",
    "arabic_vocalized_tashkeel": "اللُّغَةُ العَرَبِيَّةُ لُغَةٌ جَمِيلَةٌ ذَاتُ قَوَاعِدَ دَقِيقَةٍ.",
    "arabic_bidi_mixed": "Order status #98765: تم تأكيد الدفع بنجاح for client ID: 4421.",
    "persian_farsi": "مدل زبانی هلیوس برای پردازش زبان طبیعی فارسی و معماری چندسطحی حافظه.",
    "urdu_nastaliq": "ہیلیوس اوپن ویٹ فاؤنڈیشن ماڈل اردو زبان کی پروسیسنگ اور یادداشت کے لیے۔",
    "devanagari_hindi": "नमस्ते दुनिया! हीलियोस ओपन-वेट लैंग्वेज मॉडल में आपका स्वागत है।",
    "devanagari_conjuncts": "क्षत्रिय, त्रिशूल, ज्ञानी, विद्या, श्रीमद्भगवद्गीता",
    "devanagari_nukta": "क़लम, ख़रगोश, ग़ज़ल, ज़मीन, फ़िल्म, ड़, ढ़",
    "thai_tone_and_vowels": "ภาษาไทยเป็นภาษาที่มีวรรณยุกต์ เช่น น้ำ ผู้หญิง กรุงเทพมหานคร",
    "cyrillic_russian": "Языковая модель Гелиос с многоуровневой архитектурой памяти.",
    "cyrillic_ukrainian_apostrophe": "Українська мова: розв'язати м'ясо пам'ять під'їзд.",
    "cyrillic_serbian_ligatures": "Српски језик: ћирилица и латиница (Ђ, Љ, Њ, Ћ, Џ).",
    "combining_nfc_nfd": "NFC: \u00e9\u00f1\u00fc vs NFD: e\u0301n\u0303u\u0308",
    "vietnamese_stacked_diacritics": "Tiếng Việt có nhiều dấu phức tạp: ế, ồ, ử, ẵ, ặ, ỗ, ỹ",
    "zalgo_extreme_combining": (
        "H\u0300\u0301\u0302\u0303\u0304\u0305e\u0306\u0307\u0308\u0309\u030a"
        "l\u030b\u030c\u030d\u030e\u030fl\u0310\u0311\u0312\u0313\u0314o\u0315\u0316\u0317"
    ),
}


@pytest.mark.unit
@pytest.mark.tier5
@pytest.mark.parametrize("case_name,text", list(TRICKY_UNICODE_CASES.items()))
def test_tricky_unicode_roundtrip(
    target_tokenizer: BaseTokenizer, case_name: str, text: str
) -> None:
    """Verify exact lossless roundtrip across tricky Unicode scripts and combining marks."""
    encoded = target_tokenizer.encode(text, add_special_tokens=False)
    decoded = target_tokenizer.decode(encoded)
    assert decoded == text, f"Failed on {case_name}: expected {text!r}, got {decoded!r}"


# ============================================================================
# 2. Complex Emoji Invariant Tests
# ============================================================================

COMPLEX_EMOJI_CASES: dict[str, str] = {
    "skin_tone_modifiers": "👍 👍🏻 👍🏼 👍🏽 👍🏾 👍🏿 👩🏻‍🔬 🧙🏿‍♂️ 🧗🏽‍♀️ 🏃🏿‍♂️",
    "zwj_family_variants": "👨‍👩‍👦 👨‍👩‍👧 👨‍👩‍👧‍👦 👨‍👨‍👦 👩‍👩‍👧‍👧 🧑‍🧒 🧑‍🧑‍🧒‍🧒",
    "zwj_couples_and_relationships": "👩‍❤️‍👨 👩‍❤️‍💋‍👨 🧑‍🤝‍🧑 💏 💑",
    "zwj_special_symbols": "🏳️‍🌈 🏳️‍⚧️ 👁️‍🗨️ 🐈‍⬛ 🐻‍❄️ 🐦‍⬛",
    "flags_regional_indicators": "🇵🇭 🇺🇸 🇯🇵 🇩🇪 🇫🇷 🇧🇷 🇿🇦 🇦🇺 🇰🇷 🇨🇦 🇬🇧 🇮🇳",
    "flags_subdivision_tags": "🏴󠁧󠁢󠁳󠁣󠁴󠁿 🏴󠁧󠁢󠁥󠁮󠁧󠁿 🏴󠁧󠁢󠁷󠁬󠁳󠁿",
    "keycaps_full_sequence": "0️⃣ 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣ 8️⃣ 9️⃣ #️⃣ *️⃣ 🔟",
    "dense_mixed_emoji_text": "Helios 🚀 to Mars 🔴: 👨‍🚀 + 👩‍🚀 = 🛸! Score: #️⃣1️⃣ at 100% 💯 👍🏽",
}


@pytest.mark.unit
@pytest.mark.tier5
@pytest.mark.parametrize("case_name,text", list(COMPLEX_EMOJI_CASES.items()))
def test_complex_emoji_roundtrip(
    target_tokenizer: BaseTokenizer, case_name: str, text: str
) -> None:
    """Verify lossless roundtrip across skin-tones, ZWJ sequences, flags, and keycaps."""
    encoded = target_tokenizer.encode(text, add_special_tokens=False)
    decoded = target_tokenizer.decode(encoded)
    assert decoded == text, f"Failed on {case_name}: expected {text!r}, got {decoded!r}"


# ============================================================================
# 3. Raw Byte Stress Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.tier5
def test_all_256_individual_bytes(target_tokenizer: BaseTokenizer) -> None:
    """Verify all 256 individual byte values (0..255) roundtrip losslessly."""
    for b in range(256):
        char = chr(b)
        encoded = target_tokenizer.encode(char, add_special_tokens=False)
        decoded = target_tokenizer.decode(encoded)
        assert decoded == char, f"Byte {b} ({char!r}) failed: got {decoded!r}"


@pytest.mark.unit
@pytest.mark.tier5
def test_binary_executable_headers(target_tokenizer: BaseTokenizer) -> None:
    """Verify binary executable and container file headers preserved via Latin-1."""
    headers = {
        "ELF": b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00>\x00",
        "PNG": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x04\x00\x00\x00\x03\x00\x08\x06\x00",
        "PDF": b"%PDF-1.7\r\n%\xe2\xe3\xcf\xd3\r\n",
        "ZIP": b"PK\x03\x04\x14\x00\x00\x00\x08\x00Z|kX\x00\x00\x00\x00\x00\x00\x00\x00",
        "GZIP": b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00",
        "Mach-O": b"\xcf\xfa\xed\xfe\x07\x00\x00\x01\x03\x00\x00\x00\x02\x00\x00\x00",
    }
    for name, raw_bytes in headers.items():
        text = raw_bytes.decode("latin-1")
        encoded = target_tokenizer.encode(text, add_special_tokens=False)
        decoded = target_tokenizer.decode(encoded)
        assert decoded == text, f"Binary header {name} failed: got {decoded!r}"


@pytest.mark.unit
@pytest.mark.tier5
def test_random_byte_sequences(target_tokenizer: BaseTokenizer) -> None:
    """Verify 50 pseudo-random byte buffers decoded via Latin-1 roundtrip losslessly."""
    rng = random.Random(42)
    for _ in range(50):
        length = rng.choice([1, 2, 7, 16, 64, 256, 512])
        raw_bytes = bytes(rng.randint(0, 255) for _ in range(length))
        text = raw_bytes.decode("latin-1")
        encoded = target_tokenizer.encode(text, add_special_tokens=False)
        decoded = target_tokenizer.decode(encoded)
        assert decoded == text, "Random byte stream mismatch"


# ============================================================================
# 4. Boundary String Tests
# ============================================================================

BOUNDARY_CASES: dict[str, str] = {
    "empty": "",
    "single_space": " ",
    "single_tab": "\t",
    "single_newline": "\n",
    "single_cr": "\r",
    "crlf": "\r\n",
    "massive_spaces_1000": " " * 1000,
    "massive_tabs_500": "\t" * 500,
    "massive_mixed_whitespace": " \t\r\n  \r \n\t " * 100,
    "nbsp": "Hello\u00a0World\u00a0Helios",
    "zero_width_space": "Hello\u200bWorld\u200bHelios",
    "zero_width_non_joiner": "Hello\u200cWorld\u200cHelios",
    "zero_width_joiner_standalone": "Hello\u200dWorld\u200dHelios",
    "narrow_nbsp": "Hello\u202fWorld\u202fHelios",
    "cjk_ideographic_space": "你好\u3000世界",
    "bom_character": "\ufeffHelios with BOM",
    "unicode_quad_spaces": "".join(chr(c) for c in range(0x2000, 0x200B)),
    "control_nul": "Hello\x00World",
    "control_all_c0": "".join(chr(c) for c in range(32)),
    "control_del": "Hello\x7fWorld",
    "control_all_c1": "".join(chr(c) for c in range(0x80, 0xA0)),
    "ansi_escape_sequences": "\x1b[31;1mRed Bold\x1b[0m \x1b[2J\x1b[H",
    "shifted_byte_space": "".join(chr(c) for c in range(256, 324)),
}


@pytest.mark.unit
@pytest.mark.tier5
@pytest.mark.parametrize("case_name,text", list(BOUNDARY_CASES.items()))
def test_boundary_strings_roundtrip(
    target_tokenizer: BaseTokenizer, case_name: str, text: str
) -> None:
    """Verify extreme boundary strings (spaces, zero-width, controls, shifted bytes)."""
    # Without special tokens
    encoded = target_tokenizer.encode(text, add_special_tokens=False)
    decoded = target_tokenizer.decode(encoded)
    assert decoded == text, f"Failed on boundary {case_name}: expected {text!r}, got {decoded!r}"

    # With special tokens (skip_special_tokens=True)
    enc_sp = target_tokenizer.encode(text, add_special_tokens=True)
    dec_sp = target_tokenizer.decode(enc_sp, skip_special_tokens=True)
    err_msg = f"Failed with special tokens on {case_name}: expected {text!r}, got {dec_sp!r}"
    assert dec_sp == text, err_msg


# ============================================================================
# 5. Performance & Scaling Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.tier5
def test_scaling_long_text_100k(target_tokenizer: BaseTokenizer) -> None:
    """Verify performance and roundtrip fidelity on 100k+ character document."""
    seed = "Helios 10M-40M open-weight model. Ang bilis at memory ay walang kapantay! 你好世界！\n"
    repeat_count = (120000 // len(seed)) + 1
    doc_100k = (seed * repeat_count)[:120000]

    t0 = time.perf_counter()
    tokens = target_tokenizer.encode(doc_100k, add_special_tokens=False)
    t_enc = time.perf_counter() - t0

    t0 = time.perf_counter()
    decoded = target_tokenizer.decode(tokens)
    t_dec = time.perf_counter() - t0

    assert decoded == doc_100k, "Long document roundtrip mismatch"
    assert t_enc < 1.0, f"Encoding 120k chars took {t_enc:.3f}s (budget: 1.0s)"
    assert t_dec < 1.0, f"Decoding 120k chars took {t_dec:.3f}s (budget: 1.0s)"


@pytest.mark.unit
@pytest.mark.tier5
def test_scaling_deeply_repeated_tokens(target_tokenizer: BaseTokenizer) -> None:
    """Verify scaling on deeply repeated single character ('a' * 50,000)."""
    text = "a" * 50000
    t0 = time.perf_counter()
    tokens = target_tokenizer.encode(text, add_special_tokens=False)
    t_enc = time.perf_counter() - t0

    decoded = target_tokenizer.decode(tokens)
    assert decoded == text, "Repeated character roundtrip mismatch"
    assert t_enc < 0.5, f"Encoding 50k 'a' took {t_enc:.3f}s (budget: 0.5s)"


# ============================================================================
# 6. Zero Out-of-Vocabulary (<unk>) Guarantee Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.tier5
def test_zero_oov_guarantee_pua_and_unassigned(target_tokenizer: BaseTokenizer) -> None:
    """Verify that Private Use Areas and unassigned codepoints never produce <unk>."""
    adversarial_inputs = [
        "PUA Area A: \ue000 \ue001 \uf8ff",
        "PUA Area B: \U000f0000 \U000ffffd \U00100000 \U0010fffd",
        "Unassigned: \u0378 \u0379 \u0380 \u0381 \u0530 \u0890",
        "Musical & Math: 𝄞 𝄢 𝔄 𝔅 𝔇 𝕬 𝕭 𝕮 ∰ ∱ ∲ ∳",
    ]
    for inp in adversarial_inputs:
        tokens = target_tokenizer.encode(inp, add_special_tokens=False)
        for tid in tokens:
            assert tid in target_tokenizer.inv_vocab, f"Token ID {tid} not in vocabulary"
            tok_str = target_tokenizer.inv_vocab[tid]
            assert tok_str != "<unk>", f"Token ID {tid} produced <unk>"
        decoded = target_tokenizer.decode(tokens)
        assert decoded == inp, f"Roundtrip failed on PUA/unassigned: {inp!r}"


# ============================================================================
# 7. Unigram Viterbi Conflicting Subword Candidate Tests
# ============================================================================


@pytest.mark.unit
@pytest.mark.tier5
def test_viterbi_greedy_trap_avoidance() -> None:
    """Verify Viterbi DP selects globally optimal path over locally greedy choice.

    Word: 'prefixsuffix'
    Candidate 'fixsuf' has the highest individual score (-0.5), but choosing it
    forces 'pre' (-4.0) and 'fix' (-4.0) for a total of -8.5.
    The partition ['prefix', 'suffix'] has scores -1.5 + -1.5 = -3.0.
    Viterbi dynamic programming must choose ['prefix', 'suffix'].
    """
    uni = UnigramTokenizer()

    def to_byte_chars(s: str) -> str:
        return "".join(uni.byte_encoder[b] for b in s.encode("utf-8"))

    uni.scores[to_byte_chars("pre")] = -4.0
    uni.scores[to_byte_chars("prefix")] = -1.5
    uni.scores[to_byte_chars("fixsuf")] = -0.5
    uni.scores[to_byte_chars("suffix")] = -1.5
    uni.scores[to_byte_chars("fix")] = -4.0
    uni.max_subword_len = max(len(tok) for tok in uni.scores)

    seg = uni.viterbi_segment(to_byte_chars("prefixsuffix"))
    subwords = [bytes(uni.byte_decoder[c] for c in s).decode("utf-8") for s in seg]
    assert subwords == ["prefix", "suffix"], f"Greedy trap failed: got {subwords}"


@pytest.mark.unit
@pytest.mark.tier5
def test_viterbi_longer_vs_shorter_tradeoff() -> None:
    """Verify Viterbi handles tradeoffs where shorter subwords beat longer and vice versa."""
    uni = UnigramTokenizer()

    def to_byte_chars(s: str) -> str:
        return "".join(uni.byte_encoder[b] for b in s.encode("utf-8"))

    # Case A: Two short subwords (-1.0 + -1.0 = -2.0) beat one long subword (-5.0)
    uni.scores[to_byte_chars("cat")] = -1.0
    uni.scores[to_byte_chars("dog")] = -1.0
    uni.scores[to_byte_chars("catdog")] = -5.0
    uni.max_subword_len = max(len(tok) for tok in uni.scores)

    seg_a = uni.viterbi_segment(to_byte_chars("catdog"))
    sub_a = [bytes(uni.byte_decoder[c] for c in s).decode("utf-8") for s in seg_a]
    assert sub_a == ["cat", "dog"], f"Expected ['cat', 'dog'], got {sub_a}"

    # Case B: One long subword (-1.0) beats two short subwords (-4.0 + -4.0 = -8.0)
    uni.scores[to_byte_chars("sun")] = -4.0
    uni.scores[to_byte_chars("flower")] = -4.0
    uni.scores[to_byte_chars("sunflower")] = -1.0
    uni.max_subword_len = max(len(tok) for tok in uni.scores)

    seg_b = uni.viterbi_segment(to_byte_chars("sunflower"))
    sub_b = [bytes(uni.byte_decoder[c] for c in s).decode("utf-8") for s in seg_b]
    assert sub_b == ["sunflower"], f"Expected ['sunflower'], got {sub_b}"


@pytest.mark.unit
@pytest.mark.tier5
def test_viterbi_partial_subword_with_byte_fallback() -> None:
    """Verify reachability and fallback to byte tokens when intermediate subword is missing."""
    uni = UnigramTokenizer()

    def to_byte_chars(s: str) -> str:
        return "".join(uni.byte_encoder[b] for b in s.encode("utf-8"))

    # Word 'abcde': 'ab' (-1.0) and 'de' (-1.0) exist; 'c' has only default base score (-12.0)
    uni.scores[to_byte_chars("ab")] = -1.0
    uni.scores[to_byte_chars("de")] = -1.0
    uni.max_subword_len = max(len(tok) for tok in uni.scores)

    seg = uni.viterbi_segment(to_byte_chars("abcde"))
    subwords = [bytes(uni.byte_decoder[c] for c in s).decode("utf-8") for s in seg]
    assert subwords == ["ab", "c", "de"], f"Expected ['ab', 'c', 'de'], got {subwords}"
