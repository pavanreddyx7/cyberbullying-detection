"""
Emoji Abuse / Bullying Analyser
Scores a message based on threatening, violent, or offensive emoji usage.
"""

import re
import emoji

# ── Threat-level emoji dictionary ────────────────────────────────────────────
DANGEROUS = {
    "🔪","🗡️","⚔️","🪓","🔫","💣","🧨","☠️","💀","🖕",
    "🤬","😡","😤","👊","🤛","🤜","👋","🖐️","✊","💢",
    "🩸","🔥","💥","🤮","🤢","🤡","👿","😈","🤯",
}

OFFENSIVE = {
    "💩","🐷","🐖","🐮","🐄","🐀","🐁","🦗","🐛","🦟",
    "😒","🙄","😑","😐","🫤","😶","😏","😬","🫠",
    "🤥","😈","👎","🖕","🤦","🤦‍♂️","🤦‍♀️",
}

POSITIVE = {
    "❤️","💖","💗","💓","💞","💕","💝","🥰","😍","😘",
    "😊","😁","😄","😀","🤗","🥳","🎉","🎊","✨","🌟",
    "👍","👏","🙌","🤝","🫶","💪","🌈","🌸","🌺","🌻",
    "😂","🤣","😆","😋","😛","🤩","🥹","☺️","😇",
}

WARNING_COMBOS = [
    ({"🔪", "😡"}, "Knife + anger emoji combination detected"),
    ({"💀", "😡"}, "Death + anger emoji combination detected"),
    ({"🔫", "😡"}, "Gun + anger emoji combination detected"),
    ({"☠️", "🤬"}, "Skull + rage emoji detected"),
    ({"🖕", "😡"}, "Offensive gesture + anger detected"),
    ({"🔪", "🩸"}, "Knife + blood emoji — violent content"),
    ({"💣", "💥"}, "Bomb + explosion emoji combination"),
]


def extract_emojis(text: str) -> list[str]:
    return [ch["emoji"] for ch in emoji.emoji_list(text)]


def analyse_emojis(text: str) -> dict:
    found        = extract_emojis(text)
    found_set    = set(found)

    danger_hits  = found_set & DANGEROUS
    offense_hits = found_set & OFFENSIVE
    positive_hits = found_set & POSITIVE

    # Check dangerous combos
    triggered_combos = [
        msg for combo, msg in WARNING_COMBOS
        if combo.issubset(found_set)
    ]

    # Scoring
    danger_score  = min(len(danger_hits) * 35, 95)
    offense_score = min(len(offense_hits) * 20, 70)
    positive_score = min(len(positive_hits) * 15, 80)

    if triggered_combos:
        danger_score = min(danger_score + 25, 99)

    net_score = max(0, danger_score + offense_score - positive_score)

    if net_score >= 60 or triggered_combos:
        label = "Abusive"
    elif net_score >= 30:
        label = "Warning"
    else:
        label = "Safe"

    # Clean text (replace emojis with their names for DistilBERT)
    clean = emoji.demojize(text, delimiters=(" :", ": "))

    return {
        "emoji_label"     : label,
        "emoji_score"     : round(net_score, 1),
        "found_emojis"    : found,
        "danger_emojis"   : list(danger_hits),
        "offense_emojis"  : list(offense_hits),
        "positive_emojis" : list(positive_hits),
        "combo_warnings"  : triggered_combos,
        "demojized_text"  : clean,
    }


def combined_predict(text: str, bert_predictor) -> dict:
    """
    Combines DistilBERT text prediction + emoji analysis into one verdict.
    bert_predictor: callable(text) -> {"prediction": ..., "confidence": ...}
    """
    emoji_result = analyse_emojis(text)

    # Run BERT on demojized text for better understanding
    bert_input  = emoji_result["demojized_text"] if emoji_result["found_emojis"] else text
    bert_result = bert_predictor(bert_input)

    bert_abusive  = bert_result["prediction"] == "Abusive"
    emoji_abusive = emoji_result["emoji_label"] in ("Abusive", "Warning")

    # Combine scores
    bert_conf  = bert_result["confidence"] / 100
    emoji_conf = emoji_result["emoji_score"] / 100

    if bert_abusive and emoji_abusive:
        final_label = "Abusive"
        final_conf  = round(min((bert_conf * 0.6 + emoji_conf * 0.4) * 100, 99), 2)
        reason      = "Both text meaning and emojis indicate abuse."
    elif bert_abusive:
        final_label = "Abusive"
        final_conf  = round(bert_result["confidence"], 2)
        reason      = "Abusive language detected in text."
    elif emoji_abusive:
        final_label = "Warning"
        final_conf  = round(emoji_result["emoji_score"], 2)
        reason      = "Threatening or offensive emojis detected."
    else:
        final_label = "Safe"
        final_conf  = round((1 - bert_conf) * 100, 2)
        reason      = "No abuse detected in text or emojis."

    return {
        "text"            : text,
        "final_label"     : final_label,
        "final_confidence": final_conf,
        "reason"          : reason,
        "bert_prediction" : bert_result["prediction"],
        "bert_confidence" : bert_result["confidence"],
        **emoji_result,
        "model"           : "DistilBERT + Emoji Analyser",
    }
