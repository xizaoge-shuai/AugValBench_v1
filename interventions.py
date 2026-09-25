import random
from typing import Dict, Any

from utils import safe_text


def clean_text(m: Dict[str, Any]) -> str:
    parts = []

    if m.get("main_category"):
        parts.append(f"Category: {m['main_category']}")
    if m.get("categories"):
        parts.append(f"Subcategories: {m['categories']}")
    if m.get("features"):
        parts.append(f"Features: {m['features']}")
    if m.get("description"):
        parts.append(f"Description: {m['description']}")
    if m.get("store"):
        parts.append(f"Brand/Store: {m['store']}")

    return "\n".join(parts)[:1100]


def redundant_text(m: Dict[str, Any]) -> str:
    title = safe_text(m.get("title", ""), 240)
    cat = safe_text(m.get("main_category", ""), 120)

    return (
        f"Product name again: {title}\n"
        f"Catalog category: {cat}\n"
        f"Repeated catalog note: {title} belongs to {cat}."
    )[:700]


def partial_text(m: Dict[str, Any]) -> str:
    cat = safe_text(m.get("main_category", ""), 120)
    feat = safe_text(m.get("features", ""), 180)
    desc = safe_text(m.get("description", ""), 220)

    return (
        f"Category: {cat}\n"
        f"Partial features: {feat}\n"
        f"Partial description: {desc[:100]}"
    )[:450]


def choose_other_item(item_id, meta, rng):
    cur_cat = meta[item_id].get("main_category", "")

    same = [
        x for x, m in meta.items()
        if x != item_id
        and m.get("main_category", "") == cur_cat
        and m.get("title")
    ]

    if same:
        return rng.choice(same)

    fallback = [
        x for x, m in meta.items()
        if x != item_id and m.get("title")
    ]
    return rng.choice(fallback)


def augmentation_for(item_id, condition, meta, rng):
    m = meta[item_id]

    if condition == "base":
        return ""
    if condition == "clean":
        return clean_text(m)
    if condition == "redundant":
        return redundant_text(m)
    if condition == "partial":
        return partial_text(m)

    other = choose_other_item(item_id, meta, rng)
    other_meta = meta[other]

    if condition == "irrelevant":
        return (
            "Unrelated catalog note (not necessarily about this candidate):\n"
            + clean_text(other_meta)
        )[:1100]

    if condition == "conflict":
        return clean_text(other_meta)

    raise ValueError(condition)
