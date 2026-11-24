"""Pure-Python helpers for EcoBite's food-rescue workflow."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Iterable


INJECTION_PATTERNS = (
    r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?)",
    r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above)",
    r"\b(reveal|show|print|repeat)\b.{0,45}\b(system\s+prompt|hidden\s+instructions?)\b",
    r"\b(jailbreak|developer\s+message|act\s+as\s+(?:a\s+)?(?:different|unrestricted))\b",
    r"\bdo\s+not\s+follow\b.{0,45}\binstructions?\b",
)

HARMFUL_FOOD_PATTERNS = (
    r"\b(?:poison|contaminat|spoil|tamper\s+with)\b.{0,40}\b(?:food|meal|donation|person|people)\b",
    r"\bhow\s+to\s+(?:poison|contaminat|spoil)\b",
)

# A scope guard declines clear non-food-rescue requests locally. It makes the
# app's boundary predictable and avoids needless model calls during the demo.
OUT_OF_SCOPE_PATTERNS = (
    r"\b(?:what(?:'s| is) the weather|weather (?:in|for)|weather forecast|forecast for)\b",
    r"\b(?:sports? scores?|who won (?:the )?(?:game|match)|stock price|crypto price)\b",
    r"\b(?:tell me a joke|write (?:me )?(?:a )?poem|write code|debug (?:my )?code)\b",
)


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    message: str | None = None


@dataclass(frozen=True)
class Partner:
    name: str
    recipient_type: str
    accepts: str
    priority: str
    note: str


def inspect_user_input(text: str) -> GuardResult:
    """Reject unsafe, malicious, and clearly out-of-scope input before an API call."""
    normalized = " ".join(text.lower().split())
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in HARMFUL_FOOD_PATTERNS):
        return GuardResult(False, "I can’t help with spoiling, poisoning, or contaminating food. I can help you keep donations safe and usable.")
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS):
        return GuardResult(False, "I can only assist with safe surplus-food donation and rescue logistics. Please share the food type, quantity, and when it was prepared or expires.")
    if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in OUT_OF_SCOPE_PATTERNS):
        return GuardResult(False, "EcoBite is focused on surplus-food donation, safe packaging, and recipient matching. How can I help with a food donation today?")
    return GuardResult(True)


def classify_food(text: str) -> str:
    """Return a conservative donation category from a short inventory description."""
    value = text.lower()
    if any(word in value for word in ("hot", "cooked", "prepared", "baked ziti", "rice", "soup", "catering", "meal", "leftover")):
        return "prepared_hot_meal"
    if any(word in value for word in ("frozen", "ice cream", "frozen meat")):
        return "frozen"
    if any(word in value for word in ("milk", "cheese", "yogurt", "meat", "chicken", "fish", "dairy", "sandwich", "salad", "produce", "fruit", "vegetable")):
        return "refrigerated_perishable"
    if any(word in value for word in ("can", "canned", "rice", "pasta", "flour", "beans", "shelf", "dry", "biscuit")):
        return "shelf_stable"
    return "needs_review"


def safety_checklist(category: str) -> list[str]:
    checklists = {
        "prepared_hot_meal": [
            "Record the **preparation time** and donation handoff time.",
            "Keep food **hot at 60°C / 140°F or above** until handoff, or cool rapidly for refrigerated transport.",
            "Use clean, food-grade, sealed containers; label allergens and reheating needs.",
            "Arrange the earliest possible handoff and confirm recipient capacity first.",
        ],
        "refrigerated_perishable": [
            "Keep items at **5°C / 41°F or below** using a cooler and ice packs.",
            "Label the **use-by date**, allergens, and storage conditions.",
            "Separate raw foods from ready-to-eat food and keep packaging intact.",
            "Confirm cold-storage capacity with the recipient before transport.",
        ],
        "frozen": [
            "Keep items **frozen solid**; use insulated transport and avoid thawing and refreezing.",
            "Label the **freeze date**, contents, and allergens.",
            "Confirm freezer capacity before pickup or delivery.",
        ],
        "shelf_stable": [
            "Check packaging is sealed, dry, and not swollen, leaking, or badly damaged.",
            "Group and label items with their **best-before date** and allergens.",
            "Keep cartons dry and off the floor during staging and transport.",
        ],
    }
    return checklists.get(category, [
        "Confirm the food type, **quantity**, and preparation or expiry time.",
        "Keep the item in safe storage while the recipient verifies it can accept the donation.",
        "Do not donate food with damaged packaging, unknown handling history, or signs of spoilage.",
    ])


def load_partners(rows: Iterable[dict[str, str]]) -> list[Partner]:
    return [
        Partner(**{key: str(row.get(key) or "").strip() for key in Partner.__annotations__})
        for row in rows
    ]


def match_partners(category: str, partners: Iterable[Partner]) -> list[Partner]:
    """Recommend directory entries by food category; never imply a live availability check."""
    accepted = {
        "prepared_hot_meal": {"prepared_hot_meal", "refrigerated_perishable"},
        "refrigerated_perishable": {"refrigerated_perishable"},
        "frozen": {"frozen"},
        "shelf_stable": {"shelf_stable"},
        "needs_review": set(),
    }.get(category, set())
    matches = [partner for partner in partners if partner.accepts in accepted]
    return sorted(matches, key=lambda partner: (partner.priority, partner.name))


def parse_inventory_csv(raw: bytes) -> list[dict[str, str]]:
    """Read a CSV inventory without assuming a fixed column schema."""
    decoded = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise ValueError("The CSV needs a header row, for example: item,quantity,prepared_or_expiry.")
    return [{key.strip(): (value or "").strip() for key, value in row.items() if key} for row in reader]


def summarize_inventory(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped = {"Immediate Rescue": [], "Long-term Storage": [], "Needs Review": []}
    for row in rows:
        description = " ".join(row.values())
        category = classify_food(description)
        enriched = {**row, "category": category}
        if category in {"prepared_hot_meal", "refrigerated_perishable"}:
            grouped["Immediate Rescue"].append(enriched)
        elif category in {"frozen", "shelf_stable"}:
            grouped["Long-term Storage"].append(enriched)
        else:
            grouped["Needs Review"].append(enriched)
    return grouped
