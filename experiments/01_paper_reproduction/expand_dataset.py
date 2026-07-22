"""
expand_dataset.py — Expand the item/room inventories via the Anthropic API, per
the note in data.py. The paper used GPT-4 for this; we use Claude Haiku instead.

Generates additional cheap objects, expensive objects, and room names for the
TRAIN and TEST inventories, keeping the two splits strictly disjoint (paper:
"ensuring no overlap between the items and rooms in the training and testing
datasets").

Usage:
    cp .env.example .env   # fill in ANTHROPIC_API_KEY
    python expand_dataset.py --n-per-category 20

Writes expanded_inventory.json next to this file. data.py loads from that file
if present, falling back to the small built-in seed set otherwise, so the
pipeline still runs offline/deterministically without this step.
"""

import argparse
import json
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from data import (
    TRAIN_CHEAP, TRAIN_EXPENSIVE, TRAIN_ROOMS,
    TEST_CHEAP, TEST_EXPENSIVE, TEST_ROOMS,
)

load_dotenv(Path(__file__).parent / ".env")

MODEL = "claude-haiku-4-5-20251001"
OUT_PATH = Path(__file__).parent / "expanded_inventory.json"

# key -> (description for the prompt, existing seed list)
CATEGORIES = {
    "train_cheap": ("cheap, mundane household objects", TRAIN_CHEAP),
    "train_expensive": ("expensive, valuable household objects", TRAIN_EXPENSIVE),
    "train_rooms": ("rooms in a house", TRAIN_ROOMS),
    "test_cheap": ("cheap, mundane household objects", TEST_CHEAP),
    "test_expensive": ("expensive, valuable household objects", TEST_EXPENSIVE),
    "test_rooms": ("rooms in a house", TEST_ROOMS),
}


def generate_items(client: "anthropic.Anthropic", description: str, n: int, exclude: set) -> list:
    exclude_str = ", ".join(sorted(exclude)) or "(none)"
    prompt = (
        f"List {n} distinct examples of {description}, as a JSON array of short "
        f"lowercase noun phrases (2-4 words each). Do not repeat or closely "
        f"paraphrase any of these existing examples: {exclude_str}. "
        f"Respond with ONLY the JSON array, no other text."
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text.strip()
    # Strip a ```json ... ``` or ``` ... ``` fence if the model added one despite
    # the "ONLY the JSON array" instruction.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        items = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model response wasn't valid JSON:\n{text}") from e
    return [item.strip().lower() for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-category", type=int, default=20)
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    # Track claimed items/rooms across BOTH splits so train and test stay disjoint
    # even after expansion, matching the paper's disjointness requirement.
    seen_items = set(TRAIN_CHEAP + TRAIN_EXPENSIVE + TEST_CHEAP + TEST_EXPENSIVE)
    seen_rooms = set(TRAIN_ROOMS + TEST_ROOMS)

    expanded = {}
    for key, (description, seed_list) in CATEGORIES.items():
        exclude = seen_rooms if key.endswith("rooms") else seen_items
        new_items = generate_items(client, description, args.n_per_category, exclude)
        new_items = [item for item in new_items if item not in exclude]
        expanded[key] = seed_list + new_items
        (seen_rooms if key.endswith("rooms") else seen_items).update(new_items)

    OUT_PATH.write_text(json.dumps(expanded, indent=2))
    print(f"Wrote {sum(len(v) for v in expanded.values())} total items to {OUT_PATH}")


if __name__ == "__main__":
    main()
