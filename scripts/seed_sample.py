"""Deterministically generate voicelens/data/sample_reviews.jsonl.

The set is synthetic — not real Amazon reviews. ~80% are valid English reviews
across CE-style SKUs; the remainder are intentionally crafted to exercise
each DQ check (duplicate, missing asin, invalid rating, non-English, low
lang_confidence, short text, empty/spam).
"""
from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta

from voicelens.config import SAMPLE_REVIEWS_PATH

PRODUCTS = [
    ("Anker", "B0ANK10001", "PowerCore-10K", "power_bank"),
    ("Anker", "B0ANK10002", "PowerCore-24K", "power_bank"),
    ("Anker", "B0ANK10003", "Charger-Nano-II-65W", "charger"),
    ("Anker", "B0ANK10004", "Cable-USBC-PD-3FT", "cable"),
    ("Soundcore", "B0SND20001", "Liberty-4-NC", "earbuds"),
    ("Soundcore", "B0SND20002", "Space-Q45", "headphones"),
    ("Soundcore", "B0SND20003", "Boom-2", "speaker"),
    ("RAVPower", "B0RAV30001", "PD-Pioneer-20W", "charger"),
    ("Aukey", "B0AUK40001", "Omnia-65W", "charger"),
    ("Bose", "B0BOS50001", "QuietComfort-Ultra", "headphones"),
    ("JBL", "B0JBL60001", "Flip-6", "speaker"),
    ("UGREEN", "B0UGR70001", "Nexode-100W", "charger"),
]

POSITIVE_TEMPLATES = {
    "power_bank": [
        "Charges my phone two full times before needing a top up. Very compact for travel.",
        "Held its charge for weeks in my bag. Solid build quality, no complaints.",
        "Fast pass-through charging works exactly as advertised, great for road trips.",
    ],
    "charger": [
        "Tiny GaN charger that powers my laptop and phone at the same time. Recommended.",
        "USB-C PD delivers full speed to my MacBook, no issues. Stays cool under load.",
        "Replaced my bulky OEM brick with this. Same wattage, half the size.",
    ],
    "cable": [
        "Braided cable feels durable, transfers data and PD power without drops.",
        "Lasted longer than three other cables I tried. Worth the price.",
        "Plug fits snugly, no wiggle, charges at full speed every time.",
    ],
    "earbuds": [
        "Active noise cancellation is impressive for the price. Comfortable for long flights.",
        "Battery lasts a full work day with ANC on. Bluetooth pairing is instant.",
        "Sound quality is balanced, bass is punchy without drowning vocals.",
    ],
    "headphones": [
        "Wore these for an 8 hour flight, no ear fatigue. ANC blocks engine drone well.",
        "Crisp highs, deep bass, clean mids. Bluetooth range is solid across rooms.",
        "Build quality feels premium, foldable design fits in my carry on.",
    ],
    "speaker": [
        "Punchy sound at the beach, battery lasted the entire afternoon. Pairs fast.",
        "Bluetooth held connection from 30 feet away, no audio drops.",
        "Waterproof claim held up after dunking in the pool. Bass is surprisingly strong.",
    ],
}

NEGATIVE_TEMPLATES = {
    "power_bank": [
        "Battery drained from full to dead in under 24 hours sitting on the shelf.",
        "After two weeks it gets uncomfortably hot while charging my phone. Worried about safety.",
        "Charges much slower than advertised, takes nearly six hours from empty.",
    ],
    "charger": [
        "Got dangerously hot after about twenty minutes, had to unplug it. Returning.",
        "Stopped working after a month, will not deliver PD to any of my devices.",
        "Loud coil whine when laptop is at full load, very annoying.",
    ],
    "cable": [
        "Frayed near the USB-C end after just three weeks of normal use.",
        "Stopped transferring data after a single drop, only charges now.",
        "Plug is too loose in my phone port, charging cuts in and out constantly.",
    ],
    "earbuds": [
        "Bluetooth keeps disconnecting from my phone every few minutes, unusable for calls.",
        "Right earbud died completely after about a month, left one still works.",
        "Touch controls are unresponsive half the time, lots of mis-taps.",
    ],
    "headphones": [
        "Bluetooth latency on video is awful, lip sync is way off on every show.",
        "Headband cracked after three months of light use. Not durable.",
        "ANC produces a constant hiss in quiet rooms, ruins the experience.",
    ],
    "speaker": [
        "Bluetooth pairing fails on every other attempt, very frustrating.",
        "Distorts heavily at higher volumes, treble becomes harsh.",
        "Battery indicator is wildly inaccurate, jumps from 50 percent to dead.",
    ],
}

NEUTRAL_TEMPLATES = [
    "Decent product overall. Does what it claims, nothing more, nothing less.",
    "Works as expected. Packaging was a bit damaged on arrival but device is fine.",
    "Average for the price. Not the best I have used but not the worst either.",
]


def _pick_text(rng: random.Random, sentiment: str, category: str) -> str:
    if sentiment == "positive":
        bank = POSITIVE_TEMPLATES.get(category, POSITIVE_TEMPLATES["charger"])
    elif sentiment == "negative":
        bank = NEGATIVE_TEMPLATES.get(category, NEGATIVE_TEMPLATES["charger"])
    else:
        bank = NEUTRAL_TEMPLATES
    return rng.choice(bank)


def _gen_valid(rng: random.Random, idx: int, start: datetime) -> dict:
    brand, asin, model, category = rng.choice(PRODUCTS)
    sentiment_roll = rng.random()
    if sentiment_roll < 0.55:
        sentiment = "positive"
        rating = rng.choice([4, 5, 5, 5])
    elif sentiment_roll < 0.85:
        sentiment = "negative"
        rating = rng.choice([1, 1, 2, 2, 3])
    else:
        sentiment = "neutral"
        rating = rng.choice([3, 4])

    posted = start + timedelta(days=rng.randint(0, 180), hours=rng.randint(0, 23))

    return {
        "source": "amazon_reviews_2023",
        "source_id": f"R{idx:07d}",
        "asin": asin,
        "brand": brand,
        "model_number": model,
        "category": category,
        "rating": rating,
        "verified": rng.random() < 0.85,
        "posted_at": posted.isoformat(),
        "helpful_count": rng.randint(0, 25),
        "text_raw": _pick_text(rng, sentiment, category),
        "language": "en",
        "lang_confidence": round(rng.uniform(0.85, 0.99), 3),
        "locale": "en-US",
    }


def _gen_invalid(rng: random.Random, idx: int, base_valid: list[dict]) -> list[dict]:
    invalid: list[dict] = []

    for k in range(3):
        clone = dict(base_valid[k])
        clone["source_id"] = clone["source_id"]
        invalid.append(clone)

    for k in range(3):
        brand, _, model, category = rng.choice(PRODUCTS)
        invalid.append({
            "source": "amazon_reviews_2023",
            "source_id": f"R{idx + k:07d}",
            "asin": None,
            "brand": brand,
            "model_number": model,
            "category": category,
            "rating": 5,
            "verified": True,
            "posted_at": "2025-02-01T10:00:00",
            "helpful_count": 0,
            "text_raw": "Works fine but Amazon did not link this to a SKU somehow.",
            "language": "en",
            "lang_confidence": 0.95,
            "locale": "en-US",
        })

    for k, bad_rating in enumerate([0, 6, -1]):
        brand, asin, model, category = rng.choice(PRODUCTS)
        invalid.append({
            "source": "amazon_reviews_2023",
            "source_id": f"R{idx + 100 + k:07d}",
            "asin": asin,
            "brand": brand,
            "model_number": model,
            "category": category,
            "rating": bad_rating,
            "verified": True,
            "posted_at": "2025-02-02T10:00:00",
            "helpful_count": 0,
            "text_raw": "This row has an invalid rating value for DQ testing.",
            "language": "en",
            "lang_confidence": 0.95,
            "locale": "en-US",
        })

    for k, (lang, text) in enumerate([
        ("de", "Sehr gutes Produkt, schnelles Aufladen und tolle Verarbeitung."),
        ("ja", "音質はとても良いです。ノイズキャンセリングも効果的です。"),
        ("fr", "Tres bon produit, la batterie dure longtemps et le son est clair."),
    ]):
        brand, asin, model, category = rng.choice(PRODUCTS)
        invalid.append({
            "source": "amazon_reviews_2023",
            "source_id": f"R{idx + 200 + k:07d}",
            "asin": asin,
            "brand": brand,
            "model_number": model,
            "category": category,
            "rating": 4,
            "verified": True,
            "posted_at": "2025-02-03T10:00:00",
            "helpful_count": 0,
            "text_raw": text,
            "language": lang,
            "lang_confidence": 0.95,
            "locale": "en-US",
        })

    for k in range(3):
        brand, asin, model, category = rng.choice(PRODUCTS)
        invalid.append({
            "source": "amazon_reviews_2023",
            "source_id": f"R{idx + 300 + k:07d}",
            "asin": asin,
            "brand": brand,
            "model_number": model,
            "category": category,
            "rating": 4,
            "verified": True,
            "posted_at": "2025-02-04T10:00:00",
            "helpful_count": 0,
            "text_raw": "Mixed feelings, kind of ok but the language detector was not sure.",
            "language": "en",
            "lang_confidence": 0.42,
            "locale": "en-US",
        })

    for k, txt in enumerate(["nice!", "great", "ok"]):
        brand, asin, model, category = rng.choice(PRODUCTS)
        invalid.append({
            "source": "amazon_reviews_2023",
            "source_id": f"R{idx + 400 + k:07d}",
            "asin": asin,
            "brand": brand,
            "model_number": model,
            "category": category,
            "rating": 5,
            "verified": True,
            "posted_at": "2025-02-05T10:00:00",
            "helpful_count": 0,
            "text_raw": txt,
            "language": "en",
            "lang_confidence": 0.95,
            "locale": "en-US",
        })

    spam_texts = [
        "https://buycheap.example.com https://promo.example.com",
        "BEST PRODUCT EVER BUY NOW WITH DISCOUNT CODE PROMO",
        "",
    ]
    for k, txt in enumerate(spam_texts):
        brand, asin, model, category = rng.choice(PRODUCTS)
        invalid.append({
            "source": "amazon_reviews_2023",
            "source_id": f"R{idx + 500 + k:07d}",
            "asin": asin,
            "brand": brand,
            "model_number": model,
            "category": category,
            "rating": 5,
            "verified": False,
            "posted_at": "2025-02-06T10:00:00",
            "helpful_count": 0,
            "text_raw": txt,
            "language": "en",
            "lang_confidence": 0.95,
            "locale": "en-US",
        })

    return invalid


def generate(n_valid: int = 80, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    start = datetime(2025, 1, 1)
    valid = [_gen_valid(rng, i, start) for i in range(n_valid)]
    invalid = _gen_invalid(rng, idx=n_valid, base_valid=valid)
    return valid + invalid


def main() -> int:
    rows = generate()
    SAMPLE_REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_REVIEWS_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {SAMPLE_REVIEWS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
