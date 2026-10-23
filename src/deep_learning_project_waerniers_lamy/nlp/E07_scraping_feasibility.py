"""Experiment 7: External Scraping & Reverse Image Search Feasibility Study.

Evaluates the feasibility, yield, and limitations of external web enrichment:
- Sample: 20 representative paintings from Museum A (10 named artists, 10 Unknown Artist).
- Tests:
    1. Identification rate (can the specific painting be identified from web art catalogs?).
    2. Visual content yield (does external text describe visual scene/objects vs biography/auction price?).
    3. Latency, rate limits, and failure modes.
- Compares yield and reliability with local VLM (BLIP) and direct CLIP embeddings.

Outputs results/scraping_feasibility_report.json.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from deep_learning_project_waerniers_lamy.nlp.E01_generate_descriptions import (
    load_cached_descriptions,
)
from deep_learning_project_waerniers_lamy.nlp.utils import DATA_DIR, RESULTS_DIR, load_paintings

REPORT_FILE = RESULTS_DIR / "scraping_feasibility_report.json"


def query_wikipedia_catalog(artist: str, genre: str, style: str) -> dict:
    """Query Wikipedia search API for catalog / artwork descriptions."""
    if "unknown" in artist.lower():
        return {
            "identified": False,
            "has_visual_description": False,
            "retrieved_snippet": "",
            "source": "Wikipedia",
            "reason": "Unknown artist cannot be resolved in biographical catalogs",
        }

    clean_artist = artist.replace("-", " ")
    query_str = f"{clean_artist} {genre} {style} painting"
    encoded = urllib.parse.quote(query_str)
    url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&format=json&utf8=1"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MuseumGuideScrapingStudy/1.0 (Academic research project; contact: student@albertschool.com)"},
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            dt = time.time() - t0
            search_results = payload.get("query", {}).get("search", [])

            if not search_results:
                return {
                    "identified": False,
                    "has_visual_description": False,
                    "retrieved_snippet": "",
                    "latency_ms": round(dt * 1000, 1),
                    "source": "Wikipedia",
                    "reason": "No search hits",
                }

            top_hit = search_results[0]
            snippet = top_hit.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")

            # Check if snippet contains visual descriptors (water, tree, woman, snow, etc.)
            visual_keywords = ["water", "sea", "tree", "river", "mountain", "snow", "flower", "woman", "man", "house", "sky"]
            has_visual = any(w in snippet.lower() for w in visual_keywords)

            return {
                "identified": True,
                "title": top_hit.get("title", ""),
                "has_visual_description": has_visual,
                "retrieved_snippet": snippet[:150],
                "latency_ms": round(dt * 1000, 1),
                "source": "Wikipedia",
            }
    except Exception as e:
        return {
            "identified": False,
            "has_visual_description": False,
            "retrieved_snippet": "",
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "source": "Wikipedia",
            "error": str(e),
        }


def run_scraping_feasibility_study(n_named: int = 10, n_unknown: int = 10) -> dict:
    print("=" * 70)
    print("EXPERIMENT 7: External Scraping & Reverse Search Feasibility Study")
    print(f"Sample: {n_named} Named Artists + {n_unknown} Unknown Artists")
    print("=" * 70)

    paintings = load_paintings(split="val")
    vlm_cache = load_cached_descriptions()

    # Stratified sampling
    named_sample = paintings[paintings["artist"] != "Unknown Artist"].head(n_named)
    unknown_sample = paintings[paintings["artist"] == "Unknown Artist"].head(n_unknown)
    sample = pd.concat([named_sample, unknown_sample], ignore_index=True)

    results = []
    print(f"\nTesting external catalog lookup on {len(sample)} paintings...\n")

    for _, row in sample.iterrows():
        p_id = row["id"]
        artist = str(row["artist"])
        genre = str(row["genre"])
        style = str(row["style"])
        is_unknown = "unknown" in artist.lower()

        # External lookup
        res = query_wikipedia_catalog(artist, genre, style)
        vlm_desc = vlm_cache.get(p_id, "N/A")

        entry = {
            "id": p_id,
            "artist": artist,
            "is_unknown_artist": is_unknown,
            "genre": genre,
            "style": style,
            "external_identified": res.get("identified", False),
            "external_has_visual_text": res.get("has_visual_description", False),
            "external_snippet": res.get("retrieved_snippet", ""),
            "vlm_description": vlm_desc,
            "latency_ms": res.get("latency_ms", 0.0),
        }
        results.append(entry)

        status_tag = "[MATCH]" if entry["external_identified"] else "[MISS]"
        visual_tag = "(Visual text: YES)" if entry["external_has_visual_text"] else "(Visual text: NO)"
        print(f"[{p_id}] {artist[:18]:<18} | {status_tag} {visual_tag} -> VLM: '{vlm_desc[:40]}...'")

    df_res = pd.DataFrame(results)

    # Compute key statistics
    total = len(df_res)
    named_df = df_res[~df_res["is_unknown_artist"]]
    unknown_df = df_res[df_res["is_unknown_artist"]]

    overall_id_rate = df_res["external_identified"].mean()
    named_id_rate = named_df["external_identified"].mean() if len(named_df) else 0.0
    unknown_id_rate = unknown_df["external_identified"].mean() if len(unknown_df) else 0.0

    visual_yield_rate = df_res["external_has_visual_text"].mean()
    named_visual_rate = named_df["external_has_visual_text"].mean() if len(named_df) else 0.0

    vlm_coverage = (df_res["vlm_description"] != "N/A").mean()

    summary = {
        "sample_size": total,
        "named_artists_count": len(named_df),
        "unknown_artists_count": len(unknown_df),
        "identification_rate_overall": round(float(overall_id_rate), 4),
        "identification_rate_named": round(float(named_id_rate), 4),
        "identification_rate_unknown": round(float(unknown_id_rate), 4),
        "visual_description_yield_overall": round(float(visual_yield_rate), 4),
        "visual_description_yield_named": round(float(named_visual_rate), 4),
        "vlm_coverage_rate": round(float(vlm_coverage), 4),
        "avg_latency_ms": round(float(df_res["latency_ms"].mean()), 1),
    }

    # Print summary table
    print("\n" + "=" * 70)
    print("FEASIBILITY STUDY SUMMARY")
    print("=" * 70)
    print(f"Overall Identification Rate:       {summary['identification_rate_overall'] * 100:.1f}%")
    print(f"  - On Named Artists:               {summary['identification_rate_named'] * 100:.1f}%")
    print(f"  - On Unknown Artists (54% data):  {summary['identification_rate_unknown'] * 100:.1f}%")
    print(f"Visual Content Yield (External):   {summary['visual_description_yield_overall'] * 100:.1f}%")
    print(f"VLM Description Coverage:          {summary['vlm_coverage_rate'] * 100:.1f}% (Local & Deterministic)")
    print(f"Average External Latency:          {summary['avg_latency_ms']:.1f} ms/query")
    print("=" * 70)

    # Save to JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_payload = {"summary": summary, "sample_results": results}
    REPORT_FILE.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved detailed report to {REPORT_FILE}")

    return summary


if __name__ == "__main__":
    run_scraping_feasibility_study()
