import csv
import heapq

from src.career_scorer import career_evidence_breakdown
from src.embedding_engine import EmbeddingEngine
from src.jd_parser import extract_requirements
from src.profile_builder import build_candidate_text
from src.scorer import (
    behavioral_score,
    experience_score,
    product_company_score,
)


DEFAULT_WEIGHTS = {
    "semantic": 0.30,
    "career": 0.38,
    "experience": 0.10,
    "product": 0.07,
    "behavior": 0.15,
}

PREFILTER_LIMIT = 1000

PREFILTER_KEYWORDS = (
    "recommendation",
    "recommender",
    "retrieval",
    "search",
    "ranking",
    "semantic search",
    "vector",
    "embeddings",
    "ml engineer",
    "machine learning",
    "ai engineer",
    "applied scientist",
    "rag",
    "faiss",
    "pinecone",
    "qdrant",
    "weaviate",
    "elasticsearch",
)

STRONG_TITLE_TERMS = (
    "recommendation systems engineer",
    "recommender systems engineer",
    "retrieval engineer",
    "search engineer",
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "applied scientist",
)

MEDIUM_TITLE_TERMS = (
    "data engineer",
    "backend engineer",
    "data scientist",
)

WEAK_TITLE_TERMS = (
    "frontend engineer",
    "qa engineer",
    "support engineer",
    "devops engineer",
)

ROLE_TERMS = (
    "retrieval",
    "search",
    "ranking",
    "recommendation",
    "embedding",
    "vector",
    "evaluation",
    "production ml",
    "ml in production",
    "weaviate",
    "milvus",
    "pinecone",
    "qdrant",
    "faiss",
    "rag",
)


def combine_scores(semantic, career, experience, product, behavior, weights=None):
    selected = weights or DEFAULT_WEIGHTS

    return (
        selected["semantic"] * semantic +
        selected["career"] * career +
        selected["experience"] * experience +
        selected["product"] * product +
        selected["behavior"] * behavior
    )


def _build_title_text(candidate):
    profile = candidate.get("profile", {})
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
    ]

    for job in candidate.get("career_history", []):
        title = job.get("title", "")

        if title:
            parts.append(title)

    return " ".join(parts).lower()


def _title_family_score(candidate):
    title_text = _build_title_text(candidate)
    score = 0.0

    if any(term in title_text for term in STRONG_TITLE_TERMS):
        score += 3.0

    if any(term in title_text for term in MEDIUM_TITLE_TERMS):
        score += 1.5

    if any(term in title_text for term in WEAK_TITLE_TERMS):
        score -= 2.0

    return score


def _keyword_prefilter_score(candidate):
    profile = candidate.get("profile", {})

    current_title = profile.get("current_title", "").lower()
    headline = profile.get("headline", "").lower()
    skills = " ".join(
        skill.get("name", "")
        for skill in candidate.get("skills", [])
        if skill.get("name", "")
    ).lower()
    history_titles = " ".join(
        job.get("title", "")
        for job in candidate.get("career_history", [])
        if job.get("title", "")
    ).lower()

    score = 0.0

    for keyword in PREFILTER_KEYWORDS:
        if keyword in current_title:
            score += 3.0
        if keyword in headline:
            score += 3.0
        if keyword in skills:
            score += 1.5
        if keyword in history_titles:
            score += 2.0

    return score


def _prefilter_candidates(candidates, limit=PREFILTER_LIMIT):
    top_heap = []
    sequence = 0

    for candidate in candidates:
        score = _keyword_prefilter_score(candidate)
        candidate_id = candidate.get("candidate_id", "")
        item = (
            score,
            tuple(-ord(ch) for ch in candidate_id),
            sequence,
            candidate,
        )

        if len(top_heap) < limit:
            heapq.heappush(top_heap, item)
        elif item > top_heap[0]:
            heapq.heapreplace(top_heap, item)

        sequence += 1

    top_heap.sort(reverse=True)
    return [item[3] for item in top_heap]


def _rebalance_weights(weights, audit_summary):
    selected = dict(weights)

    semantic_mean = audit_summary["semantic_mean_contribution"]
    recruiter_mean = audit_summary["recruiter_mean_contribution"]

    if semantic_mean <= recruiter_mean:
        return selected, False

    ratio = recruiter_mean / max(semantic_mean, 1e-9)
    adjusted_semantic = selected["semantic"] * ratio
    adjusted_semantic = max(0.18, min(0.28, adjusted_semantic))

    delta = selected["semantic"] - adjusted_semantic

    if delta <= 0:
        return selected, False

    selected["semantic"] = adjusted_semantic
    selected["career"] += delta * 0.75
    selected["experience"] += delta * 0.10
    selected["product"] += delta * 0.05
    selected["behavior"] += delta * 0.10

    total = sum(selected.values())

    for key in selected:
        selected[key] = round(selected[key] / total, 4)

    return selected, True


def _ranking_quality_score(ranked_candidates):
    score = 0.0

    for rank, row in enumerate(ranked_candidates[:10], start=1):
        title_score = _title_family_score(row["candidate"])
        title_score += row.get("career_breakdown", {}).get("title_score", 0.0)
        score += (11 - rank) * title_score

    return score


def build_ranking_audit(ranked_candidates, weights, top_n=20):
    rows = []
    semantic_total = 0.0
    recruiter_total = 0.0
    strong_title_count = 0
    medium_title_count = 0
    weak_title_count = 0

    for row in ranked_candidates[:top_n]:
        career_breakdown = row.get("career_breakdown", {})
        title_boost_score = career_breakdown.get("title_score", 0.0)

        weighted_semantic = row["semantic"] * weights["semantic"]
        weighted_career = row["career"] * weights["career"]
        weighted_title_boost = title_boost_score * weights["career"]
        weighted_experience = row["experience"] * weights["experience"]
        weighted_product = row["product"] * weights["product"]
        weighted_behavior = row["behavior"] * weights["behavior"]
        recruiter_contribution = (
            weighted_career + weighted_experience + weighted_product + weighted_behavior
        )

        semantic_total += weighted_semantic
        recruiter_total += recruiter_contribution

        title_text = _build_title_text(row["candidate"])

        if any(term in title_text for term in STRONG_TITLE_TERMS):
            strong_title_count += 1
        elif any(term in title_text for term in MEDIUM_TITLE_TERMS):
            medium_title_count += 1
        elif any(term in title_text for term in WEAK_TITLE_TERMS):
            weak_title_count += 1

        rows.append({
            "candidate_id": row["candidate_id"],
            "semantic": row["semantic"],
            "career": row["career"],
            "title_boost": title_boost_score,
            "experience": row["experience"],
            "product": row["product"],
            "behavior": row["behavior"],
            "final": row["score"],
            "weighted_semantic": weighted_semantic,
            "weighted_recruiter": recruiter_contribution,
            "weighted_title_boost": weighted_title_boost,
            "weighted_experience": weighted_experience,
            "weighted_product": weighted_product,
            "weighted_behavior": weighted_behavior,
        })

    semantic_mean = semantic_total / max(len(rows), 1)
    recruiter_mean = recruiter_total / max(len(rows), 1)

    return {
        "top_n": top_n,
        "rows": rows,
        "semantic_total_contribution": semantic_total,
        "recruiter_total_contribution": recruiter_total,
        "semantic_mean_contribution": semantic_mean,
        "recruiter_mean_contribution": recruiter_mean,
        "semantic_dominates": semantic_mean > recruiter_mean,
        "title_mix": {
            "strong": strong_title_count,
            "medium": medium_title_count,
            "weak": weak_title_count,
        },
    }


def _rank_candidates_with_weights(candidates, jd_text, batch_size=64, engine=None, weights=None, limit=100):
    engine = engine or EmbeddingEngine()
    selected_weights = weights or DEFAULT_WEIGHTS
    jd_requirements = extract_requirements(jd_text)
    jd_embedding = engine.encode_text(jd_text)

    print("Scanning candidates...")
    prefiltered_candidates = _prefilter_candidates(candidates, limit=PREFILTER_LIMIT)
    print(f"Prefilter complete. Kept {len(prefiltered_candidates)} candidates.")
    print("Generating embeddings...")

    top_heap = []
    batch = []

    def to_heap_key(score, candidate_id):
        return (
            score,
            tuple(-ord(ch) for ch in candidate_id),
        )

    def flush_batch(batch_items):
        if not batch_items:
            return

        texts = [build_candidate_text(candidate) for candidate in batch_items]
      

       

      
        batch_embeddings = engine.similarities_to_embedding(
            texts,
            jd_embedding,
            batch_size=batch_size,
        )

        for candidate, semantic in zip(batch_items, batch_embeddings):
            career_breakdown = career_evidence_breakdown(candidate, jd_requirements)
            career = career_breakdown["score"]
            exp = experience_score(candidate)
            product = product_company_score(candidate)
            behavior = behavioral_score(candidate)

            final_score = combine_scores(
                float(semantic),
                career,
                exp,
                product,
                behavior,
                selected_weights,
            )

            payload = {
                "candidate": candidate,
                "candidate_id": candidate["candidate_id"],
                "semantic": float(semantic),
                "career": career,
                "title_boost_score": career_breakdown.get("title_score", 0.0),
                "career_breakdown": career_breakdown,
                "experience": exp,
                "product": product,
                "behavior": behavior,
                "score": final_score,
            }

            item = (to_heap_key(final_score, candidate["candidate_id"]), payload)

            if len(top_heap) < limit:
                heapq.heappush(top_heap, item)
            else:
                if item > top_heap[0]:
                    heapq.heapreplace(top_heap, item)

    for candidate in prefiltered_candidates:
        batch.append(candidate)

        if len(batch) >= batch_size:
            flush_batch(batch)
            batch = []

    flush_batch(batch)

    ranked = [item[1] for item in top_heap]
    ranked.sort(key=lambda row: (-row["score"], row["candidate_id"]))

    for row in ranked:
        row["reasoning"] = build_reasoning(row["candidate"], row, jd_requirements)

    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    print("Ranking complete.")

    return ranked


def build_reasoning(candidate, scores, jd_requirements):
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    title = profile.get("current_title", "Candidate")
    years = profile.get("years_of_experience", 0)

    career_breakdown = scores.get("career_breakdown", {})

    strengths = []

    for term in career_breakdown.get("professional_title_terms", [])[:2]:
        strengths.append(term)

    for term in career_breakdown.get("professional_description_terms", [])[:2]:
        if term not in strengths:
            strengths.append(term)

    strengths = strengths[:3]

    response_rate = signals.get("recruiter_response_rate", 0)

    reasoning = f"{title} with {years:.1f} years of experience"

    if strengths:
        reasoning += f" and experience in {', '.join(strengths)}"

    reasoning += "."

    if signals.get("open_to_work_flag"):
        reasoning += " Open to work"

    if response_rate > 0:
        reasoning += f" with recruiter response rate {response_rate:.2f}"

    reasoning += "."

    if scores["product"] >= 0.55:
        reasoning += " Demonstrates product-focused ML experience."

    elif scores["career"] >= 0.40:
        reasoning += " Strong alignment with the search, retrieval, and ranking requirements."

    return reasoning

def score_candidate(candidate, jd_embedding, engine, jd_requirements):
    candidate_text = build_candidate_text(candidate)
    semantic = engine.similarity_to_embedding(candidate_text, jd_embedding)

    career_breakdown = career_evidence_breakdown(candidate, jd_requirements)
    career = career_breakdown["score"]
    exp = experience_score(candidate)
    product = product_company_score(candidate)
    behavior = behavioral_score(candidate)

    score = combine_scores(
        semantic,
        career,
        exp,
        product,
        behavior,
    )

    payload = {
        "candidate": candidate,
        "candidate_id": candidate["candidate_id"],
        "semantic": semantic,
        "career": career,
        "career_breakdown": career_breakdown,
        "experience": exp,
        "product": product,
        "behavior": behavior,
        "score": score,
    }
    payload["reasoning"] = build_reasoning(candidate, payload, jd_requirements)

    return payload

def audit_and_rank_candidates(candidates, jd_text, batch_size=64, engine=None, weights=None, limit=100, audit_limit=20):
    candidate_list = candidates if isinstance(candidates, list) else list(candidates)
    base_weights = dict(weights or DEFAULT_WEIGHTS)
    engine = engine or EmbeddingEngine()

    base_ranked = _rank_candidates_with_weights(
        candidate_list,
        jd_text,
        batch_size=batch_size,
        engine=engine,
        weights=base_weights,
        limit=limit,
    )
    base_audit = build_ranking_audit(base_ranked, base_weights, top_n=audit_limit)

    rebalanced_weights, should_rebalance = _rebalance_weights(base_weights, base_audit)
    proposed_ranked = base_ranked
    proposed_audit = base_audit
    proposed_weights = base_weights
    final_ranked = base_ranked
    final_audit = base_audit
    final_weights = base_weights
    rebalanced = False

    if should_rebalance:
        proposed_ranked = _rank_candidates_with_weights(
            candidate_list,
            jd_text,
            batch_size=batch_size,
            engine=engine,
            weights=rebalanced_weights,
            limit=limit,
        )
        proposed_audit = build_ranking_audit(proposed_ranked, rebalanced_weights, top_n=audit_limit)
        proposed_weights = rebalanced_weights

        if _ranking_quality_score(proposed_ranked) > _ranking_quality_score(base_ranked):
            final_ranked = proposed_ranked
            final_audit = proposed_audit
            final_weights = rebalanced_weights
            rebalanced = True

    return {
        "ranked_candidates": final_ranked,
        "old_ranked": base_ranked,
        "proposed_ranked": proposed_ranked,
        "new_ranked": final_ranked,
        "old_weights": base_weights,
        "proposed_weights": proposed_weights,
        "new_weights": final_weights,
        "base_audit": base_audit,
        "proposed_audit": proposed_audit,
        "final_audit": final_audit,
        "rebalanced": rebalanced,
    }


def rank_candidates(candidates, jd_text, batch_size=64, engine=None, weights=None, limit=100):
    if weights is not None:
        return _rank_candidates_with_weights(
            candidates,
            jd_text,
            batch_size=batch_size,
            engine=engine,
            weights=weights,
            limit=limit,
        )

    audit_result = audit_and_rank_candidates(
        candidates,
        jd_text,
        batch_size=batch_size,
        engine=engine,
        limit=limit,
    )

    return audit_result["ranked_candidates"]


def build_submission_rows(ranked_candidates, limit=100):
    rows = []
    ordered = sorted(
        ranked_candidates[:limit],
        key=lambda row: (-round(row["score"], 4), row["candidate_id"])
    )

    for rank, row in enumerate(ordered, start=1):
        rows.append({
            "candidate_id": row["candidate_id"],
            "rank": rank,
            "score": f"{row['score']:.4f}",
            "reasoning": row["reasoning"],
        })

    return rows


def write_submission_csv(rows, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_id", "rank", "score", "reasoning"],
        )
        writer.writeheader()
        writer.writerows(rows)