import argparse

from src.parser import iter_candidates, load_job_description
from src.ranker import (
    audit_and_rank_candidates,
    build_submission_rows,
    rank_candidates,
    write_submission_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Rank candidates for the Redrob challenge.")
    parser.add_argument(
        "--candidates",
        default="data/candidates.jsonl",
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--jd",
        default="data/job_description.docx",
        help="Path to the job description docx",
    )
    parser.add_argument(
        "--out",
        default="outputs/submission.csv",
        help="Path to write the submission CSV",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of ranked candidates to include in the output",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Batch size used for embedding the candidate texts",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print an audit report with top-20 score breakdown and old-vs-new top-10 comparison",
    )
    parser.add_argument(
        "--audit-limit",
        type=int,
        default=20,
        help="Number of top candidates to include in the audit breakdown",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading JD...")
    jd_text = load_job_description(args.jd)

    if args.audit:
        print("Auditing and ranking candidates...")
        audit_result = audit_and_rank_candidates(
            iter_candidates(args.candidates),
            jd_text,
            batch_size=args.batch_size,
            limit=args.limit,
            audit_limit=args.audit_limit,
        )
        ranked_candidates = audit_result["ranked_candidates"]

        print("\nPrevious weights:", audit_result["old_weights"])
        print("Proposed weights:", audit_result["proposed_weights"])
        print("Final weights:", audit_result["new_weights"])
        print("Semantic dominance detected:", audit_result["base_audit"]["semantic_dominates"])
        print("Rebalance kept:", audit_result["rebalanced"])

        print("\nTOP 20 SCORE BREAKDOWN")
        for row in audit_result["final_audit"]["rows"]:
            print({
                "candidate_id": row["candidate_id"],
                "semantic_score": round(row["semantic"], 4),
                "career_score": round(row["career"], 4),
                "title_boost_score": round(row["title_boost"], 4),
                "experience_score": round(row["experience"], 4),
                "product_score": round(row["product"], 4),
                "behavior_score": round(row["behavior"], 4),
                "final_score": round(row["final"], 4),
            })

        print("\nOLD TOP 10")
        for row in audit_result["old_ranked"][:10]:
            print({
                "candidate_id": row["candidate_id"],
                "score": round(row["score"], 4),
                "reasoning": row["reasoning"],
            })

        print("\nNEW TOP 10")
        for row in audit_result["proposed_ranked"][:10]:
            print({
                "candidate_id": row["candidate_id"],
                "score": round(row["score"], 4),
                "reasoning": row["reasoning"],
            })
    else:
        print("Ranking candidates...")
        ranked_candidates = rank_candidates(
            iter_candidates(args.candidates),
            jd_text,
            batch_size=args.batch_size,
            
        )

    rows = build_submission_rows(ranked_candidates, limit=args.limit)
    write_submission_csv(rows, args.out)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()