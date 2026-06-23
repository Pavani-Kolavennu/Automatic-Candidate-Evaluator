import time

from src.parser import load_candidates, load_job_description
from src.ranker import (
    rank_candidates,
    build_submission_rows,
    write_submission_csv,
)


def main():
    start_time = time.perf_counter()

    print("Loading JD...")
    jd = load_job_description("data/job_description.docx")

    print("Loading candidates...")
    candidates = load_candidates("data/candidates.jsonl")
    print("Number of candidates:", len(candidates))

    print("Ranking candidates...")
    results = rank_candidates(
    candidates,
    jd,
    batch_size=64,
    limit=100
    )
    end_time = time.perf_counter()

    print(f"Runtime: {end_time - start_time:.2f} seconds")

    rows = build_submission_rows(results, limit=100)
    write_submission_csv(rows, "submission.csv")
    print("Saved submission.csv successfully")


    print("\n========================")
    print("BEST CANDIDATE PROFILE")
    print("========================\n")

    top_id = results[0]["candidate_id"] if results else None

    for candidate in candidates:

        if candidate["candidate_id"] == top_id:

            print("Candidate ID:", candidate["candidate_id"])
            print()

            print("Headline:")
            print(candidate["profile"]["headline"])
            print()

            print("Current Title:")
            print(candidate["profile"]["current_title"])
            print()

            print("Years of Experience:")
            print(candidate["profile"]["years_of_experience"])
            print()

            print("Summary:")
            print(candidate["profile"]["summary"])
            print()

            print("Skills:")

            skill_names = []

            for skill in candidate.get("skills", []):
                skill_names.append(skill["name"])

            print(", ".join(skill_names))

            break


if __name__ == "__main__":
    main()