CONSULTING_COMPANIES = {
    "TCS",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant",
    "Capgemini",
    "Mindtree",
    "HCL"
}


def experience_score(candidate):

    years = candidate["profile"]["years_of_experience"]

    if 5 <= years <= 9:
        return 1.0

    if 4 <= years < 5:
        return 0.8

    if 9 < years <= 12:
        return 0.8

    if 3 <= years < 4:
        return 0.6

    return 0.3


def product_company_score(candidate):

    jobs = candidate.get("career_history", [])

    consulting_count = 0

    for job in jobs:

        company = job.get("company", "")

        if company in CONSULTING_COMPANIES:
            consulting_count += 1

    if len(jobs) == 0:
        return 0.5

    ratio = consulting_count / len(jobs)

    return 1 - ratio


def behavioral_score(candidate):

    signals = candidate["redrob_signals"]

    score = 0.0

    if signals.get("open_to_work_flag"):
        score += 0.18

    github = signals.get("github_activity_score", -1)

    if github > 0:
        score += min(github / 100, 0.15)

    score += min(
        signals.get("recruiter_response_rate", 0),
        0.20
    )

    score += min(
        signals.get("interview_completion_rate", 0),
        0.15
    )

    score += min(
        signals.get("profile_completeness_score", 0) / 100,
        0.10
    )

    profile_views = signals.get("profile_views_received_30d", 0)
    saved_by_recruiters = signals.get("saved_by_recruiters_30d", 0)
    search_appearance = signals.get("search_appearance_30d", 0)

    score += min((profile_views ** 0.5) / 10, 0.05)
    score += min((saved_by_recruiters ** 0.5) / 4, 0.05)
    score += min((search_appearance ** 0.5) / 25, 0.04)

    last_active_date = signals.get("last_active_date")

    if last_active_date:
        from datetime import date

        try:
            days_since_active = (date.today() - date.fromisoformat(last_active_date)).days
        except ValueError:
            days_since_active = None

        if days_since_active is not None:
            if days_since_active <= 14:
                score += 0.06
            elif days_since_active <= 30:
                score += 0.04
            elif days_since_active <= 60:
                score += 0.02

    if signals.get("verified_email"):
        score += 0.02

    if signals.get("verified_phone"):
        score += 0.02

    if signals.get("linkedin_connected"):
        score += 0.02

    return min(score, 1.0)