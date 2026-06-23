def build_candidate_text(candidate):
    """
    Build a compact candidate representation for semantic embedding.

    Important:
    - Keeps only the most useful information.
    - Limits extremely long text fields.
    - Uses only recent career history.
    - Prevents tokenizer slowdown on 100k candidates.
    """

    profile = candidate.get("profile", {})

    sections = []

    # Core profile information
    headline = profile.get("headline", "")
    current_title = profile.get("current_title", "")
    summary = profile.get("summary", "")[:500]
    industry = profile.get("current_industry", "")
    years = profile.get("years_of_experience", 0)

    if headline:
        sections.append(f"Headline: {headline}")

    if current_title:
        sections.append(f"Current Role: {current_title}")

    if industry:
        sections.append(f"Industry: {industry}")

    sections.append(f"Experience: {years} years")

    if summary:
       sections.append(f"Summary: {profile.get('summary', '')[:300]}")

    # Skills (most important for matching)
    skill_names = [
        skill.get("name", "")
        for skill in candidate.get("skills", [])[:10]
        if skill.get("name")
    ]

    if skill_names:
        sections.append(
            "Skills: " + ", ".join(skill_names[:30])
        )

    # Only recent jobs (first 3)
    career_history = candidate.get("career_history", [])[:3]

    if career_history:
        sections.append("Career History:")

    for job in career_history:

        company = job.get("company", "")
        title = job.get("title", "")
        desc = job.get("description", "")[:200]

        job_text = []

        if title:
            job_text.append(title)

        if company:
            job_text.append(f"at {company}")

        if desc:
            job_text.append(desc)

        sections.append(" ".join(job_text))

    # Education
    education_entries = []

    for edu in candidate.get("education", [])[:2]:

        degree = edu.get("degree", "")
        field = edu.get("field_of_study", "")

        text = ""

        if degree:
            text += degree

        if field:
            text += f" in {field}"

        if text:
            education_entries.append(text)

    if education_entries:
        sections.append(
            "Education: " + "; ".join(education_entries)
        )

    text = "\n".join(sections)

    # Hard cap to prevent tokenizer explosion
    return text[:1500]