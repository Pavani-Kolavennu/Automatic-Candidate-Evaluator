POSITIVE_EVIDENCE = {
    "recommendation system": 0.18,
    "recommendation engine": 0.18,
    "search engine": 0.16,
    "retrieval system": 0.18,
    "information retrieval": 0.18,
    "semantic search": 0.16,
    "vector database": 0.16,
    "vector store": 0.14,
    "embeddings": 0.16,
    "ranking system": 0.16,
    "ranking models": 0.14,
    "relevance tuning": 0.14,
    "evaluation framework": 0.14,
    "evaluation frameworks": 0.14,
    "production ml": 0.16,
    "ml in production": 0.16,
    "deployed": 0.08,
    "production": 0.08,
    "shipped": 0.08,
    "launched": 0.06,
    "served": 0.08,
    "monitoring": 0.08,
    "feature pipeline": 0.10,
    "model monitoring": 0.10,
    "search quality": 0.14,
    "retrieval": 0.12,
    "rag": 0.12,
    "a/b testing": 0.10,
    "ab testing": 0.10,
    "weaviate": 0.06,
    "milvus": 0.06,
    "pinecone": 0.06,
    "qdrant": 0.06,
    "faiss": 0.06,
    "elasticsearch": 0.06,
    "opensearch": 0.06,
}

PROFESSIONAL_AI_TERMS = {
    "recommendation system": 0.18,
    "recommendation engine": 0.18,
    "search engine": 0.16,
    "retrieval system": 0.18,
    "information retrieval": 0.18,
    "semantic search": 0.16,
    "vector database": 0.16,
    "vector store": 0.14,
    "embeddings": 0.16,
    "ranking system": 0.16,
    "ranking models": 0.14,
    "relevance tuning": 0.14,
    "evaluation framework": 0.14,
    "evaluation frameworks": 0.14,
    "production ml": 0.16,
    "ml in production": 0.16,
    "search quality": 0.14,
    "retrieval": 0.12,
    "rag": 0.12,
    "weaviate": 0.06,
    "milvus": 0.06,
    "pinecone": 0.06,
    "qdrant": 0.06,
    "faiss": 0.06,
    "elasticsearch": 0.06,
    "opensearch": 0.06,
}

NEGATIVE_EVIDENCE = {
    "side project": 0.12,
    "personal project": 0.10,
    "hobby project": 0.12,
    "experimental": 0.06,
    "experimented with": 0.08,
    "toy project": 0.12,
    "haven't done it in a professional capacity": 0.22,
    "not done it in a professional capacity": 0.22,
    "no professional experience": 0.18,
}

TITLE_BONUS = {
    "recommendation systems engineer": 0.26,
    "recommender systems engineer": 0.26,
    "recommendation engineer": 0.24,
    "search engineer": 0.24,
    "retrieval engineer": 0.24,
    "search relevance engineer": 0.22,
    "ranking engineer": 0.22,
    "ml engineer": 0.20,
    "machine learning engineer": 0.20,
    "ai engineer": 0.20,
    "applied scientist": 0.18,
    "research scientist": 0.16,
    "data scientist": 0.14,
    "data engineer": 0.10,
    "backend engineer": 0.08,
    "platform engineer": 0.08,
    "software engineer": 0.06,
    "frontend engineer": 0.02,
    "qa engineer": 0.00,
    "support engineer": 0.00,
    "devops engineer": 0.02,
}

PROFESSIONAL_ROLE_ANCHORS = {
    "recommendation systems engineer",
    "recommender systems engineer",
    "recommendation engineer",
    "search engineer",
    "retrieval engineer",
    "search relevance engineer",
    "ranking engineer",
    "ml engineer",
    "machine learning engineer",
    "ai engineer",
    "applied scientist",
    "research scientist",
    "data scientist",
}

CURRENT_TITLE_WEIGHT = 1.35
HISTORY_TITLE_WEIGHT = 1.0
DESCRIPTION_WEIGHT = 0.85
SUMMARY_WEIGHT = 0.18
SKILLS_WEIGHT = 0.10
JD_TITLE_WEIGHT = 0.06
JD_HISTORY_TITLE_WEIGHT = 0.05
JD_DESCRIPTION_WEIGHT = 0.04
JD_WEAK_WEIGHT = 0.01
PROFESSIONAL_ONLY_CAP = 0.30


def _count_phrase_hits(text, phrases):
    score = 0.0
    matched = []

    for phrase, weight in phrases.items():
        if phrase in text:
            score += weight
            matched.append(phrase)

    return score, matched


def _score_text_lower(text, phrases, multiplier=1.0):
    if not text:
        return 0.0, []

    score, matched = _count_phrase_hits(text, phrases)
    return score * multiplier, matched


def _score_text(text, phrases, multiplier=1.0):
    if not text:
        return 0.0, []

    return _score_text_lower(text.lower(), phrases, multiplier)


def _score_title(text):
    if not text:
        return 0.0, []

    return _score_text_lower(text, TITLE_BONUS)


def _collect_candidate_fields(candidate):
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])

    current_title_parts = []
    history_title_parts = []
    description_parts = []

    current_title = profile.get("current_title", "")
    summary = profile.get("summary", "")
    current_title_lower = current_title.lower()
    summary_lower = summary.lower()

    if current_title:
        current_title_parts.append(current_title)

    for job in history:
        title = job.get("title", "")
        description = job.get("description", "")

        if title:
            history_title_parts.append(title)

        if description:
            description_parts.append(description)

    skill_names = [skill.get("name", "") for skill in candidate.get("skills", []) if skill.get("name", "")]
    skills = " ".join(skill_names)
    skills_lower = skills.lower()
    weak_parts = [summary, skills]
    weak_text = " ".join(weak_parts)
    weak_text_lower = weak_text.lower()
    current_title_text = " ".join(current_title_parts).lower()
    history_title_text = " ".join(history_title_parts).lower()
    description_text = " ".join(description_parts).lower()
    professional_text = " ".join(
        part for part in [current_title_text, history_title_text, description_text] if part
    )

    return {
        "current_title": current_title,
        "current_title_lower": current_title_lower,
        "summary": summary,
        "summary_lower": summary_lower,
        "titles": history_title_parts,
        "descriptions": description_parts,
        "skills": skills,
        "skills_lower": skills_lower,
        "current_title_text": current_title_text,
        "history_title_text": history_title_text,
        "title_text": " ".join(part for part in [current_title_text, history_title_text] if part),
        "description_text": description_text,
        "professional_text": professional_text,
        "weak_text": weak_text_lower,
        "weak_text_lower": weak_text_lower,
    }


def career_evidence_breakdown(candidate, jd_requirements=None):
    fields = _collect_candidate_fields(candidate)

    if not fields["title_text"] and not fields["description_text"] and not fields["weak_text"]:
        return {
            "score": 0.0,
            "professional_score": 0.0,
            "weak_score": 0.0,
            "title_score": 0.0,
            "description_score": 0.0,
            "positive_terms": [],
            "negative_terms": [],
            "unrelated_terms": [],
            "professional_title_terms": [],
            "professional_description_terms": [],
            "weak_terms": [],
            "professional_ai_terms": [],
        }

    current_title_score = 0.0
    current_title_terms = []

    if fields["current_title"]:
        current_title_score, current_title_terms = _score_title(fields["current_title_lower"])
        current_title_score *= CURRENT_TITLE_WEIGHT

    professional_ai_terms = []

    if fields["title_text"]:
        for phrase in PROFESSIONAL_ROLE_ANCHORS:
            if phrase in fields["title_text"]:
                professional_ai_terms.append(phrase)

    history_title_score = 0.0
    history_title_terms = []

    if fields["titles"]:
        for title in fields["titles"]:
            title_score, title_terms = _score_title(title.lower())
            history_title_score += title_score
            history_title_terms.extend(title_terms)

        for phrase in PROFESSIONAL_ROLE_ANCHORS:
            if phrase in fields["history_title_text"]:
                professional_ai_terms.append(phrase)

    history_title_score *= HISTORY_TITLE_WEIGHT

    description_score, description_terms = _score_text_lower(
        fields["description_text"],
        POSITIVE_EVIDENCE,
        DESCRIPTION_WEIGHT,
    )

    description_text = fields["description_text"]

    summary_score, summary_terms = _score_text_lower(
        fields["summary_lower"],
        POSITIVE_EVIDENCE,
        SUMMARY_WEIGHT,
    )
    skills_score, skills_terms = _score_text_lower(
        fields["skills_lower"],
        POSITIVE_EVIDENCE,
        SKILLS_WEIGHT,
    )

    weak_score = summary_score + skills_score

    negative_score = 0.0
    negative_terms = []

    for field_name, text in (
        ("current_title", fields["current_title_text"]),
        ("career_history.title", fields["history_title_text"]),
        ("career_history.description", fields["description_text"]),
        ("summary", fields["summary_lower"]),
        ("skills", fields["skills_lower"]),
    ):
        field_negative_score, field_negative_terms = _score_text_lower(text, NEGATIVE_EVIDENCE)

        if field_name in {"summary", "skills"}:
            field_negative_score *= 1.15

        negative_score += field_negative_score
        negative_terms.extend(field_negative_terms)

    professional_title_terms = list(dict.fromkeys(current_title_terms + history_title_terms))
    professional_description_terms = list(dict.fromkeys(description_terms))
    weak_terms = list(dict.fromkeys(summary_terms + skills_terms))

    for phrase in PROFESSIONAL_AI_TERMS:
        if phrase in fields["professional_text"]:
            professional_ai_terms.append(phrase)

    if jd_requirements:
        title_text = fields["title_text"]
        current_title_text = fields["current_title_text"]
        history_title_text = fields["history_title_text"]
        professional_text = fields["professional_text"]
        weak_text = fields["weak_text"]

        for skill in jd_requirements.get("skills", []):
            skill_text = skill.lower()

            if not skill_text:
                continue

            in_current_title = skill_text in current_title_text
            in_history_titles = skill_text in history_title_text
            in_descriptions = skill_text in description_text
            in_weak_fields = skill_text in weak_text

            if in_current_title:
                current_title_score += JD_TITLE_WEIGHT
                professional_title_terms.append(skill_text)
            elif in_history_titles:
                history_title_score += JD_HISTORY_TITLE_WEIGHT
                professional_title_terms.append(skill_text)
            elif in_descriptions:
                description_score += JD_DESCRIPTION_WEIGHT
                professional_description_terms.append(skill_text)
            elif in_weak_fields:
                weak_score += JD_WEAK_WEIGHT
                weak_terms.append(skill_text)

    professional_score = current_title_score + history_title_score + description_score
    score = professional_score + weak_score - negative_score

    if not professional_ai_terms and weak_score > 0:
        score = min(score, PROFESSIONAL_ONLY_CAP)

    if score < 0:
        score = 0.0
    elif score > 1.0:
        score = 1.0

    return {
        "score": score,
        "professional_score": min(professional_score, 1.0),
        "weak_score": min(weak_score, 1.0),
        "title_score": min(current_title_score + history_title_score, 1.0),
        "description_score": min(description_score, 1.0),
        "positive_terms": list(
            dict.fromkeys(
                professional_title_terms
                + professional_description_terms
                + weak_terms
                + professional_ai_terms
            )
        ),
        "negative_terms": list(dict.fromkeys(negative_terms)),
        "unrelated_terms": [],
        "professional_title_terms": list(dict.fromkeys(professional_title_terms)),
        "professional_description_terms": list(dict.fromkeys(professional_description_terms)),
        "weak_terms": list(dict.fromkeys(weak_terms)),
        "professional_ai_terms": list(dict.fromkeys(professional_ai_terms)),
    }


def career_evidence_score(candidate, jd_requirements=None):
    return career_evidence_breakdown(candidate, jd_requirements)["score"]