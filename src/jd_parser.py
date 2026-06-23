import re

IMPORTANT_SKILLS = [
    "python",
    "embeddings",
    "retrieval",
    "ranking",
    "vector",
    "milvus",
    "pinecone",
    "weaviate",
    "qdrant",
    "faiss",
    "elasticsearch",
    "opensearch",
    "llm",
    "fine-tuning",
    "lora",
    "qlora",
    "peft",
    "evaluation",
    "ndcg",
    "mrr",
    "map",
    "a/b testing",
    "recommendation",
    "semantic search"
]

def extract_requirements(jd_text):
    text = jd_text.lower()

    skills = []

    for skill in IMPORTANT_SKILLS:
        if skill.lower() in text:
            skills.append(skill)

    years = None

    match = re.search(r"(\d+)\s*[–-]\s*(\d+)\s*years", text)

    if match:
        years = (
            int(match.group(1)),
            int(match.group(2))
        )

    return {
        "skills": skills,
        "experience": years
    }