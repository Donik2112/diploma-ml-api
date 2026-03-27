import re
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 1. CONFIG
# =========================
DATA_FILE = "final_dataset_diploma_final_clean.xlsx"
SHEET_NAME = "cleaned_dataset"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# =========================
# 2. TEXT HELPERS
# =========================
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"[^a-zA-Zа-яА-Я0-9+#/.\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_city(city: str) -> str:
    city = clean_text(city)

    city_map = {
        "almaty": "алматы",
        "алматы": "алматы",
        "astana": "астана",
        "nursultan": "астана",
        "nur sultan": "астана",
        "астана": "астана",
        "shymkent": "шымкент",
        "шимкент": "шымкент",
        "шымкент": "шымкент",
        "karaganda": "караганда",
        "караганда": "караганда",
    }
    return city_map.get(city, city)


def normalize_experience(exp: str) -> str:
    exp = clean_text(exp)

    if any(x in exp for x in ["intern", "trainee", "junior", "jr", "стаж", "стажер", "стажёр"]):
        return "junior"
    if any(x in exp for x in ["middle", "mid", "мидл"]):
        return "middle"
    if any(x in exp for x in ["senior", "sr", "lead", "ведущий", "сеньор"]):
        return "senior"

    return exp


def normalize_employment(emp: str) -> str:
    emp = clean_text(emp)

    if any(x in emp for x in ["full-time", "full time", "полная", "full"]):
        return "full-time"
    if any(x in emp for x in ["part-time", "part time", "частичная", "part"]):
        return "part-time"
    if any(x in emp for x in ["remote", "удален", "удалён"]):
        return "remote"

    return emp


def normalize_skills_text(skills_text: str) -> str:
    if pd.isna(skills_text):
        return ""

    text = str(skills_text).lower()

    replacements = {
        r"\bjs\b": "javascript",
        r"\bts\b": "typescript",
        r"\bpy\b": "python",
        r"\bpostgre\b": "postgresql",
        r"\bpostgres\b": "postgresql",
        r"\bms sql\b": "sql",
        r"\bpowerbi\b": "power bi",
        r"\bml\b": "machine learning",
        r"\bai\b": "artificial intelligence",
        r"\bui ux\b": "ui ux",
        r"\bnodejs\b": "node js",
        r"\bnextjs\b": "next js",
    }

    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    return clean_text(text)


# =========================
# 3. DATA PREPARATION
# =========================
def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    needed_cols = [
        "job_title",
        "skills",
        "text",
        "experience_level",
        "employment_type",
        "city",
        "salary",
    ]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    df = df.copy()
    for col in needed_cols:
        df[col] = df[col].fillna("")

    df["job_title_clean"] = df["job_title"].apply(clean_text)
    df["skills_clean"] = df["skills"].apply(normalize_skills_text)
    df["text_clean"] = df["text"].apply(clean_text)
    df["experience_clean"] = df["experience_level"].apply(normalize_experience)
    df["employment_clean"] = df["employment_type"].apply(normalize_employment)
    df["city_clean"] = df["city"].apply(normalize_city)

    df = df.drop_duplicates(
        subset=[
            "job_title_clean",
            "skills_clean",
            "text_clean",
            "experience_clean",
            "employment_clean",
            "city_clean",
            "salary",
        ]
    ).reset_index(drop=True)

    return df


def build_student_profile(
    skills: str,
    experience: str = "",
    employment: str = "",
    city: str = "",
    interests: str = "",
) -> dict:
    return {
        "skills_clean": normalize_skills_text(skills),
        "experience_clean": normalize_experience(experience),
        "employment_clean": normalize_employment(employment),
        "city_clean": normalize_city(city),
        "interests_clean": clean_text(interests),
    }


# =========================
# 4. MODEL HELPERS
# =========================
def encode_texts(model: SentenceTransformer, texts: List[str]):
    texts = [t if isinstance(t, str) and t.strip() else "empty" for t in texts]
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def compute_meta_bonus(row, student_profile: dict) -> float:
    bonus = 0.0

    if student_profile["experience_clean"]:
        if row["experience_clean"] == student_profile["experience_clean"]:
            bonus += 0.03

    if student_profile["employment_clean"]:
        if row["employment_clean"] == student_profile["employment_clean"]:
            bonus += 0.02

    if student_profile["city_clean"]:
        if row["city_clean"] == student_profile["city_clean"]:
            bonus += 0.05

    return bonus


def compute_experience_penalty(row, student_profile: dict) -> float:
    """
    Soft penalty:
    if student is junior and vacancy is senior/middle, reduce score a bit
    """
    student_exp = student_profile["experience_clean"]
    vacancy_exp = row["experience_clean"]

    if student_exp == "junior":
        if vacancy_exp == "senior":
            return 0.08
        if vacancy_exp == "middle":
            return 0.04

    return 0.0


# =========================
# 5. RECOMMEND FUNCTION
# =========================
def recommend_jobs(
    df: pd.DataFrame,
    model: SentenceTransformer,
    student_skills: str,
    student_experience: str = "",
    student_employment: str = "",
    student_city: str = "",
    student_interests: str = "",
    top_n: int = 10,
    strict_city: bool = False,
):
    working_df = df.copy()

    student_profile = build_student_profile(
        skills=student_skills,
        experience=student_experience,
        employment=student_employment,
        city=student_city,
        interests=student_interests,
    )

    if strict_city and student_profile["city_clean"]:
        city_filtered = working_df[working_df["city_clean"] == student_profile["city_clean"]].copy()
        if not city_filtered.empty:
            working_df = city_filtered

    if working_df.empty:
        return []

    student_title_text = student_profile["skills_clean"]
    student_skills_text = student_profile["skills_clean"]
    student_text_text = " ".join(
        [
            student_profile["skills_clean"],
            student_profile["interests_clean"],
            student_profile["experience_clean"],
            student_profile["employment_clean"],
        ]
    ).strip()

    job_title_emb = encode_texts(model, working_df["job_title_clean"].tolist())
    skills_emb = encode_texts(model, working_df["skills_clean"].tolist())
    text_emb = encode_texts(model, working_df["text_clean"].tolist())

    student_title_emb = encode_texts(model, [student_title_text])
    student_skills_emb = encode_texts(model, [student_skills_text])
    student_text_emb = encode_texts(model, [student_text_text])

    title_scores = cosine_similarity(student_title_emb, job_title_emb).flatten()
    skills_scores = cosine_similarity(student_skills_emb, skills_emb).flatten()
    text_scores = cosine_similarity(student_text_emb, text_emb).flatten()

    final_scores = []

    for i, (_, row) in enumerate(working_df.iterrows()):
        meta_bonus = compute_meta_bonus(row, student_profile)
        exp_penalty = compute_experience_penalty(row, student_profile)

        final_score = (
            0.15 * title_scores[i]
            + 0.50 * skills_scores[i]
            + 0.30 * text_scores[i]
            + meta_bonus
            - exp_penalty
        )

        final_scores.append(final_score)

    working_df = working_df.copy()
    working_df["title_score"] = title_scores
    working_df["skills_score"] = skills_scores
    working_df["text_score"] = text_scores
    working_df["final_score"] = final_scores
    working_df["final_score_percent"] = (working_df["final_score"] * 100).round(2)

    result = working_df.sort_values(by="final_score", ascending=False).head(top_n).copy()

    response = []
    for _, row in result.iterrows():
        response.append(
            {
                "job_title": row["job_title"],
                "skills": row["skills"],
                "text": row["text"],
                "experience_level": row["experience_level"],
                "employment_type": row["employment_type"],
                "city": row["city"],
                "salary": row["salary"],
                "final_score": round(float(row["final_score"]), 4),
                "final_score_percent": round(float(row["final_score_percent"]), 2),
            }
        )

    return response


# =========================
# 6. API SCHEMAS
# =========================
class RecommendRequest(BaseModel):
    skills: str
    experience: Optional[str] = ""
    employment: Optional[str] = ""
    city: Optional[str] = ""
    interests: Optional[str] = ""
    top_n: Optional[int] = 10
    strict_city: Optional[bool] = False


class RecommendResponse(BaseModel):
    recommendations: List[dict]


# =========================
# 7. LOAD ON START
# =========================
print("Loading dataset...")
raw_df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)

print("Preparing dataset...")
prepared_df = prepare_dataset(raw_df)

print("Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)

print("API is ready.")


# =========================
# 8. FASTAPI APP
# =========================
app = FastAPI(title="Diploma Recommendation API", version="1.0")


@app.get("/")
def root():
    return {"message": "Recommendation API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rows_loaded": int(len(prepared_df)),
        "model": MODEL_NAME,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    results = recommend_jobs(
        df=prepared_df,
        model=embedding_model,
        student_skills=request.skills,
        student_experience=request.experience,
        student_employment=request.employment,
        student_city=request.city,
        student_interests=request.interests,
        top_n=request.top_n or 10,
        strict_city=request.strict_city or False,
    )
    return {"recommendations": results}