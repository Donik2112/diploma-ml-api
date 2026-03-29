import re
import joblib
import pandas as pd

from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRanker


# =========================
# 1. CONFIG
# =========================
DATA_FILE = "final_dataset_with_job_family.xlsx"
SHEET_NAME = 0

STUDENT_CLASSIFIER_FILE = "student_profile_classifier_svm.pkl"
RANKER_FILE = "catboost_ranker_v3.pkl"

TOP_FAMILIES_K = 3
MAX_CANDIDATES_PER_FAMILY = 30

RANKER_FEATURE_COLS = [
    "candidate_similarity",
    "skill_overlap_count",
    "skill_overlap_ratio",
    "reverse_skill_overlap_ratio",
    "interests_in_title",
    "interests_in_skills",
    "interests_in_text",
    "title_overlap",
    "city_match",
    "employment_match",
    "experience_match",
    "experience_gap",
    "abs_experience_gap",
]

TEACHING_KEYWORDS = [
    "teacher", "teaching", "lecturer", "tutor", "teaching assistant",
    "преподаватель", "учитель", "педагог", "лектор"
]


# =========================
# 2. TEXT / NORMALIZATION HELPERS
# =========================
def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def safe_float(value, default=0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def split_skills(skills_text: str) -> List[str]:
    skills_text = clean_text(skills_text)
    if not skills_text:
        return []

    parts = re.split(r"[,;/|]+", skills_text)
    parts = [p.strip() for p in parts if p.strip()]
    return list(dict.fromkeys(parts))


def token_set(text: str) -> set:
    text = clean_text(text)
    if not text:
        return set()
    return set(re.findall(r"[a-zA-Zа-яА-Я0-9+#.\-]+", text))


def normalize_experience(exp: str) -> str:
    exp = clean_text(exp)

    if any(x in exp for x in ["intern", "trainee", "junior", "jr", "стаж", "стажер", "стажёр"]):
        return "junior"
    if any(x in exp for x in ["middle", "mid", "мидл"]):
        return "middle"
    if any(x in exp for x in ["senior", "sr", "lead", "ведущий", "сеньор"]):
        return "senior"

    return ""


def normalize_employment(emp: str) -> str:
    emp = clean_text(emp)

    if any(x in emp for x in ["full-time", "full time", "полная", "full"]):
        return "full-time"
    if any(x in emp for x in ["part-time", "part time", "частичная", "part"]):
        return "part-time"
    if any(x in emp for x in ["remote", "удален", "удалён"]):
        return "remote"

    return ""


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


def experience_to_num(exp: str) -> int:
    mapping = {
        "": 0,
        "junior": 1,
        "middle": 2,
        "senior": 3,
    }
    return mapping.get(exp, 0)


def count_token_overlap(text1: str, text2: str) -> int:
    t1 = token_set(text1)
    t2 = token_set(text2)

    if not t1 or not t2:
        return 0

    return len(t1.intersection(t2))


def skill_overlap_count(student_skills: str, vacancy_skills: str) -> int:
    s1 = set(split_skills(student_skills))
    s2 = set(split_skills(vacancy_skills))

    if not s1 or not s2:
        return 0

    return len(s1.intersection(s2))


def skill_overlap_ratio(student_skills: str, vacancy_skills: str) -> float:
    s1 = set(split_skills(student_skills))
    s2 = set(split_skills(vacancy_skills))

    if not s1 or not s2:
        return 0.0

    return len(s1.intersection(s2)) / max(len(s1), 1)


def reverse_skill_overlap_ratio(student_skills: str, vacancy_skills: str) -> float:
    s1 = set(split_skills(student_skills))
    s2 = set(split_skills(vacancy_skills))

    if not s1 or not s2:
        return 0.0

    return len(s1.intersection(s2)) / max(len(s2), 1)


def compute_candidate_similarity(student_skills: str, student_interests: str, vacancy_title: str, vacancy_skills: str, vacancy_text: str) -> float:
    score = 0.0
    score += 3.0 * skill_overlap_count(student_skills, vacancy_skills)
    score += 2.0 * skill_overlap_ratio(student_skills, vacancy_skills)
    score += 1.5 * count_token_overlap(student_interests, vacancy_title)
    score += 1.0 * count_token_overlap(student_interests, vacancy_skills)
    score += 0.5 * count_token_overlap(student_interests, vacancy_text)
    return score


# =========================
# 3. STUDENT PROFILE HELPERS
# =========================
def build_student_profile_text(
    skills: str,
    interests: str = "",
    experience: str = "",
    employment: str = "",
    city: str = "",
) -> str:
    student_skills = clean_text(skills)
    student_interests = clean_text(interests)
    student_experience = normalize_experience(experience)
    student_employment = normalize_employment(employment)
    student_city = normalize_city(city)

    return f"{student_skills} {student_interests} {student_experience} {student_employment} {student_city}".strip()


def build_student_profile_dict(
    skills: str,
    interests: str = "",
    experience: str = "",
    employment: str = "",
    city: str = "",
) -> dict:
    return {
        "student_skills": clean_text(skills),
        "student_interests": clean_text(interests),
        "student_experience": normalize_experience(experience),
        "student_employment": normalize_employment(employment),
        "student_city": normalize_city(city),
    }


# =========================
# 4. VACANCY DATA PREPARATION
# =========================
def prepare_vacancies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    needed_cols = [
        "job_title",
        "skills",
        "text",
        "experience_level",
        "employment_type",
        "city",
        "salary",
        "job_family",
    ]

    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    for col in needed_cols:
        df[col] = df[col].fillna("")

    df = df[df["job_family"] != "other"].copy()

    df["job_title_clean"] = df["job_title"].apply(clean_text)
    df["skills_clean"] = df["skills"].apply(clean_text)
    df["text_clean"] = df["text"].apply(clean_text)
    df["job_family_clean"] = df["job_family"].apply(clean_text)
    df["experience_clean"] = df["experience_level"].apply(normalize_experience)
    df["employment_clean"] = df["employment_type"].apply(normalize_employment)
    df["city_clean"] = df["city"].apply(normalize_city)

    df = df.reset_index(drop=True)
    return df


# =========================
# 5. LOAD ARTIFACTS
# =========================
print("Loading vacancies dataset...")
raw_vacancies_df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
vacancies_df = prepare_vacancies(raw_vacancies_df)

print("Loading student profile classifier...")
student_classifier = joblib.load(STUDENT_CLASSIFIER_FILE)

print("Loading ranker v3...")
ranker: CatBoostRanker = joblib.load(RANKER_FILE)

print("System is ready.")


# =========================
# 6. FAMILY PREDICTION
# =========================
def predict_top_families(profile_text: str, top_k: int = 3):
    clf = student_classifier.named_steps["clf"]
    classes = list(clf.classes_)

    scores = student_classifier.decision_function([profile_text])

    if hasattr(scores, "ndim") and scores.ndim == 1:
        score_row = scores.tolist()
    else:
        score_row = scores[0].tolist()

    pairs = list(zip(classes, score_row))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    return pairs[:top_k]


# =========================
# 7. CANDIDATE SELECTION
# =========================
def select_candidate_vacancies(student_profile: dict, top_families: List[str], max_per_family: int = 30) -> pd.DataFrame:
    candidate_frames = []

    for fam in top_families:
        family_df = vacancies_df[vacancies_df["job_family_clean"] == clean_text(fam)].copy()

        if family_df.empty:
            continue

        family_df["candidate_similarity"] = family_df.apply(
            lambda row: compute_candidate_similarity(
                student_skills=student_profile["student_skills"],
                student_interests=student_profile["student_interests"],
                vacancy_title=row["job_title"],
                vacancy_skills=row["skills"],
                vacancy_text=row["text"],
            ),
            axis=1
        )

        family_df = family_df.sort_values("candidate_similarity", ascending=False).head(max_per_family)
        candidate_frames.append(family_df)

    if not candidate_frames:
        fallback = vacancies_df.copy().head(20)
        fallback["candidate_similarity"] = 0.0
        return fallback

    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates = candidates.drop_duplicates(subset=["job_title", "skills", "text", "job_family"])
    candidates = candidates.sort_values("candidate_similarity", ascending=False).reset_index(drop=True)

    return candidates


# =========================
# 8. FEATURE ENGINEERING FOR RANKER
# =========================
def build_ranker_features(student_profile: dict, candidates_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in candidates_df.iterrows():
        vacancy_title = safe_str(row.get("job_title", ""))
        vacancy_skills = safe_str(row.get("skills", ""))
        vacancy_text = safe_str(row.get("text", ""))
        vacancy_experience = safe_str(row.get("experience_clean", ""))
        vacancy_employment = safe_str(row.get("employment_clean", ""))
        vacancy_city = safe_str(row.get("city_clean", ""))

        cand_similarity = compute_candidate_similarity(
            student_skills=student_profile["student_skills"],
            student_interests=student_profile["student_interests"],
            vacancy_title=vacancy_title,
            vacancy_skills=vacancy_skills,
            vacancy_text=vacancy_text,
        )

        overlap_count = skill_overlap_count(student_profile["student_skills"], vacancy_skills)
        overlap_ratio = skill_overlap_ratio(student_profile["student_skills"], vacancy_skills)
        reverse_ratio = reverse_skill_overlap_ratio(student_profile["student_skills"], vacancy_skills)

        interests_in_title = count_token_overlap(student_profile["student_interests"], vacancy_title)
        interests_in_skills = count_token_overlap(student_profile["student_interests"], vacancy_skills)
        interests_in_text = count_token_overlap(student_profile["student_interests"], vacancy_text)
        title_overlap = count_token_overlap(student_profile["student_skills"], vacancy_title)

        city_match = int(
            student_profile["student_city"] != "" and
            vacancy_city != "" and
            student_profile["student_city"] == vacancy_city
        )

        employment_match = int(
            student_profile["student_employment"] != "" and
            vacancy_employment != "" and
            student_profile["student_employment"] == vacancy_employment
        )

        experience_match = int(
            student_profile["student_experience"] != "" and
            vacancy_experience != "" and
            student_profile["student_experience"] == vacancy_experience
        )

        student_exp_num = experience_to_num(student_profile["student_experience"])
        vacancy_exp_num = experience_to_num(vacancy_experience)
        experience_gap = vacancy_exp_num - student_exp_num
        abs_experience_gap = abs(experience_gap)

        rows.append({
            "candidate_similarity": cand_similarity,
            "skill_overlap_count": overlap_count,
            "skill_overlap_ratio": overlap_ratio,
            "reverse_skill_overlap_ratio": reverse_ratio,
            "interests_in_title": interests_in_title,
            "interests_in_skills": interests_in_skills,
            "interests_in_text": interests_in_text,
            "title_overlap": title_overlap,
            "city_match": city_match,
            "employment_match": employment_match,
            "experience_match": experience_match,
            "experience_gap": experience_gap,
            "abs_experience_gap": abs_experience_gap,
        })

    return pd.DataFrame(rows)


# =========================
# 9. EXPLAINABILITY HELPERS
# =========================
def detect_teaching_role(job_title: str) -> bool:
    title = clean_text(job_title)
    return any(keyword in title for keyword in TEACHING_KEYWORDS)


def build_match_reason(student_profile: dict, row: pd.Series) -> str:
    reasons = []

    student_skills_set = set(split_skills(student_profile["student_skills"]))
    vacancy_skills_set = set(split_skills(safe_str(row.get("skills", ""))))
    common_skills = list(student_skills_set.intersection(vacancy_skills_set))

    if common_skills:
        shown_skills = common_skills[:3]
        reasons.append(f"совпадают навыки: {', '.join(shown_skills)}")

    if safe_str(row.get("job_family", "")):
        reasons.append(f"подходит направление: {safe_str(row.get('job_family', ''))}")

    vacancy_city = normalize_city(safe_str(row.get("city", "")))
    if student_profile["student_city"] and vacancy_city and student_profile["student_city"] == vacancy_city:
        reasons.append(f"совпадает город: {vacancy_city}")

    vacancy_emp = normalize_employment(safe_str(row.get("employment_type", "")))
    if student_profile["student_employment"] and vacancy_emp and student_profile["student_employment"] == vacancy_emp:
        reasons.append(f"подходит тип занятости: {vacancy_emp}")

    vacancy_exp = normalize_experience(safe_str(row.get("experience_level", "")))
    if student_profile["student_experience"] and vacancy_exp and student_profile["student_experience"] == vacancy_exp:
        reasons.append(f"подходит уровень опыта: {vacancy_exp}")

    if not reasons:
        reasons.append("релевантна по совокупности навыков и текста вакансии")

    return "; ".join(reasons)


def apply_business_adjustments(student_profile: dict, result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Небольшая пост-обработка для более естественной выдачи.
    """
    result_df = result_df.copy()
    result_df["final_score"] = result_df["rank_score"]

    # Мягкий штраф для teaching-ролей, если пользователь явно не указывал интерес к преподаванию
    user_text = f"{student_profile['student_skills']} {student_profile['student_interests']}"
    user_text = clean_text(user_text)

    user_wants_teaching = any(keyword in user_text for keyword in TEACHING_KEYWORDS)

    if not user_wants_teaching:
        teaching_mask = result_df["job_title"].apply(detect_teaching_role)
        result_df.loc[teaching_mask, "final_score"] = result_df.loc[teaching_mask, "final_score"] - 1.2

    # Мягкий бонус за совпадение города
    city_mask = result_df.apply(
        lambda row: int(
            student_profile["student_city"] != "" and
            normalize_city(safe_str(row.get("city", ""))) == student_profile["student_city"]
        ),
        axis=1
    ).astype(bool)
    result_df.loc[city_mask, "final_score"] = result_df.loc[city_mask, "final_score"] + 0.10

    return result_df


# =========================
# 10. FINAL RECOMMENDATION FUNCTION
# =========================
def recommend_jobs(
    student_skills: str,
    student_interests: str = "",
    student_experience: str = "",
    student_employment: str = "",
    student_city: str = "",
    top_n: int = 10,
):
    profile_text = build_student_profile_text(
        skills=student_skills,
        interests=student_interests,
        experience=student_experience,
        employment=student_employment,
        city=student_city,
    )

    student_profile = build_student_profile_dict(
        skills=student_skills,
        interests=student_interests,
        experience=student_experience,
        employment=student_employment,
        city=student_city,
    )

    top_family_scores = predict_top_families(profile_text, top_k=TOP_FAMILIES_K)
    top_families = [family for family, _ in top_family_scores]

    candidate_df = select_candidate_vacancies(
        student_profile=student_profile,
        top_families=top_families,
        max_per_family=MAX_CANDIDATES_PER_FAMILY,
    )

    feature_df = build_ranker_features(student_profile, candidate_df)
    feature_df = feature_df[RANKER_FEATURE_COLS].fillna(0)

    rank_scores = ranker.predict(feature_df)

    result_df = candidate_df.copy()
    result_df["rank_score"] = rank_scores

    result_df = apply_business_adjustments(student_profile, result_df)

    result_df = result_df.sort_values(
        by=["final_score", "candidate_similarity"],
        ascending=[False, False]
    ).head(top_n).copy()

    recommendations = []
    for _, row in result_df.iterrows():
        recommendations.append({
            "job_title": safe_str(row.get("job_title", "")),
            "skills": safe_str(row.get("skills", "")),
            "text": safe_str(row.get("text", "")),
            "experience_level": safe_str(row.get("experience_level", "")),
            "employment_type": safe_str(row.get("employment_type", "")),
            "city": safe_str(row.get("city", "")),
            "salary": safe_str(row.get("salary", "")),
            "job_family": safe_str(row.get("job_family", "")),
            "candidate_similarity": round(safe_float(row.get("candidate_similarity", 0.0)), 4),
            "rank_score": round(safe_float(row.get("rank_score", 0.0)), 4),
            "final_score": round(safe_float(row.get("final_score", 0.0)), 4),
            "match_reason": build_match_reason(student_profile, row),
        })

    family_predictions = [
        {"job_family": safe_str(fam), "score": round(safe_float(score), 4)}
        for fam, score in top_family_scores
    ]

    return {
        "predicted_job_families": family_predictions,
        "recommendations": recommendations,
    }


# =========================
# 11. API SCHEMAS
# =========================
class RecommendRequest(BaseModel):
    skills: str
    interests: Optional[str] = ""
    experience: Optional[str] = ""
    employment: Optional[str] = ""
    city: Optional[str] = ""
    top_n: Optional[int] = 10


class JobRecommendation(BaseModel):
    job_title: str
    skills: str
    text: str
    experience_level: str
    employment_type: str
    city: str
    salary: Optional[str] = ""
    job_family: str
    candidate_similarity: float
    rank_score: float
    final_score: float
    match_reason: str


class PredictedFamily(BaseModel):
    job_family: str
    score: float


class RecommendResponse(BaseModel):
    predicted_job_families: List[PredictedFamily]
    recommendations: List[JobRecommendation]


# =========================
# 12. FASTAPI APP
# =========================
app = FastAPI(title="Diploma Job Recommendation API", version="2.2")


@app.get("/")
def root():
    return {"message": "Diploma recommendation API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "vacancies_loaded": int(len(vacancies_df)),
        "student_classifier": STUDENT_CLASSIFIER_FILE,
        "ranker": RANKER_FILE,
    }


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        return recommend_jobs(
            student_skills=request.skills,
            student_interests=request.interests or "",
            student_experience=request.experience or "",
            student_employment=request.employment or "",
            student_city=request.city or "",
            top_n=request.top_n or 10,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

# =========================
# 13. PROJECT-BASED RECOMMENDATIONS (NEW)
# =========================

class Project(BaseModel):
    project_id: str
    title: str
    skills: str = ""
    text: str = ""
    experience_level: str = ""
    employment_type: str = ""
    city: str = ""
    category: Optional[str] = ""
    budget_min: Optional[float] = 0
    budget_max: Optional[float] = 0


class RecommendProjectsRequest(BaseModel):
    skills: str
    interests: Optional[str] = ""
    experience: Optional[str] = ""
    employment: Optional[str] = ""
    city: Optional[str] = ""
    top_n: Optional[int] = 10
    projects: List[Project]


@app.post("/recommend-projects")
def recommend_projects(request: RecommendProjectsRequest):
    try:
        # === 1. student profile ===
        student_profile = build_student_profile_dict(
            skills=request.skills,
            interests=request.interests or "",
            experience=request.experience or "",
            employment=request.employment or "",
            city=request.city or "",
        )

        # === 2. projects → DataFrame ===
        if not request.projects:
            return {"recommendations": []}

        df = pd.DataFrame([p.dict() for p in request.projects])

        # нормализация колонок
        df["job_title"] = df["title"]
        df["skills"] = df["skills"].fillna("")
        df["text"] = df["text"].fillna("")
        df["experience_level"] = df["experience_level"].fillna("")
        df["employment_type"] = df["employment_type"].fillna("")
        df["city"] = df["city"].fillna("")
        df["job_family"] = df["category"].fillna("other")

        # чистка
        df["experience_clean"] = df["experience_level"].apply(normalize_experience)
        df["employment_clean"] = df["employment_type"].apply(normalize_employment)
        df["city_clean"] = df["city"].apply(normalize_city)

        # === 3. candidate similarity ===
        df["candidate_similarity"] = df.apply(
            lambda row: compute_candidate_similarity(
                student_skills=student_profile["student_skills"],
                student_interests=student_profile["student_interests"],
                vacancy_title=row["job_title"],
                vacancy_skills=row["skills"],
                vacancy_text=row["text"],
            ),
            axis=1
        )

        # === 4. feature engineering ===
        feature_df = build_ranker_features(student_profile, df)
        feature_df = feature_df[RANKER_FEATURE_COLS].fillna(0)

        # === 5. ranking ===
        rank_scores = ranker.predict(feature_df)

        df["rank_score"] = rank_scores
        df = apply_business_adjustments(student_profile, df)

        df = df.sort_values(
            by=["final_score", "candidate_similarity"],
            ascending=[False, False]
        ).head(request.top_n or 10)

        # === 6. response ===
        results = []
        for _, row in df.iterrows():
            results.append({
                "project_id": str(row.get("project_id", "")),
                "title": safe_str(row.get("title", "")),
                "text": safe_str(row.get("text", "")),
                "city": safe_str(row.get("city", "")),
                "employment_type": safe_str(row.get("employment_type", "")),
                "experience_level": safe_str(row.get("experience_level", "")),
                "category": safe_str(row.get("category", "")),
                "budget_min": safe_float(row.get("budget_min", 0)),
                "budget_max": safe_float(row.get("budget_max", 0)),
                "final_score": round(safe_float(row.get("final_score", 0.0)), 4),
                "match_reason": build_match_reason(student_profile, row),
            })

        return {
            "count": len(results),
            "recommendations": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
