import re
import numpy as np
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

TOP_FAMILIES_K = 4           # было 3 → 4: рассматриваем больше направлений
MAX_CANDIDATES_PER_FAMILY = 30
DIVERSITY_THRESHOLD = 20     # если кандидатов < этого — расширяем поиск

# Эти 13 фич идут в CatBoostRanker БЕЗ ИЗМЕНЕНИЙ — модель обучена именно на них
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
    "преподаватель", "учитель", "педагог", "лектор",
]


# =========================
# 2. SKILL ALIASES  [НОВОЕ]
# Нормализует варианты написания к каноническому виду
# "js" → "javascript", "postgres" → "postgresql" и т.д.
# =========================
SKILL_ALIASES: dict[str, str] = {
    # JavaScript
    "js": "javascript", "es6": "javascript", "es2015": "javascript",
    "ecmascript": "javascript", "vanilla js": "javascript",
    # TypeScript
    "ts": "typescript",
    # Python
    "py": "python", "python3": "python", "python 3": "python",
    # Frontend
    "reactjs": "react", "react.js": "react",
    "vuejs": "vue", "vue.js": "vue",
    "angular2": "angular", "angularjs": "angular",
    "nextjs": "next.js",
    # Backend / Node
    "nodejs": "node.js", "node": "node.js",
    "expressjs": "express", "express.js": "express",
    "django rest": "django", "drf": "django",
    # Databases
    "postgres": "postgresql", "pg": "postgresql",
    "mongo": "mongodb", "mongoose": "mongodb",
    "mysql2": "mysql",
    "mssql": "sql server",
    "redis cache": "redis",
    "elastic": "elasticsearch",
    # ML / DS
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "tf": "tensorflow", "tensorflow2": "tensorflow",
    "torch": "pytorch",
    "sklearn": "scikit-learn", "sci-kit learn": "scikit-learn",
    "cv": "computer vision",
    "nlp": "natural language processing",
    "lgbm": "lightgbm",
    # DevOps / Cloud
    "k8s": "kubernetes", "kube": "kubernetes",
    "docker compose": "docker",
    "aws ec2": "aws", "amazon web services": "aws",
    "gcp": "google cloud", "google cloud platform": "google cloud",
    "azure devops": "azure",
    "ci/cd": "ci cd", "cicd": "ci cd",
    "github actions": "ci cd", "gitlab ci": "ci cd",
    # Mobile
    "rn": "react native",
    # .NET
    "c#": "csharp", ".net": "dotnet", "asp.net": "dotnet",
    # Java
    "spring boot": "spring", "jpa": "java", "hibernate": "java",
    # Misc
    "git hub": "git",
    "linux ubuntu": "linux",
    "bash scripting": "bash",
    "rest api": "rest", "restful": "rest",
    "graphql api": "graphql",
    "sql queries": "sql", "postgresql queries": "postgresql",
}


def normalize_skill_list(skills_text: str) -> List[str]:
    """Разбивает навыки, применяет алиасы, дедуплицирует."""
    parts = split_skills(skills_text)
    normalized = []
    for p in parts:
        lower = p.lower().strip()
        canonical = SKILL_ALIASES.get(lower, lower)
        normalized.append(canonical)
    return list(dict.fromkeys(normalized))


# =========================
# 3. SEMANTIC ENGINE  [НОВОЕ, ОПЦИОНАЛЬНО]
# Загружает многоязычную MiniLM для семантической близости.
# Работает без интернета после первой загрузки (~90 MB).
# Если sentence-transformers не установлен — деградирует к токенному оверлапу.
# =========================
SEMANTIC_ENABLED = False
SEMANTIC_MODEL = None
VACANCY_EMBEDDINGS: Optional[np.ndarray] = None   # (n_vacancies, 384)

try:
    from sentence_transformers import SentenceTransformer
    print("Loading semantic model paraphrase-multilingual-MiniLM-L12-v2 ...")
    SEMANTIC_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    SEMANTIC_ENABLED = True
    print("Semantic engine ready.")
except ImportError:
    print(
        "[INFO] sentence-transformers not found — semantic features disabled.\n"
        "       Install with: pip install sentence-transformers"
    )


def encode_texts(texts: List[str]) -> np.ndarray:
    if not SEMANTIC_ENABLED or SEMANTIC_MODEL is None:
        return np.zeros((len(texts), 1))
    return SEMANTIC_MODEL.encode(
        texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =========================
# 4. TEXT / NORMALIZATION HELPERS
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
    if any(x in exp for x in [
        "intern", "trainee", "junior", "jr",
        "стаж", "стажер", "стажёр", "нет опыта", "без опыта", "no experience",
    ]):
        return "junior"
    if any(x in exp for x in ["middle", "mid", "мидл", "от 1 до 3", "1 до 3"]):
        return "middle"
    if any(x in exp for x in [
        "senior", "sr", "lead", "ведущий", "сеньор", "от 3", "более 6",
    ]):
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
        "almaty": "алматы",    "алматы": "алматы",
        "astana": "астана",    "nursultan": "астана",
        "nur sultan": "астана","астана": "астана",
        "shymkent": "шымкент", "шимкент": "шымкент", "шымкент": "шымкент",
        "karaganda": "караганда", "караганда": "караганда",
    }
    return city_map.get(city, city)


def experience_to_num(exp: str) -> int:
    return {"": 0, "junior": 1, "middle": 2, "senior": 3}.get(exp, 0)


def count_token_overlap(text1: str, text2: str) -> int:
    t1 = token_set(text1)
    t2 = token_set(text2)
    return len(t1.intersection(t2)) if t1 and t2 else 0


# --- Skill overlap functions теперь используют normalize_skill_list ---

def skill_overlap_count(student_skills: str, vacancy_skills: str) -> int:
    s1 = set(normalize_skill_list(student_skills))
    s2 = set(normalize_skill_list(vacancy_skills))
    return len(s1.intersection(s2)) if s1 and s2 else 0


def skill_overlap_ratio(student_skills: str, vacancy_skills: str) -> float:
    s1 = set(normalize_skill_list(student_skills))
    s2 = set(normalize_skill_list(vacancy_skills))
    return len(s1.intersection(s2)) / max(len(s1), 1) if s1 and s2 else 0.0


def reverse_skill_overlap_ratio(student_skills: str, vacancy_skills: str) -> float:
    s1 = set(normalize_skill_list(student_skills))
    s2 = set(normalize_skill_list(vacancy_skills))
    return len(s1.intersection(s2)) / max(len(s2), 1) if s1 and s2 else 0.0


def compute_candidate_similarity(
    student_skills: str, student_interests: str,
    vacancy_title: str, vacancy_skills: str, vacancy_text: str,
) -> float:
    """Similarity score для pre-фильтрации кандидатов.
    ВАЖНО: формула не меняется — результат идёт в CatBoostRanker как фича
    с ожидаемым диапазоном. Алиасы улучшают skill_overlap_count внутри."""
    score = 0.0
    score += 3.0 * skill_overlap_count(student_skills, vacancy_skills)
    score += 2.0 * skill_overlap_ratio(student_skills, vacancy_skills)
    score += 1.5 * count_token_overlap(student_interests, vacancy_title)
    score += 1.0 * count_token_overlap(student_interests, vacancy_skills)
    score += 0.5 * count_token_overlap(student_interests, vacancy_text)
    return score


# =========================
# 5. STUDENT PROFILE HELPERS
# =========================
def build_student_profile_text(
    skills: str, interests: str = "", experience: str = "",
    employment: str = "", city: str = "",
) -> str:
    return (
        f"{clean_text(skills)} {clean_text(interests)} "
        f"{normalize_experience(experience)} "
        f"{normalize_employment(employment)} "
        f"{normalize_city(city)}"
    ).strip()


def build_student_profile_dict(
    skills: str, interests: str = "", experience: str = "",
    employment: str = "", city: str = "",
) -> dict:
    return {
        "student_skills":     clean_text(skills),
        "student_interests":  clean_text(interests),
        "student_experience": normalize_experience(experience),
        "student_employment": normalize_employment(employment),
        "student_city":       normalize_city(city),
    }


def profile_completeness(student_profile: dict) -> float:
    """Доля заполненных полей профиля [0.0, 1.0].
    Используется для адаптивного веса ML vs эвристика."""
    fields = ["student_skills", "student_interests", "student_experience", "student_city"]
    filled = sum(1 for f in fields if student_profile.get(f, "").strip())
    return filled / len(fields)


# =========================
# 6. VACANCY DATA PREPARATION
# =========================
def prepare_vacancies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    needed = ["job_title", "skills", "text", "experience_level",
              "employment_type", "city", "salary", "job_family"]
    for col in needed:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df = df[df["job_family"] != "other"].copy()

    df["job_title_clean"]   = df["job_title"].apply(clean_text)
    df["skills_clean"]      = df["skills"].apply(clean_text)
    df["text_clean"]        = df["text"].apply(clean_text)
    df["job_family_clean"]  = df["job_family"].apply(clean_text)
    df["experience_clean"]  = df["experience_level"].apply(normalize_experience)
    df["employment_clean"]  = df["employment_type"].apply(normalize_employment)
    df["city_clean"]        = df["city"].apply(normalize_city)
    # Сигнал наличия зарплаты (новая фича)
    df["has_salary"] = df["salary"].apply(
        lambda x: 1 if str(x).strip() not in ["", "nan", "0.0", "0"] else 0
    )
    df = df.reset_index(drop=True)
    return df


# =========================
# 7. LOAD ARTIFACTS
# =========================
print("Loading vacancies dataset...")
raw_vacancies_df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME)
vacancies_df = prepare_vacancies(raw_vacancies_df)
print(f"Loaded {len(vacancies_df)} vacancies ({vacancies_df['job_family_clean'].nunique()} families).")

# Предвычисляем эмбеддинги вакансий один раз при старте
if SEMANTIC_ENABLED:
    print(f"Precomputing embeddings for {len(vacancies_df)} vacancies...")
    _vac_texts = (vacancies_df["job_title_clean"] + " " + vacancies_df["skills_clean"]).tolist()
    VACANCY_EMBEDDINGS = encode_texts(_vac_texts)
    print("Vacancy embeddings ready.")

print("Loading student profile classifier...")
student_classifier = joblib.load(STUDENT_CLASSIFIER_FILE)

print("Loading ranker v3...")
ranker: CatBoostRanker = joblib.load(RANKER_FILE)

print("=== System v3.0 ready ===")


# =========================
# 8. FAMILY PREDICTION
# =========================
def predict_top_families(profile_text: str, top_k: int = TOP_FAMILIES_K):
    clf = student_classifier.named_steps["clf"]
    classes = list(clf.classes_)
    scores = student_classifier.decision_function([profile_text])
    score_row = scores.tolist() if scores.ndim == 1 else scores[0].tolist()
    pairs = sorted(zip(classes, score_row), key=lambda x: x[1], reverse=True)
    return pairs[:top_k]


# =========================
# 9. CANDIDATE SELECTION  [УЛУЧШЕНО — diversity fallback]
# Сохраняем оригинальные индексы vacancies_df для поиска эмбеддингов
# =========================
def select_candidate_vacancies(
    student_profile: dict, top_families: List[str],
    max_per_family: int = MAX_CANDIDATES_PER_FAMILY,
) -> pd.DataFrame:
    candidate_frames = []
    found_families: set = set()

    for fam in top_families:
        fam_clean = clean_text(fam)
        family_df = vacancies_df[vacancies_df["job_family_clean"] == fam_clean].copy()
        found_families.add(fam_clean)

        if family_df.empty:
            continue

        family_df["candidate_similarity"] = family_df.apply(
            lambda row: compute_candidate_similarity(
                student_profile["student_skills"],
                student_profile["student_interests"],
                row["job_title"], row["skills"], row["text"],
            ),
            axis=1,
        )
        # Для редких семей берём все записи; для крупных — top max_per_family
        n_take = min(max_per_family, len(family_df))
        family_df = family_df.sort_values("candidate_similarity", ascending=False).head(n_take)
        candidate_frames.append(family_df)

    # Diversity fallback: если кандидатов мало — добавляем по 3 из всех остальных семей
    total = sum(len(f) for f in candidate_frames)
    if total < DIVERSITY_THRESHOLD:
        for fam_clean in vacancies_df["job_family_clean"].unique():
            if fam_clean in found_families:
                continue
            extra = vacancies_df[vacancies_df["job_family_clean"] == fam_clean].copy()
            extra["candidate_similarity"] = extra.apply(
                lambda row: compute_candidate_similarity(
                    student_profile["student_skills"],
                    student_profile["student_interests"],
                    row["job_title"], row["skills"], row["text"],
                ),
                axis=1,
            )
            extra = extra.sort_values("candidate_similarity", ascending=False).head(3)
            candidate_frames.append(extra)

    if not candidate_frames:
        fallback = vacancies_df.copy().head(20)
        fallback["candidate_similarity"] = 0.0
        return fallback

    # Сохраняем оригинальные индексы (нужны для поиска эмбеддингов)
    candidates = pd.concat(candidate_frames)
    candidates = candidates[~candidates.index.duplicated(keep="first")]
    candidates = candidates.sort_values("candidate_similarity", ascending=False)
    return candidates


# =========================
# 10. FEATURE ENGINEERING ДЛЯ RANKER
# НЕ МЕНЯЕМ: модель обучена ровно на этих 13 фичах
# =========================
def build_ranker_features(student_profile: dict, candidates_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in candidates_df.iterrows():
        vac_title  = safe_str(row.get("job_title", ""))
        vac_skills = safe_str(row.get("skills", ""))
        vac_text   = safe_str(row.get("text", ""))
        vac_exp    = safe_str(row.get("experience_clean", ""))
        vac_emp    = safe_str(row.get("employment_clean", ""))
        vac_city   = safe_str(row.get("city_clean", ""))

        cand_sim = compute_candidate_similarity(
            student_profile["student_skills"], student_profile["student_interests"],
            vac_title, vac_skills, vac_text,
        )
        ov_count  = skill_overlap_count(student_profile["student_skills"], vac_skills)
        ov_ratio  = skill_overlap_ratio(student_profile["student_skills"], vac_skills)
        rev_ratio = reverse_skill_overlap_ratio(student_profile["student_skills"], vac_skills)

        int_title  = count_token_overlap(student_profile["student_interests"], vac_title)
        int_skills = count_token_overlap(student_profile["student_interests"], vac_skills)
        int_text   = count_token_overlap(student_profile["student_interests"], vac_text)
        title_ov   = count_token_overlap(student_profile["student_skills"], vac_title)

        city_match = int(
            student_profile["student_city"] and vac_city and
            student_profile["student_city"] == vac_city
        )
        emp_match = int(
            student_profile["student_employment"] and vac_emp and
            student_profile["student_employment"] == vac_emp
        )
        exp_match = int(
            student_profile["student_experience"] and vac_exp and
            student_profile["student_experience"] == vac_exp
        )

        s_exp_num = experience_to_num(student_profile["student_experience"])
        v_exp_num = experience_to_num(vac_exp)
        exp_gap   = v_exp_num - s_exp_num

        rows.append({
            "candidate_similarity":       cand_sim,
            "skill_overlap_count":        ov_count,
            "skill_overlap_ratio":        ov_ratio,
            "reverse_skill_overlap_ratio":rev_ratio,
            "interests_in_title":         int_title,
            "interests_in_skills":        int_skills,
            "interests_in_text":          int_text,
            "title_overlap":              title_ov,
            "city_match":                 city_match,
            "employment_match":           emp_match,
            "experience_match":           exp_match,
            "experience_gap":             exp_gap,
            "abs_experience_gap":         abs(exp_gap),
        })
    return pd.DataFrame(rows, index=candidates_df.index)


# =========================
# 11. ЭВРИСТИЧЕСКИЙ СКОР  [НОВОЕ]
# Чистая heuristic-оценка для адаптивного блендинга
# =========================
def compute_pure_heuristic(student_profile: dict, row: pd.Series) -> float:
    """Возвращает score; может быть отрицательным при сильном несоответствии уровня."""
    s_skills = set(normalize_skill_list(student_profile["student_skills"]))
    v_skills = set(normalize_skill_list(safe_str(row.get("skills", ""))))

    skill_score = len(s_skills & v_skills) / max(len(s_skills), 1) if s_skills else 0.0

    city_score = float(
        bool(student_profile["student_city"]) and
        normalize_city(safe_str(row.get("city", ""))) == student_profile["student_city"]
    )

    # Experience: совпадение +1.0, нейтрально 0.5, превышение −0.5 / −1.0
    s_exp = student_profile["student_experience"]
    v_exp = normalize_experience(safe_str(row.get("experience_level", "")))
    if not s_exp or not v_exp:
        exp_score = 0.5   # нет данных — нейтрально
    elif s_exp == v_exp:
        exp_score = 1.0   # точное попадание
    else:
        gap = experience_to_num(v_exp) - experience_to_num(s_exp)
        if gap == 1:
            exp_score = -0.5   # junior → middle: умеренный штраф
        elif gap >= 2:
            exp_score = -1.0   # junior → senior: сильный штраф
        else:
            exp_score = 0.7    # переквалификация (middle → junior): небольшой минус

    # Employment: совпадение +1.0, жёсткий конфликт (хочу remote, есть офис) −0.5
    s_emp = student_profile["student_employment"]
    v_emp = normalize_employment(safe_str(row.get("employment_type", "")))
    if not s_emp or not v_emp:
        emp_score = 0.5
    elif s_emp == v_emp:
        emp_score = 1.0
    elif s_emp in ("remote", "part-time") and v_emp == "full-time":
        emp_score = -0.5   # студент хочет remote/part-time, а вакансия требует офис
    else:
        emp_score = 0.3    # мягкое несоответствие

    return 0.60 * skill_score + 0.20 * city_score + 0.10 * exp_score + 0.10 * emp_score


# =========================
# 12. EXPLAINABILITY
# =========================
def detect_teaching_role(job_title: str) -> bool:
    return any(kw in clean_text(job_title) for kw in TEACHING_KEYWORDS)


def build_match_reason(student_profile: dict, row: pd.Series) -> str:
    reasons = []

    # Используем нормализованные навыки для более точного совпадения
    s_skills = set(normalize_skill_list(student_profile["student_skills"]))
    v_skills = set(normalize_skill_list(safe_str(row.get("skills", ""))))
    common = list(s_skills & v_skills)
    if common:
        reasons.append(f"совпадают навыки: {', '.join(common[:3])}")

    if safe_str(row.get("job_family", "")):
        reasons.append(f"направление: {safe_str(row.get('job_family', ''))}")

    if student_profile["student_city"] and (
        normalize_city(safe_str(row.get("city", ""))) == student_profile["student_city"]
    ):
        reasons.append(f"город: {student_profile['student_city']}")

    if student_profile["student_employment"] and (
        normalize_employment(safe_str(row.get("employment_type", ""))) == student_profile["student_employment"]
    ):
        reasons.append(f"занятость: {student_profile['student_employment']}")

    if student_profile["student_experience"] and (
        normalize_experience(safe_str(row.get("experience_level", ""))) == student_profile["student_experience"]
    ):
        reasons.append(f"опыт: {student_profile['student_experience']}")

    return "; ".join(reasons) if reasons else "релевантна по совокупности навыков и текста"


# =========================
# 13. АДАПТИВНОЕ ФИНАЛЬНОЕ РАНЖИРОВАНИЕ  [УЛУЧШЕНО]
#
# final_score = w_ml * norm_rank + w_h * heuristic + semantic_boost + coverage_bonus + salary_bonus
#
# w_ml зависит от полноты профиля:
#   пустой профиль  → w_ml=0.45 (больше доверяем эвристике)
#   полный профиль  → w_ml=0.75 (больше доверяем ML)
# =========================
def apply_business_adjustments(student_profile: dict, result_df: pd.DataFrame) -> pd.DataFrame:
    result_df = result_df.copy()

    # --- Адаптивное блендирование ML + эвристика ---
    completeness = profile_completeness(student_profile)
    w_ml = 0.45 + 0.30 * completeness   # [0.45, 0.75]
    w_h  = 1.0 - w_ml                   # [0.25, 0.55]

    # Нормализуем rank_score в [0,1] для корректного блендинга
    r_min, r_max = result_df["rank_score"].min(), result_df["rank_score"].max()
    r_range = r_max - r_min
    if len(result_df) > 1 and r_range > 1e-8:
        norm_rank = (result_df["rank_score"] - r_min) / r_range
    else:
        # Если один кандидат или все скоры равны — используем raw
        norm_rank = result_df["rank_score"].clip(0, 1)

    heuristic_scores = result_df.apply(
        lambda row: compute_pure_heuristic(student_profile, row), axis=1
    )

    result_df["final_score"] = w_ml * norm_rank + w_h * heuristic_scores

    # --- Семантический буст (если модель загружена) ---
    if SEMANTIC_ENABLED and VACANCY_EMBEDDINGS is not None:
        student_text = (
            student_profile["student_skills"] + " " + student_profile["student_interests"]
        ).strip()
        student_emb = encode_texts([student_text])[0]

        def semantic_boost(df_idx: int) -> float:
            if df_idx < len(VACANCY_EMBEDDINGS):
                return max(0.0, cosine_sim(student_emb, VACANCY_EMBEDDINGS[df_idx]))
            return 0.0

        sem_boosts = pd.Series(
            [semantic_boost(idx) for idx in result_df.index],
            index=result_df.index,
        )
        result_df["final_score"] += 0.25 * sem_boosts

    # --- Бонус за покрытие навыков студента (алиас-нормализованные) ---
    s_skills = set(normalize_skill_list(student_profile["student_skills"]))
    if s_skills:
        def coverage_bonus(row) -> float:
            v_skills = set(normalize_skill_list(safe_str(row.get("skills", ""))))
            return len(s_skills & v_skills) / max(len(s_skills), 1) if v_skills else 0.0

        cov = result_df.apply(coverage_bonus, axis=1)
        result_df["final_score"] += 0.10 * cov

    # --- Бонус за наличие зарплаты (небольшой, 0.03) ---
    if "has_salary" in result_df.columns:
        result_df["final_score"] += 0.03 * result_df["has_salary"]

    # --- Штраф за несоответствие уровня опыта [НОВОЕ] ---
    # junior → middle:  -0.25
    # junior → senior:  -0.50
    s_exp_num = experience_to_num(student_profile["student_experience"])
    if s_exp_num > 0:   # применяем только когда уровень указан
        vac_exp_nums = result_df.apply(
            lambda row: experience_to_num(
                normalize_experience(safe_str(row.get("experience_level", "")))
            ),
            axis=1,
        )
        exp_gap_vec = vac_exp_nums - s_exp_num
        result_df.loc[exp_gap_vec == 1,  "final_score"] -= 0.25
        result_df.loc[exp_gap_vec >= 2,  "final_score"] -= 0.50

    # --- Штраф за формат (хочу remote/part-time, а вакансия — офис) [НОВОЕ] ---
    s_emp = student_profile["student_employment"]
    if s_emp in ("remote", "part-time"):
        vac_emp_vec = result_df.apply(
            lambda row: normalize_employment(safe_str(row.get("employment_type", ""))),
            axis=1,
        )
        result_df.loc[vac_emp_vec == "full-time", "final_score"] -= 0.20

    # --- Штраф за teaching-роли (без изменений) ---
    user_text = clean_text(
        f"{student_profile['student_skills']} {student_profile['student_interests']}"
    )
    user_wants_teaching = any(kw in user_text for kw in TEACHING_KEYWORDS)
    if not user_wants_teaching:
        teaching_mask = result_df["job_title"].apply(detect_teaching_role)
        result_df.loc[teaching_mask, "final_score"] -= 1.2

    # --- Бонус за совпадение города (оставлен без изменений) ---
    city_mask = result_df.apply(
        lambda row: bool(
            student_profile["student_city"] and
            normalize_city(safe_str(row.get("city", ""))) == student_profile["student_city"]
        ),
        axis=1,
    )
    result_df.loc[city_mask, "final_score"] += 0.10

    return result_df


# =========================
# 14. ОСНОВНАЯ ФУНКЦИЯ РЕКОМЕНДАЦИЙ
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
        student_skills, student_interests, student_experience,
        student_employment, student_city,
    )
    student_profile = build_student_profile_dict(
        student_skills, student_interests, student_experience,
        student_employment, student_city,
    )

    top_family_scores = predict_top_families(profile_text)
    top_families = [fam for fam, _ in top_family_scores]

    # Оригинальные индексы vacancies_df сохраняются для эмбеддингов
    candidate_df = select_candidate_vacancies(student_profile, top_families)

    feature_df = build_ranker_features(student_profile, candidate_df)
    feature_df = feature_df[RANKER_FEATURE_COLS].fillna(0)

    rank_scores = ranker.predict(feature_df)

    result_df = candidate_df.copy()
    result_df["rank_score"] = rank_scores

    result_df = apply_business_adjustments(student_profile, result_df)

    result_df = result_df.sort_values(
        by=["final_score", "candidate_similarity"], ascending=[False, False]
    ).head(top_n).copy()

    recommendations = []
    for _, row in result_df.iterrows():
        recommendations.append({
            "job_title":        safe_str(row.get("job_title", "")),
            "skills":           safe_str(row.get("skills", "")),
            "text":             safe_str(row.get("text", "")),
            "experience_level": safe_str(row.get("experience_level", "")),
            "employment_type":  safe_str(row.get("employment_type", "")),
            "city":             safe_str(row.get("city", "")),
            "salary":           safe_str(row.get("salary", "")),
            "job_family":       safe_str(row.get("job_family", "")),
            "candidate_similarity": round(safe_float(row.get("candidate_similarity", 0.0)), 4),
            "rank_score":       round(safe_float(row.get("rank_score", 0.0)), 4),
            "final_score":      round(safe_float(row.get("final_score", 0.0)), 4),
            "match_reason":     build_match_reason(student_profile, row),
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
# 15. API SCHEMAS
# =========================
class RecommendRequest(BaseModel):
    skills: str
    interests:  Optional[str] = ""
    experience: Optional[str] = ""
    employment: Optional[str] = ""
    city:       Optional[str] = ""
    top_n:      Optional[int] = 10


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


class Project(BaseModel):
    project_id: str
    title: str
    skills:           str  = ""
    text:             str  = ""
    experience_level: str  = ""
    employment_type:  str  = ""
    city:             str  = ""
    category: Optional[str]   = ""
    budget_min: Optional[float] = 0
    budget_max: Optional[float] = 0


class RecommendProjectsRequest(BaseModel):
    skills: str
    interests:  Optional[str] = ""
    experience: Optional[str] = ""
    employment: Optional[str] = ""
    city:       Optional[str] = ""
    top_n:      Optional[int] = 10
    projects: List[Project]


# =========================
# 16. FASTAPI APP
# =========================
app = FastAPI(title="UniWork Job Recommendation API", version="3.0")


@app.get("/")
def root():
    return {"message": "UniWork recommendation API v3.0 is running"}


@app.get("/health")
def health():
    return {
        "status":           "ok",
        "version":          "3.0",
        "vacancies_loaded": int(len(vacancies_df)),
        "semantic_engine":  SEMANTIC_ENABLED,
        "classifier":       STUDENT_CLASSIFIER_FILE,
        "ranker":           RANKER_FILE,
        "top_families_k":   TOP_FAMILIES_K,
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


@app.post("/recommend-projects")
def recommend_projects(request: RecommendProjectsRequest):
    try:
        student_profile = build_student_profile_dict(
            skills=request.skills,
            interests=request.interests or "",
            experience=request.experience or "",
            employment=request.employment or "",
            city=request.city or "",
        )

        if not request.projects:
            return {"count": 0, "recommendations": []}

        df = pd.DataFrame([p.dict() for p in request.projects])
        df["job_title"]        = df["title"]
        df["skills"]           = df["skills"].fillna("")
        df["text"]             = df["text"].fillna("")
        df["experience_level"] = df["experience_level"].fillna("")
        df["employment_type"]  = df["employment_type"].fillna("")
        df["city"]             = df["city"].fillna("")
        df["job_family"]       = df["category"].fillna("other")
        df["experience_clean"] = df["experience_level"].apply(normalize_experience)
        df["employment_clean"] = df["employment_type"].apply(normalize_employment)
        df["city_clean"]       = df["city"].apply(normalize_city)
        df["has_salary"]       = 0  # у клиентских проектов нет зарплаты из датасета

        df["candidate_similarity"] = df.apply(
            lambda row: compute_candidate_similarity(
                student_profile["student_skills"],
                student_profile["student_interests"],
                row["job_title"], row["skills"], row["text"],
            ),
            axis=1,
        )

        feature_df = build_ranker_features(student_profile, df)
        feature_df = feature_df[RANKER_FEATURE_COLS].fillna(0)
        rank_scores = ranker.predict(feature_df)

        df["rank_score"] = rank_scores

        # Для проектов семантику считаем on-the-fly (маленький список)
        if SEMANTIC_ENABLED:
            student_text = (
                student_profile["student_skills"] + " " + student_profile["student_interests"]
            ).strip()
            student_emb = encode_texts([student_text])[0]
            project_texts = (df["job_title"] + " " + df["skills"]).tolist()
            project_embs = encode_texts(project_texts)
            df["sem_boost"] = [
                max(0.0, cosine_sim(student_emb, pe)) for pe in project_embs
            ]
        else:
            df["sem_boost"] = 0.0

        # Временно убираем index-based semantic из apply_business_adjustments для проектов
        # и добавляем вручную
        df_adj = apply_business_adjustments(student_profile, df)
        df_adj["final_score"] += 0.25 * df["sem_boost"]

        df_adj = df_adj.sort_values(
            by=["final_score", "candidate_similarity"], ascending=[False, False]
        ).head(request.top_n or 10)

        results = []
        for _, row in df_adj.iterrows():
            results.append({
                "project_id":       str(row.get("project_id", "")),
                "title":            safe_str(row.get("title", "")),
                "text":             safe_str(row.get("text", "")),
                "city":             safe_str(row.get("city", "")),
                "employment_type":  safe_str(row.get("employment_type", "")),
                "experience_level": safe_str(row.get("experience_level", "")),
                "category":         safe_str(row.get("category", "")),
                "budget_min":       safe_float(row.get("budget_min", 0)),
                "budget_max":       safe_float(row.get("budget_max", 0)),
                "final_score":      round(safe_float(row.get("final_score", 0.0)), 4),
                "match_reason":     build_match_reason(student_profile, row),
            })

        return {"count": len(results), "recommendations": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
