import json, random, time, copy
import ollama
from tqdm import tqdm

# =========================
# 1) ASPECTS + ANCHORS
# =========================
ASPECTS = [
    "Cost of Living", "Healthcare", "Education", "Personal Security",
    "Employment", "Transportation", "Government", "Environment",
    "Social Equality", "Taxation"
]

ASPECT_GROUPS = {
    "econ": ["Cost of Living", "Employment", "Taxation"],
    "services": ["Healthcare", "Education", "Transportation"],
    "values": ["Social Equality", "Environment", "Government"],
}

ASPECT_ANCHORS = {
    "Cost of Living": ["rent", "groceries", "bills", "prices", "utilities", "inflation"],
    "Healthcare": ["waiting months", "clinic", "doctor visit", "hospital staff shortage", "insurance", "prescription costs"],
    "Education": ["school quality", "teachers", "tuition", "class sizes", "exams", "campus resources"],
    "Personal Security": ["neighborhood safety", "crime", "street harassment", "police response", "feeling safe at night"],
    "Employment": ["job market", "layoffs", "hiring freeze", "wages", "job security", "work conditions"],
    "Transportation": ["commute", "bus delays", "train", "traffic", "new metro line", "public transport reliability"],
    "Government": ["bureaucracy", "public services", "policy decisions", "corruption", "efficiency", "paperwork"],
    "Environment": ["air quality", "pollution", "heatwaves", "recycling", "green spaces", "climate"],
    "Social Equality": ["inequality", "discrimination", "equal rights", "access for everyone", "poverty gap", "fairness"],
    "Taxation": ["paycheck deductions", "tax refund", "VAT", "tax burden", "forms", "small business taxes"],
}

# =========================
# 2) PERSONA_HINTS + PERSONAS + PERSONA_PRIORS
# =========================

PERSONA_HINTS = {
    "struggling student": (
        "You are a struggling student. You constantly worry about rent, groceries, and tuition. "
        "You talk about classes, exams, campus life, part-time jobs, and the pressure of rising prices. "
        "Your tone can be tired but practical."
    ),
    "wealthy business owner": (
        "You are a wealthy business owner. You focus on taxes, regulation, government decisions, hiring, and markets. "
        "You speak confidently, sometimes dismissive of complaints, and you mention business costs, payroll, and bureaucracy."
    ),
    "tired parent": (
        "You are a tired parent. Your priorities are your children's safety, school quality, and household expenses. "
        "You talk about childcare, commuting with kids, neighborhood safety, and how prices affect the family budget. "
        "Your tone is stressed, direct, and personal."
    ),
    "political activist": (
        "You are a political activist. You care about social equality, public policy, environment, and systemic problems. "
        "You often frame issues in terms of fairness, rights, accountability, and long-term consequences. "
        "Your tone is passionate and persuasive."
    ),
    "retired teacher": (
        "You are a retired teacher. You focus on education standards, public services, healthcare access, and government efficiency. "
        "You speak reflectively, comparing 'then vs now', and you care about the community and institutions."
    ),
    "tech worker": (
        "You are a tech worker. You care about job conditions, salary, taxes on paycheck, commuting/transport reliability, "
        "and the cost of living in the city. You use modern casual language, sometimes mentioning deadlines, burnout, and productivity."
    ),
    "nurse": (
        "You are a nurse. You care deeply about healthcare quality, hospital staffing, patient safety, and government funding decisions. "
        "You mention shifts, waiting rooms, shortages, and how policy impacts real people. Your tone is grounded and human."
    ),
}

PERSONAS = list(PERSONA_HINTS.keys())

PERSONA_PRIORS = {
    "struggling student": ["Cost of Living", "Education", "Employment"],
    "wealthy business owner": ["Taxation", "Government", "Employment"],
    "tired parent": ["Personal Security", "Education", "Cost of Living"],
    "political activist": ["Social Equality", "Environment", "Government"],
    "retired teacher": ["Education", "Healthcare", "Government"],
    "tech worker": ["Employment", "Transportation", "Cost of Living"],
    "nurse": ["Healthcare", "Government"],
}

# =========================
# 2.1) SENTIMENT_PRIORS
# =========================

SENTIMENT_PRIORS = {
    # struggling student: тяжело с деньгами, но иногда ок с образованием
    ("struggling student", "Cost of Living"): -0.7,
    ("struggling student", "Education"): +0.3,   # был в минусе, делаем чуть позитивнее
    ("struggling student", "Employment"): -0.3,

    # wealthy business owner: очень недоволен налогами, но доволен занятостью/рынком
    ("wealthy business owner", "Taxation"): -0.8,
    ("wealthy business owner", "Government"): -0.4,
    ("wealthy business owner", "Employment"): +0.6,   # сильный плюс

    # tired parent: дорого жить, но школу иногда хвалит
    ("tired parent", "Cost of Living"): -0.6,
    ("tired parent", "Education"): +0.2,             # небольшой плюс
    ("tired parent", "Personal Security"): -0.3,

    # political activist: в целом недоволен системой
    ("political activist", "Social Equality"): -0.8,
    ("political activist", "Environment"): -0.7,
    ("political activist", "Government"): -0.6,

    # retired teacher: любит образование, но критикует здоровье/гос-во
    ("retired teacher", "Education"): +0.4,          # делаем явно позитивным
    ("retired teacher", "Healthcare"): -0.3,
    ("retired teacher", "Government"): -0.2,

    # tech worker: доволен работой и чуть позитивен к среде
    ("tech worker", "Cost of Living"): -0.6,
    ("tech worker", "Employment"): +0.5,             # сильный плюс
    ("tech worker", "Transportation"): -0.3,
    ("tech worker", "Taxation"): -0.3,
    ("tech worker", "Environment"): +0.2,            # небольшой плюс

    # nurse: много негатива к системе, но есть гордость за профессию
    ("nurse", "Healthcare"): -0.3,                   # ослабляем чистый хейт
    ("nurse", "Government"): -0.5,
}


PRIOR_PROB = 0.75   # 75% случаев аспекты выбираются из priors персоны (даёт корреляцию)
GROUP_FOCUS_PROB = 0.8  # в ~60% случаев, когда k>=2, берём аспекты из одной группы

def choose_aspects_with_group_bias(persona: str, k: int):
    """
    Выбираем k аспектов так, чтобы часто несколько попадали
    в один и тот же ASPECT_GROUP (econ/services/values).
    Это даёт положительную корреляцию внутри групп.
    """
    pool = choose_aspect_pool(persona)
    if k <= 1 or len(pool) <= 1:
        return random.sample(pool, min(k, len(pool)))

    # Решаем: делаем групповой фокус или нет
    if random.random() >= GROUP_FOCUS_PROB:
        return random.sample(pool, min(k, len(pool)))

    # Ищем группы, которые пересекаются с pool
    candidate_groups = []
    for g, asp_list in ASPECT_GROUPS.items():
        if any(a in pool for a in asp_list):
            candidate_groups.append(g)

    if not candidate_groups:
        return random.sample(pool, min(k, len(pool)))

    # Выбираем одну группу и набираем из неё аспекты
    g = random.choice(candidate_groups)
    group_aspects = [a for a in ASPECT_GROUPS[g] if a in pool]

    chosen = set()
    if group_aspects:
        chosen.update(random.sample(group_aspects, min(len(group_aspects), k)))

    # если ещё не добрали k — добираем из всего пула
    while len(chosen) < min(k, len(pool)):
        chosen.add(random.choice(pool))

    return list(chosen)


def choose_aspect_pool(persona: str):
    if random.random() < PRIOR_PROB:
        return PERSONA_PRIORS.get(persona, ASPECTS)
    return ASPECTS

def sample_group_sentiments():
    group_sent = {}
    for g in ASPECT_GROUPS:
        # общий тон по группе: -1, 0, +1
        group_sent[g] = random.choices(
            [-1, 0, +1],
            weights=[0.4, 0.2, 0.4]
        )[0]
    return group_sent

def sample_sign(persona, aspect, base_vec_value, group_sent):
    if base_vec_value == 0:
        return 0

    mu_persona = SENTIMENT_PRIORS.get((persona, aspect), 0.0)

    # найдём группу
    g_mu = 0.0
    for g, asp_list in ASPECT_GROUPS.items():
        if aspect in asp_list:
            g_mu = 0.8 * group_sent[g]  # группа даёт +0.5/-0.5 если не 0
            break

    mu = mu_persona + g_mu
    mu = max(-1.0, min(1.0, mu))  # клип в [-1,1]

    p_pos = (mu + 1) / 2  # mu=-1 -> 0, mu=0 -> 0.5, mu=1 -> 1
    return +1 if random.random() < p_pos else -1

def make_target_vector_with_priors(persona: str):
    """
    Генерируем target вектор с приоритетом под персону.
    10% без аспектов, чаще 1–2, реже 3.
    """
    vec = {a: 0 for a in ASPECTS}

    if random.random() < 0.10:  # 10% no aspect
        return vec

    k = random.choices([1, 2, 3], weights=[0.55, 0.35, 0.10])[0]

    chosen = choose_aspects_with_group_bias(persona, k)

    # спец-правило для nurse
    if persona == "nurse" and "Government" in chosen and "Healthcare" not in chosen:
        chosen[0] = "Healthcare"

    group_sent = sample_group_sentiments()

    for a in chosen:
        vec[a] = sample_sign(persona, a, 1, group_sent)

    return vec

# =========================
# 3) BASE PROMPT
# =========================
BASE_PROMPT = r"""
You are generating a realistic social-media tweet.

TASK:
Write ONE tweet (maximum 280 characters).

INPUT INFORMATION:

1) PERSONA CONTEXT
This defines how the person thinks, what they care about, and the language they tend to use.
Persona bias should be subtle but noticeable.

{PERSONA_HINT}

2) TARGET ASPECT SENTIMENT VECTOR
Each aspect has a value:
-1 = negative sentiment
 0 = not mentioned or neutral
+1 = positive sentiment

You MUST reflect these sentiments in the tweet.

TARGET_VECTOR:
{ASPECT_VECTOR}

3) ASPECT ANCHORS
These are typical real-world situations related to each aspect.
You may use them explicitly OR implicitly.

ASPECT_ANCHORS:
{ASPECT_ANCHORS}

IMPORTANT RULES:
- The tweet must sound natural and realistic.
- If an aspect is non-zero, its sentiment MUST be expressed in the text.
- Some aspects should be expressed IMPLICITLY (without naming the aspect directly).
- Do NOT list aspects.
- Do NOT explain sentiments.
- Do NOT mention the word "aspect" or any labels.
- Do NOT add hashtags unless they feel completely natural.
- One short coherent tweet, not multiple sentences stitched together.

OUTPUT FORMAT:
Return ONLY valid JSON.
Do NOT add explanations.

{{
  "tweet_text": "..."
}}
""".strip()

def get_content(resp):
    try:
        return resp["message"]["content"].strip()
    except Exception:
        return ""

# =========================
# 4) NOISE CONFIG
# =========================
def apply_noise_config():
    r = random.random()
    implicit_mode = (r < 0.30)          # 30% implicit
    label_noise   = (0.30 <= r < 0.40)  # 10% label noise
    persona_noise = (0.40 <= r < 0.47)  # 7% persona noise
    return implicit_mode, label_noise, persona_noise

def maybe_swap_persona_hint(persona, persona_hints, persona_noise=False):
    if not persona_noise:
        return persona_hints[persona]
    others = [p for p in persona_hints.keys() if p != persona]
    return persona_hints[random.choice(others)]

def implicit_instruction(implicit_mode: bool) -> str:
    if implicit_mode:
        return "Extra constraint: Make at least one active aspect implicit (do NOT name it directly)."
    return "Extra constraint: You may mention aspects explicitly if it feels natural."

def inject_label_noise(vector: dict, max_changes=1, flip_prob=0.03):
    vec = copy.deepcopy(vector)
    active = [a for a, v in vec.items() if v != 0]
    if not active:
        return vec

    # иногда flip
    if random.random() < flip_prob:
        a = random.choice(active)
        vec[a] = -vec[a]
        return vec

    # иначе выключаем один активный аспект
    for _ in range(max_changes):
        a = random.choice(active)
        vec[a] = 0
        active.remove(a)
        if not active:
            break

    return vec

# =========================
# 6) PROMPT: anchors только нужные
# =========================
def pick_anchors_for_prompt(final_vector, distractors=2):
    active = [a for a, v in final_vector.items() if v != 0]
    chosen = {}

    for a in active:
        chosen[a] = random.sample(ASPECT_ANCHORS[a], k=min(3, len(ASPECT_ANCHORS[a])))

    others = [a for a in ASPECTS if a not in active]
    random.shuffle(others)
    for a in others[:distractors]:
        chosen[a] = random.sample(ASPECT_ANCHORS[a], k=2)

    return chosen

def build_prompt(persona, aspect_vector):
    implicit_mode, label_noise, persona_noise = apply_noise_config()

    final_vector = inject_label_noise(aspect_vector) if label_noise else aspect_vector
    persona_hint = maybe_swap_persona_hint(persona, PERSONA_HINTS, persona_noise)

    anchors_for_prompt = pick_anchors_for_prompt(final_vector, distractors=2)

    prompt = BASE_PROMPT.format(
        PERSONA_HINT=persona_hint,
        ASPECT_VECTOR=json.dumps(final_vector, ensure_ascii=False),
        ASPECT_ANCHORS=json.dumps(anchors_for_prompt, ensure_ascii=False)
    )
    prompt = prompt + "\n\n" + implicit_instruction(implicit_mode)

    noise_meta = {"implicit": implicit_mode, "label_noise": label_noise, "persona_noise": persona_noise}
    return prompt, final_vector, noise_meta, anchors_for_prompt

# =========================
# 7) VALIDATION: мягкий чек на смысл
# =========================
def passes_min_checks(tweet, final_vec, anchors_for_prompt):
    active = [a for a, v in final_vec.items() if v != 0]
    if active and len(tweet) < 40:
        return False

    if active and random.random() < 0.60:
        low = tweet.lower()
        active_words = []
        for a in active:
            for w in anchors_for_prompt.get(a, []):
                active_words.append(w.lower())
        if active_words and not any(w in low for w in active_words):
            return False

    return True

def generate_one(model="llama3.2", max_retries=8):
    persona = random.choice(PERSONAS)
    target_vec = make_target_vector_with_priors(persona)

    prompt, final_vec, noise_meta, anchors_for_prompt = build_prompt(persona, target_vec)

    for attempt in range(1, max_retries + 1):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                keep_alive=-1,
                format="json"
            )
            txt = get_content(resp)
            if not txt:
                time.sleep(0.08 * attempt)
                continue

            obj = json.loads(txt)
            tweet = str(obj.get("tweet_text", "")).replace("\n", " ").strip()
            if not tweet or len(tweet) > 280:
                time.sleep(0.08 * attempt)
                continue

            if not passes_min_checks(tweet, final_vec, anchors_for_prompt):
                time.sleep(0.05)
                continue

            return {
                "tweet_text": tweet,
                "persona": persona,
                "labels": final_vec,
                "noise": noise_meta
            }

        except Exception:
            time.sleep(0.08 * attempt)

    return None

# =========================
# 8) MAIN
# =========================
OUT_PATH = "synthetic_dataset.jsonl"
TOTAL = 1000
MODEL = "llama3.2"
MAX_RETRIES = 8
FLUSH_EVERY = 50

def main():
    ollama.chat(model=MODEL, messages=[{"role": "user", "content": "hi"}], keep_alive=-1)

    saved = 0
    failed = 0
    buffer = []

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for _ in tqdm(range(TOTAL)):
            row = generate_one(model=MODEL, max_retries=MAX_RETRIES)

            if row is None:
                failed += 1
                continue

            buffer.append(json.dumps(row, ensure_ascii=False))
            saved += 1

            if len(buffer) >= FLUSH_EVERY:
                f.write("\n".join(buffer) + "\n")
                f.flush()
                buffer.clear()

        if buffer:
            f.write("\n".join(buffer) + "\n")
            f.flush()
            buffer.clear()

    print(f"✅ Saved {saved} rows to {OUT_PATH}")
    print(f"⚠️ Failed generations: {failed}")

if __name__ == "__main__":
    main()
