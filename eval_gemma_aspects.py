import json
import argparse
from typing import List, Dict, Any
import csv

import numpy as np
from tqdm import tqdm
import ollama
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

# ---------- 1. Определяем аспекты ----------

ASPECTS = [
    "Cost of Living", "Healthcare", "Education", "Personal Security",
    "Employment", "Transportation", "Government", "Environment",
    "Social Equality", "Taxation"
]


# ---------- 2. Загрузка датасета (JSONL) ----------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)
    return data


# ---------- 3. Парсер предсказанных значений (-1/0/+1) ----------

def parse_label(val) -> int:
    """
    Аккуратно приводит значение из LLM к -1/0/+1.
    Любая грязь/странные форматы -> 0.
    """
    if isinstance(val, (int, float)):
        v = int(round(val))
    elif isinstance(val, str):
        s = val.strip()
        if s.endswith(","):
            s = s[:-1]
        s = s.split()[0]
        s = s.replace('"', '').replace("'", "")

        if s in {"1", "+1", "1.0", "+1.0"}:
            v = 1
        elif s in {"-1", "-1.0"}:
            v = -1
        elif s in {"0", "+0", "-0", "0.0"}:
            v = 0
        else:
            return 0
    else:
        return 0

    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


# ---------- 4. Промпт для Gemma ----------

def build_prompt(tweet_text: str) -> str:
    """
   we provide a tweet and a list of aspects.
   We ask [the model] to return a JSON with -1/0/+1 for each aspect.
    """
    aspects_str = "\n".join(f"- {a}" for a in ASPECTS)
    prompt = f"""
You are an assistant for aspect-based sentiment analysis.

Given a social-media post, you must assign a sentiment score (-1, 0, +1)
for each of the following aspects:

{aspects_str}

Meaning of scores:
-1 = clearly negative sentiment toward this aspect
 0 = not mentioned or neutral
+1 = clearly positive sentiment toward this aspect

Post:
\"\"\"{tweet_text}\"\"\"

Return ONLY valid JSON with keys exactly the aspect names and integer values -1, 0, or +1.
Example format:
{{
  "Cost of Living": -1,
  "Healthcare": 0,
  ...
}}
"""
    return prompt.strip()


def call_gemma(model_name: str, prompt: str, max_retries: int = 5) -> Dict[str, Any]:
    """
    Вызов Gemma через Ollama. Ожидаем JSON-ответ.
    Если не получилось, возвращаем пустой словарь.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                keep_alive=-1
            )
            content = resp["message"]["content"].strip()
            obj = json.loads(content)
            if isinstance(obj, dict):
                return obj
        except Exception:
            # можно добавить print, если хочется дебажить
            pass
    return {}


# ---------- 5. Предсказания по всему датасету ----------

def predict_on_dataset(data: List[Dict[str, Any]], model_name: str):
    """
    Возвращает:
      y_true_detect: [N, A] (0/1)
      y_true_sign:   [N, A] (-1/0/+1)
      y_pred_detect: [N, A]
      y_pred_sign:   [N, A]
    """
    N = len(data)
    A = len(ASPECTS)

    y_true_detect = np.zeros((N, A), dtype=int)
    y_true_sign = np.zeros((N, A), dtype=int)
    y_pred_detect = np.zeros((N, A), dtype=int)
    y_pred_sign = np.zeros((N, A), dtype=int)

    for i, row in enumerate(tqdm(data, desc="Predicting with Gemma")):
        text = row["tweet_text"]
        labels = row["labels"]  # dict: aspect -> -1/0/+1

        # y_true
        for j, asp in enumerate(ASPECTS):
            v = int(labels.get(asp, 0))
            y_true_sign[i, j] = v
            if v != 0:
                y_true_detect[i, j] = 1

        # запрос к Gemma
        prompt = build_prompt(text)
        pred_dict = call_gemma(model_name, prompt)

        # y_pred
        for j, asp in enumerate(ASPECTS):
            raw_v = pred_dict.get(asp, 0)
            v = parse_label(raw_v)
            if v != 0:
                y_pred_detect[i, j] = 1
                y_pred_sign[i, j] = v
            else:
                y_pred_detect[i, j] = 0
                y_pred_sign[i, j] = 0

    return y_true_detect, y_true_sign, y_pred_detect, y_pred_sign


# ---------- 6. Подсчёт metrics + сохранение в CSV ----------

def compute_metrics_and_save(
    y_true_detect, y_true_sign, y_pred_detect, y_pred_sign,
    csv_path: str = "gemma_metrics.csv"
):
    rows = []

    # по аспектам
    for j, asp in enumerate(ASPECTS):
        ytd = y_true_detect[:, j]
        ypd = y_pred_detect[:, j]
        yts = y_true_sign[:, j]
        yps = y_pred_sign[:, j]

        # DETECT: бинарные метрики
        prec_det = precision_score(ytd, ypd, zero_division=0)
        rec_det = recall_score(ytd, ypd, zero_division=0)
        f1_det = f1_score(ytd, ypd, zero_division=0)
        acc_det = accuracy_score(ytd, ypd)

        # SIGN: многоклассные метрики (macro по -1/0/+1)
        prec_sign = precision_score(
            yts, yps,
            labels=[-1, 0, 1],
            average="macro",
            zero_division=0
        )
        rec_sign = recall_score(
            yts, yps,
            labels=[-1, 0, 1],
            average="macro",
            zero_division=0
        )
        f1_sign = f1_score(
            yts, yps,
            labels=[-1, 0, 1],
            average="macro",
            zero_division=0
        )
        acc_sign = accuracy_score(yts, yps)

        rows.append({
            "aspect": asp,
            "det_precision": prec_det,
            "det_recall": rec_det,
            "det_f1": f1_det,
            "det_accuracy": acc_det,
            "sign_precision_macro": prec_sign,
            "sign_recall_macro": rec_sign,
            "sign_f1_macro": f1_sign,
            "sign_accuracy": acc_sign,
        })

    # macro по аспектам
    macro_det_f1 = float(np.mean([r["det_f1"] for r in rows]))
    macro_sign_f1 = float(np.mean([r["sign_f1_macro"] for r in rows]))
    macro_det_prec = float(np.mean([r["det_precision"] for r in rows]))
    macro_det_rec = float(np.mean([r["det_recall"] for r in rows]))
    macro_sign_prec = float(np.mean([r["sign_precision_macro"] for r in rows]))
    macro_sign_rec = float(np.mean([r["sign_recall_macro"] for r in rows]))

    print("\n=== Macro over aspects (DETECTION) ===")
    print(f"Precision: {macro_det_prec:.4f}")
    print(f"Recall:    {macro_det_rec:.4f}")
    print(f"F1:        {macro_det_f1:.4f}")

    print("\n=== Macro over aspects (SIGN -1/0/+1, macro) ===")
    print(f"Precision: {macro_sign_prec:.4f}")
    print(f"Recall:    {macro_sign_rec:.4f}")
    print(f"F1:        {macro_sign_f1:.4f}")

    # сохраняем в CSV
    fieldnames = [
        "aspect",
        "det_precision", "det_recall", "det_f1", "det_accuracy",
        "sign_precision_macro", "sign_recall_macro", "sign_f1_macro", "sign_accuracy",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nSaved per-aspect metrics to {csv_path}")

    return rows, {
        "macro_det_precision": macro_det_prec,
        "macro_det_recall": macro_det_rec,
        "macro_det_f1": macro_det_f1,
        "macro_sign_precision": macro_sign_prec,
        "macro_sign_recall": macro_sign_rec,
        "macro_sign_f1": macro_sign_f1,
    }


# ---------- 7. main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="synthetic_10000.jsonl")
    parser.add_argument("--model", type=str, default="gemma3")
    parser.add_argument("--max_examples", type=int, default=2000)
    parser.add_argument("--out_csv", type=str, default="gemma_metrics.csv")
    args = parser.parse_args()

    data = load_jsonl(args.data)
    if args.max_examples is not None:
        data = data[:args.max_examples]
    print(f"Loaded {len(data)} examples from {args.data} (after slicing)")

    y_true_detect, y_true_sign, y_pred_detect, y_pred_sign = predict_on_dataset(
        data, args.model
    )

    compute_metrics_and_save(
        y_true_detect, y_true_sign, y_pred_detect, y_pred_sign,
        csv_path=args.out_csv
    )


if __name__ == "__main__":
    main()
