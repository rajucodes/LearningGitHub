import os
import json
import math
import random
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

_OPENAI_AVAILABLE = False
try:
    from openai import OpenAI

    _OPENAI_AVAILABLE = True
except Exception:
    _OPENAI_AVAILABLE = False


class QuestionService:
    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.use_openai = _OPENAI_AVAILABLE and bool(self.openai_api_key)
        self.client = OpenAI(api_key=self.openai_api_key) if self.use_openai else None

    def generate_question_paper(
        self,
        prompt_guidelines: str,
        context_snippets: List[str],
        num_questions: int = 10,
    ) -> Dict[str, Any]:
        if self.use_openai:
            system = (
                "You are a strict exam setter. Generate a high-quality question paper as JSON with fields: "
                "title, instructions, questions[]. Each question has: id, type (mcq|short), question, options (for mcq), answer_key, marks."
            )
            user = (
                f"Guidelines: {prompt_guidelines}\n\n"
                f"Context (use to craft questions):\n\n" + "\n\n".join(context_snippets) + "\n\n"
                f"Generate exactly {num_questions} questions, ~60% mcq and ~40% short."
            )
            resp = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0.7,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            content = resp.choices[0].message.content
            try:
                return json.loads(content)
            except Exception:
                pass
        # Fallback heuristic
        title = "Generated Question Paper"
        instructions = "Answer all questions."
        questions: List[Dict[str, Any]] = []
        seed = abs(hash(prompt_guidelines)) % (10**8)
        rng = random.Random(seed)
        material = " ".join(context_snippets)
        sentences = [s.strip() for s in material.split(".") if len(s.strip()) > 20]
        pick = sentences[: max(5, min(len(sentences), num_questions * 2))]
        for i in range(num_questions):
            if i % 2 == 0 and len(pick) >= 1:
                base = pick[i % len(pick)]
                q = base.split(" ")
                if len(q) > 6:
                    blank_idx = rng.randint(2, min(len(q) - 2, 8))
                    answer = q[blank_idx]
                    options = [answer]
                    # Create distractors by small edits
                    for _ in range(3):
                        if len(answer) > 5:
                            pos = rng.randint(1, len(answer) - 2)
                            distractor = answer[:pos] + answer[pos + 1 :]
                        else:
                            distractor = answer + rng.choice(["ly", "ed", "s", "ing"])[: rng.randint(0, 3)]
                        options.append(distractor)
                    rng.shuffle(options)
                    questions.append(
                        {
                            "id": f"q{i+1}",
                            "type": "mcq",
                            "question": "Fill in the blank: " + " ".join(q[:blank_idx] + ["_____"] + q[blank_idx + 1 :]) + ".",
                            "options": options,
                            "answer_key": answer,
                            "marks": 1,
                        }
                    )
                    continue
            # short question
            src = pick[i % len(pick)] if pick else prompt_guidelines
            questions.append(
                {
                    "id": f"q{i+1}",
                    "type": "short",
                    "question": f"Briefly explain: {src[:80]}...",
                    "answer_key": src,
                    "marks": 2,
                }
            )
        return {"title": title, "instructions": instructions, "questions": questions}

    def grade_answers(self, paper: Dict[str, Any], answers: Dict[str, str]) -> Tuple[int, int, List[Dict[str, Any]]]:
        total_marks = sum(int(q.get("marks", 1)) for q in paper.get("questions", []))
        obtained = 0
        breakdown: List[Dict[str, Any]] = []
        for q in paper.get("questions", []):
            qid = q["id"]
            qtype = q.get("type", "short")
            ans = (answers.get(qid) or "").strip()
            key = (q.get("answer_key") or "").strip()
            marks = int(q.get("marks", 1))
            score = 0
            feedback = ""
            if qtype == "mcq":
                score = marks if ans.lower() == key.lower() else 0
                feedback = "Correct" if score else f"Expected: {key}"
            else:
                # semantic similarity lite
                ratio = SequenceMatcher(None, ans.lower(), key.lower()).ratio()
                score = int(round(marks * ratio))
                feedback = f"Similarity {ratio:.2f}"
            obtained += score
            breakdown.append({"id": qid, "score": score, "marks": marks, "feedback": feedback})
        return obtained, total_marks, breakdown