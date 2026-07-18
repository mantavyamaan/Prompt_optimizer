import asyncio
import json
import uuid
import random
import re
from typing import Any
from .db import conn, dumps
from .models import GenParams
from .serve import build_backend

def get_failed_real_queries(category: str, limit: int = 5) -> list[str]:
    with conn() as connection:
        rows = connection.execute("""
            SELECT t.query, f.signal 
            FROM traces t
            JOIN feedback f ON t.trace_id = f.trace_id
            WHERE t.category = ? AND f.signal LIKE 'score:%'
            ORDER BY RANDOM() LIMIT 20
        """, (category,)).fetchall()
        
    failed = []
    for row in rows:
        try:
            score = int(row["signal"].split(":")[1])
            if score < 50:
                failed.append(row["query"])
        except Exception:
            pass
    return failed[:limit]

async def _run_generation(category: str, count: int) -> None:
    backend = build_backend()
    
    failed_queries = get_failed_real_queries(category)
    failure_context = ""
    if failed_queries:
        failure_context = "\n\nCRITICAL: Here are some actual real-world queries where users were highly dissatisfied (score < 50). Use these exact failures to heavily inspire your adversarial edge cases:\n"
        for q in failed_queries:
            failure_context += f"- {json.dumps(q)}\n"

    prompt = f"""You are an expert dataset generator. Generate {count} diverse, extremely challenging test cases for the category '{category}'.{failure_context}
Return ONLY a valid JSON array of objects. Each object must have:
- "input": a dictionary representing the input data (e.g., text containing information to extract, including adversarial noise and edge cases).
- "expected": a dictionary representing the exact expected output JSON structure the model should extract.
- "difficulty": either "routine", "hard", or "adversarial". Make most of them "hard" or "adversarial".

Ensure variety. Do not include markdown backticks. Return ONLY the raw JSON array.
"""
    
    print(f"Asking LLM to generate {count} synthetic cases for '{category}'...")
    
    try:
        output = await backend.generate(prompt, {}, None, GenParams(temperature=0.8, max_tokens=4000))
        
        # Clean the output if it contains markdown formatting
        if output.strip().startswith("```"):
            output = re.sub(r"^```(?:json)?\s*|\s*```$", "", output.strip(), flags=re.I)
            
        cases = json.loads(output)
        
        if not isinstance(cases, list):
            raise ValueError("LLM did not return a JSON array.")
            
        with conn() as connection:
            for case in cases:
                difficulty = case.get("difficulty", "hard")
                if difficulty not in ("routine", "hard", "adversarial"):
                    difficulty = "hard"
                
                # Split roughly 60% train, 20% holdout, 20% vault
                r = random.random()
                if r < 0.6:
                    split = "train"
                elif r < 0.8:
                    split = "holdout"
                else:
                    split = "vault"
                    
                case_id = f"synth-{uuid.uuid4().hex[:8]}"
                
                # We expect the evaluator for extraction to look at expected["json"]
                expected_dict = {"json": case.get("expected", {})}
                
                connection.execute(
                    "INSERT INTO benchmark_cases(case_id,category,input,expected,difficulty,split,source) VALUES(?,?,?,?,?,?,?)",
                    (case_id, category, dumps(case.get("input", {})), dumps(expected_dict), difficulty, split, "synthetic")
                )
        print(f"Successfully generated and saved {len(cases)} synthetic cases to the database!")
    except Exception as e:
        print(f"Failed to generate synthetic data: {e}")

def generate_synthetic_cases(category: str, count: int = 20) -> None:
    asyncio.run(_run_generation(category, count))
