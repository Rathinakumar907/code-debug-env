"""
inference.py — LLM-powered agent for the Code Debugging RL Environment
Required by the hackathon: must be named inference.py and placed in root.
"""

import os
import asyncio
import textwrap
from openai import AsyncOpenAI
from openenv import GenericEnvClient, GenericAction

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
HF_TOKEN     = os.environ["HF_TOKEN"]
ENV_URL      = os.environ.get("ENV_URL", "http://localhost:7860")

client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Python debugger. You will be given:
    1. A description of the bug
    2. The broken Python code
    3. The test cases the code must pass

    Your task: output ONLY the corrected Python code, with no explanation,
    no markdown fences, no comments beyond what was already in the original.
    Just raw, valid Python.
""")


async def ask_llm(bug_description: str, broken_code: str,
                  test_cases: list, hint: str | None = None) -> str:
    tests_fmt = "\n".join(
        f"  {tc['input']} → {tc['expected']}" for tc in test_cases
    )
    hint_line = f"\nHint: {hint}" if hint else ""
    user_msg = textwrap.dedent(f"""\
        Bug description: {bug_description}{hint_line}

        Broken code:
        {broken_code}

        Test cases (input → expected output):
        {tests_fmt}

        Output ONLY the fixed Python code.
    """)

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


async def run_episode(bug_id: str | None = None):
    async with GenericEnvClient(base_url=ENV_URL) as env:
        reset_payload = {"bug_id": bug_id} if bug_id else {}
        obs = await env.reset(**reset_payload)
        print(f"\n{'='*60}")
        print(f"Bug: {obs['bug_description']}")
        print(f"Broken code:\n{obs['broken_code']}")

        hint = None
        total_reward = 0.0

        for attempt in range(3):
            print(f"\n── Attempt {attempt + 1} ──────────────────────────────")
            fixed_code = await ask_llm(
                bug_description=obs["bug_description"],
                broken_code=obs["broken_code"],
                test_cases=obs["test_cases"],
                hint=hint,
            )
            print(f"LLM fix:\n{fixed_code}")

            use_hint = (attempt == 1)
            result = await env.step(GenericAction(
                fixed_code=fixed_code,
                use_hint=use_hint,
            ))

            reward       = result["reward"]
            total_reward = result["info"].get("total_reward", total_reward + reward)
            done         = result["done"]
            hint         = result["observation"].get("hint")
            last_result  = result["observation"].get("last_result", "")

            print(f"Result : {last_result}")
            print(f"Reward : {reward:+.4f}  (cumulative: {total_reward:+.4f})")

            if done:
                break

        print(f"\n{'='*60}")
        print(f"Episode finished. Total reward: {total_reward:+.4f}")
        return total_reward


if __name__ == "__main__":
    asyncio.run(run_episode())