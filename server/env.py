"""
Code Debugging RL Environment — Meta PyTorch OpenEnv Hackathon
"""

import ast
import subprocess
import sys
import textwrap
import uuid
from typing import Optional
from pydantic import BaseModel, Field

PYTHON_EXE = r"C:\Users\rathi\AppData\Local\Programs\Python\Python314\python.exe"

BUG_DATASET = [
    {
        "id": "off_by_one",
        "description": "A function that should return the sum of a list has an off-by-one error.",
        "broken_code": textwrap.dedent("""\
            def sum_list(nums):
                total = 0
                for i in range(len(nums) - 1):
                    total += nums[i]
                return total
        """),
        "hint": "Check the range() bounds — are you iterating over every element?",
        "tests": [
            {"input": "[1, 2, 3, 4, 5]", "expected": "15"},
            {"input": "[10, 20]",         "expected": "30"},
            {"input": "[0]",              "expected": "0"},
            {"input": "[]",               "expected": "0"},
        ],
        "test_fn": "sum_list",
    },
    {
        "id": "wrong_operator",
        "description": "A function to check if a number is even uses the wrong operator.",
        "broken_code": textwrap.dedent("""\
            def is_even(n):
                return n / 2 == 0
        """),
        "hint": "Which operator gives you the remainder of a division?",
        "tests": [
            {"input": "4",  "expected": "True"},
            {"input": "7",  "expected": "False"},
            {"input": "0",  "expected": "True"},
            {"input": "-2", "expected": "True"},
        ],
        "test_fn": "is_even",
    },
    {
        "id": "missing_base_case",
        "description": "A recursive factorial function is missing its base case.",
        "broken_code": textwrap.dedent("""\
            def factorial(n):
                return n * factorial(n - 1)
        """),
        "hint": "All recursion needs a stopping condition. What is factorial(0)?",
        "tests": [
            {"input": "0", "expected": "1"},
            {"input": "1", "expected": "1"},
            {"input": "5", "expected": "120"},
            {"input": "3", "expected": "6"},
        ],
        "test_fn": "factorial",
    },
    {
        "id": "wrong_return",
        "description": "A function that reverses a string modifies it but returns the original.",
        "broken_code": textwrap.dedent("""\
            def reverse_string(s):
                reversed_s = s[::-1]
                return s
        """),
        "hint": "The reversed value is computed correctly — but what is actually returned?",
        "tests": [
            {"input": '"hello"', "expected": "olleh"},
            {"input": '"abcd"',  "expected": "dcba"},
            {"input": '""',      "expected": ""},
            {"input": '"a"',     "expected": "a"},
        ],
        "test_fn": "reverse_string",
    },
    {
        "id": "index_error",
        "description": "A function to get the last element uses an out-of-bounds index.",
        "broken_code": textwrap.dedent("""\
            def last_element(lst):
                return lst[len(lst)]
        """),
        "hint": "Python lists are 0-indexed. What is the index of the last element?",
        "tests": [
            {"input": "[1, 2, 3]", "expected": "3"},
            {"input": "[42]",      "expected": "42"},
            {"input": "[5, 10]",   "expected": "10"},
        ],
        "test_fn": "last_element",
    },
]


class DebugAction(BaseModel):
    fixed_code: str = Field(..., description="The agent's corrected Python code.")
    use_hint: bool  = Field(False, description="Set True to consume a hint.")


class DebugObservation(BaseModel):
    broken_code: str
    bug_description: str
    test_cases: list[dict]
    hint: Optional[str] = None
    attempts_remaining: int
    last_result: Optional[str] = None


class DebugState(BaseModel):
    session_id: str
    bug_id: str
    broken_code: str
    bug_description: str
    test_cases: list[dict]
    hint_text: str
    test_fn: str
    wrapper_code: Optional[str]
    attempts_remaining: int
    hints_used: int
    solved: bool
    total_reward: float


def _run_tests(fixed_code: str, bug: dict) -> tuple[int, int, str]:
    test_fn = bug["test_fn"]
    tests   = bug["tests"]
    passed  = 0
    errors  = []

    for tc in tests:
        inp      = tc["input"]
        expected = tc["expected"]

        script = fixed_code + f"\nresult = {test_fn}({inp})\nprint(str(result))\n"

        try:
            proc = subprocess.run(
                [PYTHON_EXE, "-c", script],
                capture_output=True, text=True, timeout=5
            )
            output = proc.stdout.strip()
            if output == expected:
                passed += 1
            else:
                errors.append(f"  input={inp} → got {output!r}, expected {expected!r}")
        except subprocess.TimeoutExpired:
            errors.append(f"  input={inp} → TIMEOUT")
        except Exception as e:
            errors.append(f"  input={inp} → runner error: {e}")

    return passed, len(tests), "\n".join(errors)

def _syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _compute_reward(passed: int, total: int, syntax_ok: bool,
                    hints_used: int, runs_clean: bool) -> float:
    if not syntax_ok:
        return -0.2
    reward = 0.0
    if runs_clean:
        reward += 0.3
    reward += 0.5 * (passed / total) if total else 0
    if passed == total and total > 0:
        reward += 0.2
    reward -= 0.05 * hints_used
    return round(reward, 4)


class CodeDebugEnv:
    MAX_ATTEMPTS = 3

    def __init__(self):
        self._sessions: dict[str, DebugState] = {}

    def reset(self, session_id: Optional[str] = None,
              bug_id: Optional[str] = None) -> DebugObservation:
        import random
        sid = session_id or str(uuid.uuid4())
        bug = (
            next((b for b in BUG_DATASET if b["id"] == bug_id), None)
            or random.choice(BUG_DATASET)
        )
        state = DebugState(
            session_id=sid,
            bug_id=bug["id"],
            broken_code=bug["broken_code"],
            bug_description=bug["description"],
            test_cases=bug["tests"],
            hint_text=bug["hint"],
            test_fn=bug["test_fn"],
            wrapper_code=bug.get("wrapper", ""),
            attempts_remaining=self.MAX_ATTEMPTS,
            hints_used=0,
            solved=False,
            total_reward=0.0,
        )
        self._sessions[sid] = state
        return DebugObservation(
            broken_code=state.broken_code,
            bug_description=state.bug_description,
            test_cases=state.test_cases,
            attempts_remaining=state.attempts_remaining,
        )

    def step(self, action: DebugAction,
             session_id: Optional[str] = None) -> dict:
        sid   = session_id or list(self._sessions)[-1]
        state = self._sessions[sid]

        if state.solved or state.attempts_remaining <= 0:
            return {
                "observation": DebugObservation(
                    broken_code=state.broken_code,
                    bug_description=state.bug_description,
                    test_cases=state.test_cases,
                    attempts_remaining=0,
                    last_result="Episode already finished.",
                ),
                "reward": 0.0,
                "done": True,
                "info": {"reason": "episode_over"},
            }

        hint_text = None
        if action.use_hint:
            state.hints_used += 1
            hint_text = state.hint_text

        state.attempts_remaining -= 1

        syntax_ok  = _syntax_ok(action.fixed_code)
        runs_clean = False
        passed, total, error_summary = 0, len(state.test_cases), ""

        if syntax_ok:
            try:
                passed, total, error_summary = _run_tests(
                    action.fixed_code,
                    {"test_fn": state.test_fn, "tests": state.test_cases},
                )
                runs_clean = True
            except Exception as e:
                error_summary = f"Runtime error: {e}"

        reward = _compute_reward(passed, total, syntax_ok, state.hints_used, runs_clean)
        state.total_reward += reward
        solved = (passed == total and total > 0)
        state.solved = solved
        done = solved or state.attempts_remaining <= 0

        result_msg = (
            f"✅ All {total} tests passed! Bug fixed." if solved
            else f"❌ {passed}/{total} tests passed.\n{error_summary}"
        )
        if not syntax_ok:
            result_msg = "❌ Syntax error in submitted code."

        return {
            "observation": DebugObservation(
                broken_code=state.broken_code,
                bug_description=state.bug_description,
                test_cases=state.test_cases,
                hint=hint_text,
                attempts_remaining=state.attempts_remaining,
                last_result=result_msg,
            ),
            "reward": reward,
            "done": done,
            "info": {
                "passed":       passed,
                "total":        total,
                "syntax_ok":    syntax_ok,
                "hints_used":   state.hints_used,
                "solved":       solved,
                "total_reward": state.total_reward,
            },
        }

    def state(self, session_id: Optional[str] = None) -> DebugState:
        sid = session_id or list(self._sessions)[-1]
        return self._sessions[sid]