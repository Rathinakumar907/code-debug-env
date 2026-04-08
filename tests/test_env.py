"""
tests/test_env.py — pytest suite for CodeDebugEnv
Run with: pytest tests/ -v
"""

import pytest
from server.env import CodeDebugEnv, DebugAction, _syntax_ok, _compute_reward

@pytest.fixture
def env():
    return CodeDebugEnv()

def fresh_episode(env, bug_id="off_by_one"):
    return env.reset(session_id="test-session", bug_id=bug_id)

# ── reset() ────────────────────────────────────────────────────────────────────

def test_reset_returns_observation(env):
    obs = fresh_episode(env)
    assert obs.broken_code
    assert obs.bug_description
    assert len(obs.test_cases) > 0
    assert obs.attempts_remaining == 3

def test_reset_different_bug(env):
    obs = fresh_episode(env, bug_id="wrong_operator")
    assert "even" in obs.bug_description.lower()

# ── correct fixes ──────────────────────────────────────────────────────────────

CORRECT_FIXES = {
    "off_by_one": """\
def sum_list(nums):
    total = 0
    for i in range(len(nums)):
        total += nums[i]
    return total
""",
    "wrong_operator": """\
def is_even(n):
    return n % 2 == 0
""",
    "missing_base_case": """\
def factorial(n):
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
""",
    "wrong_return": """\
def reverse_string(s):
    reversed_s = s[::-1]
    return reversed_s
""",
    "index_error": """\
def last_element(lst):
    return lst[-1]
""",
}

@pytest.mark.parametrize("bug_id,fix", CORRECT_FIXES.items())
def test_correct_fix_passes_all_tests(env, bug_id, fix):
    env.reset(session_id="s1", bug_id=bug_id)
    result = env.step(DebugAction(fixed_code=fix), session_id="s1")
    assert result["info"]["passed"] == result["info"]["total"]
    assert result["reward"] > 0.9
    assert result["done"] is True
    assert result["info"]["solved"] is True

# ── broken code ────────────────────────────────────────────────────────────────

def test_syntax_error_penalises(env):
    fresh_episode(env)
    result = env.step(
        DebugAction(fixed_code="def sum_list(nums):\n  return ???"),
        session_id="test-session",
    )
    assert result["reward"] == -0.2
    assert result["info"]["syntax_ok"] is False

def test_wrong_fix_partial_reward(env):
    fresh_episode(env)
    result = env.step(
        DebugAction(fixed_code="def sum_list(nums):\n    return 0"),
        session_id="test-session",
    )
    assert result["reward"] > -0.2
    assert result["info"]["solved"] is False

# ── hint mechanic ──────────────────────────────────────────────────────────────

def test_hint_populated_when_requested(env):
    fresh_episode(env)
    result = env.step(
        DebugAction(fixed_code="def sum_list(nums):\n    return 0", use_hint=True),
        session_id="test-session",
    )
    assert result["observation"].hint is not None

def test_hint_costs_reward(env):
    env.reset(session_id="no-hint",   bug_id="off_by_one")
    env.reset(session_id="with-hint", bug_id="off_by_one")
    fix = CORRECT_FIXES["off_by_one"]
    r1 = env.step(DebugAction(fixed_code=fix, use_hint=False), session_id="no-hint")
    r2 = env.step(DebugAction(fixed_code=fix, use_hint=True),  session_id="with-hint")
    assert r1["reward"] > r2["reward"]

# ── attempts / done ────────────────────────────────────────────────────────────

def test_episode_ends_after_max_attempts(env):
    fresh_episode(env)
    bad_fix = "def sum_list(nums):\n    return -1"
    for _ in range(3):
        result = env.step(DebugAction(fixed_code=bad_fix), session_id="test-session")
    assert result["done"] is True
    assert result["observation"].attempts_remaining == 0

def test_no_step_after_solved(env):
    fresh_episode(env)
    fix = CORRECT_FIXES["off_by_one"]
    env.step(DebugAction(fixed_code=fix), session_id="test-session")
    result = env.step(DebugAction(fixed_code=fix), session_id="test-session")
    assert result["reward"] == 0.0
    assert result["done"] is True

# ── state() ───────────────────────────────────────────────────────────────────

def test_state_reflects_progress(env):
    fresh_episode(env)
    fix = CORRECT_FIXES["off_by_one"]
    env.step(DebugAction(fixed_code=fix), session_id="test-session")
    state = env.state(session_id="test-session")
    assert state.solved is True
    assert state.total_reward > 0

# ── unit helpers ───────────────────────────────────────────────────────────────

def test_syntax_ok_valid():
    assert _syntax_ok("def f(x):\n    return x + 1") is True

def test_syntax_ok_invalid():
    assert _syntax_ok("def f(x)\n    return x") is False

def test_compute_reward_perfect():
    r = _compute_reward(passed=4, total=4, syntax_ok=True, hints_used=0, runs_clean=True)
    assert r == 1.0

def test_compute_reward_syntax_error():
    r = _compute_reward(passed=0, total=4, syntax_ok=False, hints_used=0, runs_clean=False)
    assert r == -0.2

def test_compute_reward_partial():
    r = _compute_reward(passed=2, total=4, syntax_ok=True, hints_used=1, runs_clean=True)
    assert r == pytest.approx(0.5)