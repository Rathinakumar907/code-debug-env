---
title: Code Debug Env
emoji: 🐛
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# 🐛 Code Debugging RL Environment
### Meta PyTorch OpenEnv Hackathon — Round 1 Submission

An OpenEnv environment where an RL agent receives **broken Python code** and must produce a working fix that passes automated test cases.

---

## 🎯 Task Definition

| | |
|---|---|
| **Agent receives** | Broken Python code + bug description + test cases |
| **Agent outputs** | Fixed Python code (optionally requests a hint) |
| **Success criterion** | Fixed code passes all test cases |
| **Max attempts** | 3 per episode |

---

## 💰 Reward Structure

| Condition | Reward |
|---|---|
| Syntax error in submission | **-0.2** |
| Code runs without crashing | **+0.3** |
| Each test case passed | **+0.5 × (passed/total)** |
| ALL test cases passed (perfect) | **+0.2 bonus** |
| Per hint consumed | **-0.05** |
| **Maximum possible reward** | **+1.0** |

---

## 🗂️ Bug Dataset

| ID | Bug Type |
|---|---|
| `off_by_one` | Off-by-one in `range()` |
| `wrong_operator` | `/` instead of `%` |
| `missing_base_case` | Recursion with no base case |
| `wrong_return` | Returns wrong variable |
| `index_error` | Out-of-bounds index |

---

## 🔌 OpenEnv API

```python
from openenv import GenericEnvClient, GenericAction

async with GenericEnvClient(base_url="https://your-space.hf.space") as env:
    obs = await env.reset()
    result = await env.step(GenericAction(
        fixed_code="def sum_list(nums):\n    return sum(nums)",
        use_hint=False,
    ))
    print(result["reward"])
```

---

## 🚀 Local Setup

```bash
pip install -r requirements.txt
uvicorn server.main:app --reload --port 7860
pytest tests/ -v
```

---

## 🐳 Docker

```bash
docker build -t code-debug-env .
docker run -p 7860:7860 code-debug-env
```

---

## 📦 Deploy to HF Spaces

```bash
openenv push --space YOUR_USERNAME/code-debug-env
```