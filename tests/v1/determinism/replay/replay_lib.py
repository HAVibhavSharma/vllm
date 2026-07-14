# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared library for the divergence-replay suite (see PLAN.md).

Replays LLM non-determinism cases harvested from trace-analyser reports:
same input, N_TRIES strictly sequential calls against the original
endpoint/config, each try logged as a root llm run in a dedicated
LangSmith project so trace_analyser.py can be pointed at it unchanged.

Stdlib-only (mirrors trace_analyser.py); divergence math is imported from
trace_analyser.py itself so the metric is apples-to-apples.
"""
import importlib.util
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone

REPLAY_DIR = os.path.dirname(os.path.abspath(__file__))
CASES_DIR = os.path.join(REPLAY_DIR, "cases")
RESULTS_DIR = os.path.join(REPLAY_DIR, "results")


# ─────────────────────────── .env / config ────────────────────────────────
def load_dotenv(path=None):
    """Populate os.environ from replay/.env. Real env vars win."""
    path = path or os.path.join(REPLAY_DIR, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(),
                                  val.strip().strip('"').strip("'"))


load_dotenv()


def cfg(key, default=None):
    return os.environ.get(key, default)


def cfg_int(key, default):
    return int(cfg(key, str(default)))


def cfg_flag(key, default=True):
    v = cfg(key)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off")


N_TRIES = cfg_int("N_TRIES", 5)
LS_API_KEY = cfg("LANGSMITH_API_KEY")
LS_ENDPOINT = cfg("LANGSMITH_ENDPOINT",
                  "https://api.smith.langchain.com").rstrip("/")
MODEL_BASE_URL = (cfg("REPLAY_MODEL_BASE_URL") or "").rstrip("/")
MODEL_API_KEY = cfg("REPLAY_MODEL_API_KEY", "EMPTY")
TRACE_ANALYSIS_DIR = os.path.expanduser(
    cfg("TRACE_ANALYSIS_DIR", "~/Projects/trace-analyser"))
LS_PROJECT_PREFIX = cfg("REPLAY_LS_PROJECT_PREFIX", "repro")


# ────────────────── trace_analyser import (metric parity) ─────────────────
_ta = None


def trace_analyser():
    """Import trace_analyser.py from TRACE_ANALYSIS_DIR for divergence()/
    scrub_ids()/NOISE so replay verdicts use the exact original metric."""
    global _ta
    if _ta is None:
        path = os.path.join(TRACE_ANALYSIS_DIR, "trace_analyser.py")
        spec = importlib.util.spec_from_file_location("trace_analyser", path)
        _ta = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_ta)
    return _ta


def noise_threshold():
    return trace_analyser().NOISE


def divergence(texts):
    ta = trace_analyser()
    return ta.divergence([ta.scrub_ids(t) for t in texts])


# ─────────────────────────── HTTP helpers ─────────────────────────────────
class ReplayError(Exception):
    pass


def _request(url, body=None, headers=None, method=None, tries=5, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {},
                                         method=method or
                                         ("POST" if data else "GET"))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            detail = e.read()[:400].decode("utf-8", "replace")
            if e.code == 429 and attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise ReplayError("HTTP %s on %s: %s" % (e.code, url, detail))
        except urllib.error.URLError as e:
            if attempt < tries - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise ReplayError("network error on %s: %s" % (url, e))


def ls_call(path, body=None, method=None):
    if not LS_API_KEY:
        raise ReplayError("LANGSMITH_API_KEY not set (replay/.env)")
    return _request(LS_ENDPOINT + path, body,
                    {"x-api-key": LS_API_KEY,
                     "Content-Type": "application/json"}, method)


# ─────────────────────────── LangSmith: read side ─────────────────────────
def list_root_runs(project_id):
    runs = ls_call("/runs/query", {
        "session": [project_id], "is_root": True, "limit": 100,
        "select": ["id", "name", "start_time"]})["runs"]
    runs.sort(key=lambda r: r.get("start_time") or "")
    return runs


def pull_tree(trace_id):
    out, cursor = [], None
    while True:
        body = {"trace": trace_id, "limit": 100,
                "select": ["id", "name", "run_type", "parent_run_id",
                           "dotted_order", "inputs", "outputs", "extra",
                           "status", "error"]}
        if cursor:
            body["cursor"] = cursor
        rv = ls_call("/runs/query", body)
        out += rv["runs"]
        cursor = rv.get("cursors", {}).get("next")
        time.sleep(0.3)
        if not cursor:
            break
    return out


def llm_sequence(tree):
    """Successful llm runs in execution order — index-compatible with
    trace_analyser's content_analysis positions (it filters events to
    status=success and keeps dotted_order, then takes kind==LLM)."""
    ok = [r for r in tree
          if r.get("status") == "success" and not r.get("error")]
    ok.sort(key=lambda r: r.get("dotted_order") or "")
    return [r for r in ok if r.get("run_type") == "llm"]


def prompt_tokens(run):
    """Prompt/input token count of a source llm run; None if unrecorded."""
    o = run.get("outputs") or {}
    tu = (o.get("llm_output") or {}).get("token_usage") or {}
    if isinstance(tu.get("prompt_tokens"), int):
        return tu["prompt_tokens"]
    g = o.get("generations")
    while isinstance(g, list) and g:
        g = g[0]
    if isinstance(g, dict):
        um = ((g.get("message") or {}).get("kwargs") or {}) \
            .get("usage_metadata") or {}
        if isinstance(um.get("input_tokens"), int):
            return um["input_tokens"]
    return None


# ──────────────── LangChain messages → OpenAI chat format ─────────────────
_ROLE = {"human": "user", "ai": "assistant", "system": "system",
         "tool": "tool", "function": "function", "chat": "user"}


def _msg_kwargs(m):
    if not isinstance(m, dict):
        return {}
    return m.get("kwargs") if isinstance(m.get("kwargs"), dict) else m


def to_openai_messages(raw_inputs):
    """Convert a LangSmith llm run's serialized LangChain messages into
    OpenAI chat-completions messages."""
    ms = raw_inputs.get("messages") if isinstance(raw_inputs, dict) \
        else raw_inputs
    while isinstance(ms, list) and ms and isinstance(ms[0], list):
        ms = ms[0]
    if not isinstance(ms, list):
        raise ReplayError("cannot locate message list in run inputs")
    out = []
    for m in ms:
        kw = _msg_kwargs(m)
        role = _ROLE.get(kw.get("type") or kw.get("role"), None)
        if role is None:
            raise ReplayError("unknown message type: %r"
                              % (kw.get("type") or kw.get("role")))
        msg = {"role": role}
        content = kw.get("content")
        msg["content"] = content if isinstance(content, (str, list)) \
            else (json.dumps(content) if content is not None else "")
        if role == "assistant" and kw.get("tool_calls"):
            msg["tool_calls"] = [{
                "id": tc.get("id") or ("call_%d" % i),
                "type": "function",
                "function": {"name": tc.get("name") or "",
                             "arguments": json.dumps(tc.get("args") or {},
                                                     ensure_ascii=False)},
            } for i, tc in enumerate(kw["tool_calls"])]
        if role == "tool":
            tcid = kw.get("tool_call_id")
            if tcid:
                msg["tool_call_id"] = tcid
        if kw.get("name") and role in ("tool", "function"):
            msg["name"] = kw["name"]
        out.append(msg)
    return out


def request_params(run):
    """Exact original sampling config of a source llm run, from
    extra.invocation_params with ls_* metadata as fallback."""
    extra = run.get("extra") or {}
    inv = extra.get("invocation_params") or {}
    md = extra.get("metadata") or {}
    params = {}
    model = inv.get("model") or inv.get("model_name") or md.get(
        "ls_model_name")
    if not model:
        raise ReplayError("cannot determine model name for run %s"
                          % run.get("id"))
    params["model"] = model
    for src_key, dst_key in (("temperature", "temperature"),
                             ("top_p", "top_p"), ("seed", "seed"),
                             ("max_tokens", "max_tokens"), ("stop", "stop"),
                             ("n", "n"), ("presence_penalty",
                                          "presence_penalty"),
                             ("frequency_penalty", "frequency_penalty")):
        v = inv.get(src_key)
        if v is None:
            v = md.get("ls_" + src_key)
        if v is not None:
            params[dst_key] = v
    if inv.get("tools"):
        params["tools"] = inv["tools"]
        if inv.get("tool_choice") is not None:
            params["tool_choice"] = inv["tool_choice"]
    return params


# ─────────────────────────── replay: model call ───────────────────────────
def chat_completion(messages, params):
    if not MODEL_BASE_URL:
        raise ReplayError("REPLAY_MODEL_BASE_URL not set (replay/.env)")
    body = dict(params)
    body["messages"] = messages
    body["stream"] = False
    return _request(MODEL_BASE_URL + "/chat/completions", body,
                    {"Authorization": "Bearer " + MODEL_API_KEY,
                     "Content-Type": "application/json"})


def completion_text(resp):
    """Render a chat completion the way trace_analyser.out_text renders an
    llm run: content, then tool calls as name(sorted-args-json)."""
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    parts = []
    if msg.get("content"):
        parts.append(msg["content"])
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        try:
            args = json.dumps(json.loads(fn.get("arguments") or "{}"),
                              sort_keys=True, ensure_ascii=False)
        except (ValueError, TypeError):
            args = fn.get("arguments") or ""
        parts.append("%s(%s)" % (fn.get("name") or "", args))
    return "\n".join(parts)


# ─────────────────────────── LangSmith: write side ────────────────────────
def ensure_project(name):
    """Create (or find) a LangSmith project; return its id."""
    try:
        return ls_call("/sessions", {"name": name})["id"]
    except ReplayError as e:
        if "409" not in str(e):
            raise
    hits = ls_call("/sessions?name=" + urllib.parse.quote(name))
    if isinstance(hits, list) and hits:
        return hits[0]["id"]
    raise ReplayError("project %r exists but could not be fetched" % name)


def _now():
    return datetime.now(timezone.utc)


def _iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def log_try_run(project_name, try_no, case, raw_inputs, resp, text,
                started, ended):
    """Log one replay try as a ROOT llm run shaped exactly like the runs
    trace_analyser consumes (generations/message/kwargs, ls_* metadata)."""
    run_id = str(uuid.uuid4())
    dotted = started.strftime("%Y%m%dT%H%M%S%fZ") + run_id
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    tool_calls = [{"name": (tc.get("function") or {}).get("name"),
                   "args": _safe_json_loads(
                       (tc.get("function") or {}).get("arguments")),
                   "id": tc.get("id")}
                  for tc in (msg.get("tool_calls") or [])]
    gen_kwargs = {"content": msg.get("content") or "", "type": "ai",
                  "id": resp.get("id")}
    if tool_calls:
        gen_kwargs["tool_calls"] = tool_calls
    p = case["request"]["params"]
    metadata = {"ls_model_name": p.get("model"),
                "ls_temperature": p.get("temperature"),
                "ls_top_p": p.get("top_p"), "ls_seed": p.get("seed"),
                "ls_max_tokens": p.get("max_tokens"),
                "ls_provider": (case.get("original", {}).get("config") or
                                {}).get("provider"),
                "replay_case_id": case["case_id"],
                "replay_try": try_no}
    body = {
        "id": run_id, "trace_id": run_id, "dotted_order": dotted,
        "name": "try-%02d" % try_no, "run_type": "llm",
        "start_time": _iso(started), "end_time": _iso(ended),
        "inputs": raw_inputs,
        "outputs": {
            "generations": [[{"text": text,
                              "message": {"kwargs": gen_kwargs}}]],
            "llm_output": {"token_usage": resp.get("usage") or {}},
        },
        "extra": {"metadata": metadata},
        "session_name": project_name,
        "status": "success",
    }
    ls_call("/runs", body)
    return run_id


def _safe_json_loads(s):
    try:
        return json.loads(s) if s else {}
    except (ValueError, TypeError):
        return {"_raw": s}


# ─────────────────────────── case store + replay ──────────────────────────
def load_cases():
    """Cases extracted by extract_cases.py, sorted by case_id."""
    if not os.path.isdir(CASES_DIR):
        return []
    cases = []
    for fn in sorted(os.listdir(CASES_DIR)):
        if fn.endswith(".json"):
            with open(os.path.join(CASES_DIR, fn)) as f:
                data = json.load(f)
            if isinstance(data, dict) and "case_id" in data:
                cases.append(data)
    return cases


def replay_case(case, n_tries=None):
    """Sequential n-try replay of one case. Returns the result record and
    writes it to results/<case_id>.json; every try is logged to the case's
    own LangSmith project for trace_analyser consumption."""
    n = n_tries or N_TRIES
    project = "%s-%s" % (LS_PROJECT_PREFIX, case["case_id"])
    ensure_project(project)
    messages = to_openai_messages(case["raw_inputs"])
    outputs, run_ids = [], []
    for k in range(1, n + 1):
        started = _now()
        resp = chat_completion(messages, case["request"]["params"])
        ended = _now()
        text = completion_text(resp)
        outputs.append(text)
        run_ids.append(log_try_run(project, k, case, case["raw_inputs"],
                                   resp, text, started, ended))
    div = divergence(outputs)
    noise = noise_threshold()
    result = {
        "case_id": case["case_id"],
        "langsmith_project": project,
        "n_tries": n,
        "divergence": round(div, 4),
        "noise_threshold": noise,
        "reproduced": div > noise,
        "distinct_outputs": len(set(outputs)),
        "original_output_divergence":
            case.get("original", {}).get("output_divergence"),
        "run_ids": run_ids,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, case["case_id"] + ".json"),
              "w") as f:
        json.dump(result, f, indent=2)
    return result
