# giants_api.py  (UPDATED WITH MariaDB + Upload)
import os
import json
import csv
import time
import asyncio
import datetime as dt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import requests
import mysql.connector   # 🟢 DB
import pandas as pd      # 🟢 For file preview

# -------------------- CONFIG --------------------
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8080))
DEBUG = os.getenv("DEBUG", "1") == "1"

TGPT_BIN = os.getenv("TGPT_BIN", "tgpt")
TGPT_TIMEOUT = int(os.getenv("TGPT_TIMEOUT", "75"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")

DATASET_DIR = Path(os.getenv("DATASET_DIR", "./datasets"))
DATASET_DIR.mkdir(parents=True, exist_ok=True)

VALID_TGPT_PROVIDERS = ["pollinations", "sky", "phind", "koboldai"]
GROUP_LIST = VALID_TGPT_PROVIDERS[:]  # All tgpt providers

executor = ThreadPoolExecutor(max_workers=5)

# -------------------- DB CONFIG --------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "chatbot",
    "password": "strongpassword",
    "database": "giants"
}

def get_db_conn():
    return mysql.connector.connect(**DB_CONFIG)

def init_tables():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS agi_giants_dialogue (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(64),
            provider VARCHAR(50),
            query TEXT,
            reply TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def save_to_db(session_id, provider, query, reply):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO agi_giants_dialogue (session_id, provider, query, reply)
            VALUES (%s, %s, %s, %s)
        """, (session_id, provider, query, reply))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("DB error:", e)

# -------------------- UTILS --------------------
def _now_str():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def dataset_path(filename: str, fmt: str):
    safe_name = (filename or f"dataset_{int(time.time())}").strip().replace("/", "_")
    ext = ".csv" if fmt=="csv" else ".json"
    return DATASET_DIR / f"{safe_name}{ext}"

def append_dataset_lines(fp: Path, session_id: str, fmt: str, entries):
    if fmt == "csv":
        exists = fp.exists()
        with fp.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["timestamp","session_id","provider","message","type"])
            for e in entries:
                writer.writerow([e["timestamp"], session_id, e["provider"], e["message"], e.get("type","bot")])
    else:
        with fp.open("a", encoding="utf-8") as f:
            for e in entries:
                obj = {"timestamp": e["timestamp"], "session_id": session_id, "provider": e["provider"], "message": e["message"], "type": e.get("type","bot")}
                f.write(json.dumps(obj, ensure_ascii=False)+"\n")

# -------------------- PROVIDER CALLS --------------------
def query_tgpt(provider: str, prompt: str):
    try:
        res = subprocess.run(
            [TGPT_BIN, "-w", "--provider", provider, prompt],
            capture_output=True, text=True, timeout=TGPT_TIMEOUT
        )
        return res.stdout.strip() or f"[tgpt {provider}] (no output)"
    except Exception as e:
        return f"[tgpt {provider} error] {e}"

def list_ollama_models():
    try:
        url = f"http://{OLLAMA_HOST}/api/tags"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models",[])]
    except Exception as e:
        return []

def query_ollama(model: str, prompt: str):
    try:
        url = f"http://{OLLAMA_HOST}/api/generate"
        data = {"model": model, "prompt": prompt, "stream": False}
        resp = requests.post(url, json=data, timeout=300)
        resp.raise_for_status()
        j = resp.json()
        return j.get("response") or j.get("text") or j.get("result") or ""
    except Exception as e:
        return f"[Ollama error: {e}]"

# -------------------- ASYNC ORCHESTRATION --------------------
async def ask_providers_parallel(providers, prompt: str):
    loop = asyncio.get_event_loop()
    tasks = []
    results = []

    ollama_models = list_ollama_models()
    normalized = {}
    for m in ollama_models:
        normalized[m.lower()] = m
        if ":" in m:
            normalized[m.split(":")[0].lower()] = m

    for p in providers:
        if p is None: continue
        p_strip = p.strip()
        if p_strip.lower() in normalized:
            model_name = normalized[p_strip.lower()]
            tasks.append(loop.run_in_executor(executor, query_ollama, model_name, prompt))
        elif p_strip in VALID_TGPT_PROVIDERS:
            tasks.append(loop.run_in_executor(executor, query_tgpt, p_strip, prompt))
        else:
            tasks.append(loop.run_in_executor(executor, lambda: f"[invalid provider: {p_strip}]"))

    replies_raw = await asyncio.gather(*tasks)
    for prov, reply in zip(providers, replies_raw):
        results.append({"provider": prov, "reply": reply, "timestamp": _now_str()})
    return results

# -------------------- BRAINSTORM / ROUNDTABLE PROMPT BUILDERS --------------------
def summarize_text(text: str, max_chars=300):
    if not text:
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rsplit(" ",1)[0] + " ..."

def build_brainstorm_prompt_from_replies(original_query: str, replies, file_comment: str = None, max_examples=8):
    lines = []
    lines.append("You are an Editor/Moderator for a short expert roundtable.")
    lines.append(f"User question: {original_query}")
    if file_comment:
        lines.append(f"File note: {file_comment}")
    lines.append("")
    lines.append("Below are short excerpts from the experts' first replies (provider: excerpt).")
    examples = 0
    for r in replies:
        if examples >= max_examples: break
        prov = r.get("provider","unknown")
        excerpt = summarize_text(r.get("reply",""), max_chars=300)
        if excerpt:
            lines.append(f"- {prov}: {excerpt}")
            examples += 1
    lines.append("")
    lines.append("Task: Produce two outputs:")
    lines.append("1) A short roundtable transcript (3-6 brief turns) where experts comment on each other's key points.")
    lines.append("2) A concise final answer (1-6 sentences) that synthesizes and cites key providers.")
    lines.append("")
    lines.append("Keep the transcript focused, concise, and avoid repeating full replies verbatim.")
    return "\n".join(lines)

# -------------------- FLASK APP --------------------
app = Flask(__name__)
CORS(app)

@app.route("/giants", methods=["POST"])
def giants():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id","default")
    providers = data.get("providers", ["group"])
    editor = data.get("editor", providers[0])
    query = data.get("query","").strip()
    file_comment = data.get("file_comment", None)
    save_dataset = bool(data.get("save_dataset", False))
    save_format = (data.get("save_format","csv")).lower()
    filename = data.get("filename","")

    if "group" in providers:
        providers = GROUP_LIST

    prompt = query
    if file_comment:
        prompt = f"User question: {query}\nFile note: {file_comment}\nPlease answer considering the file note above."

    replies = asyncio.run(ask_providers_parallel(providers, prompt))

    # 🟢 Save DB
    for r in replies:
        save_to_db(session_id, r["provider"], query, r["reply"])

    if save_dataset:
        fp = dataset_path(filename, save_format)
        append_dataset_lines(fp, session_id, save_format, [{"timestamp": r["timestamp"], "provider": r["provider"], "message": r["reply"]} for r in replies])

    final_answer = next((r["reply"] for r in replies if r["provider"]==editor), None)
    if not final_answer and replies:
        final_answer = replies[0]["reply"]

    return jsonify({"raw_replies": replies, "roundtable": None, "final_answer": final_answer})

@app.route("/giants_roundtable", methods=["POST"])
def giants_roundtable():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id","default")
    providers = data.get("providers", ["group"])
    editor = data.get("editor", providers[0])
    query = data.get("query","").strip()
    file_comment = data.get("file_comment", None)
    save_dataset = bool(data.get("save_dataset", False))
    save_format = (data.get("save_format","csv")).lower()
    filename = data.get("filename","")

    if "group" in providers:
        providers = GROUP_LIST

    prompt_for_providers = query
    if file_comment:
        prompt_for_providers = f"User question: {query}\nFile note: {file_comment}\nPlease answer considering the file note above."

    initial_replies = asyncio.run(ask_providers_parallel(providers, prompt_for_providers))

    # 🟢 Save DB
    for r in initial_replies:
        save_to_db(session_id, r["provider"], query, r["reply"])

    compact_prompt = build_brainstorm_prompt_from_replies(query, initial_replies, file_comment=file_comment)

    ollama_models = list_ollama_models()
    normalized = {}
    for m in ollama_models:
        normalized[m.lower()] = m
        if ":" in m:
            normalized[m.split(":")[0].lower()] = m

    if editor and editor.strip().lower() in normalized:
        editor_model = normalized[editor.strip().lower()]
        roundtable_text = query_ollama(editor_model, compact_prompt)
        final_answer = query_ollama(editor_model, f"Synthesize a concise final answer for the user.\n\nContext:\n{compact_prompt}\n\nRoundtable transcript:\n{roundtable_text}")
    elif editor in VALID_TGPT_PROVIDERS:
        roundtable_text = query_tgpt(editor, compact_prompt)
        final_answer = query_tgpt(editor, f"Synthesize a concise final answer for the user.\n\nContext:\n{compact_prompt}\n\nRoundtable transcript:\n{roundtable_text}")
    else:
        roundtable_text = "\n".join([f"{r['provider']}: {summarize_text(r['reply'],200)}" for r in initial_replies])
        final_answer = initial_replies[0]["reply"] if initial_replies else "(no reply)"

    if save_dataset:
        fp = dataset_path(filename, save_format)
        to_save = [{"timestamp": r["timestamp"], "provider": r["provider"], "message": r["reply"]} for r in initial_replies]
        to_save.append({"timestamp": _now_str(), "provider": "roundtable", "message": roundtable_text})
        append_dataset_lines(fp, session_id, save_format, to_save)

    # 🟢 Save DB final roundtable
    save_to_db(session_id, "roundtable", query, roundtable_text)
    save_to_db(session_id, "editor", query, final_answer)

    return jsonify({
        "raw_replies": initial_replies,
        "roundtable": roundtable_text,
        "final_answer": final_answer
    })

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    filename = f.filename
    filepath = DATASET_DIR / filename
    f.save(filepath)

    summary = "File uploaded"
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(filepath, nrows=10)
            summary = f"CSV preview: {df.shape[0]} rows x {df.shape[1]} cols, headers={list(df.columns)}"
        elif filename.endswith(".xlsx"):
            xls = pd.ExcelFile(filepath)
            first_sheet = xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=first_sheet, nrows=10)
            summary = f"Excel sheet '{first_sheet}': {df.shape[0]} rows x {df.shape[1]} cols, headers={list(df.columns)}"
        else:
            with open(filepath, "r", errors="ignore") as t:
                text = t.read(200)
            summary = f"Text preview: {text}"
    except Exception as e:
        summary = f"Could not parse file: {e}"

    return jsonify({"filename": filename, "summary": summary})

@app.route("/providers", methods=["GET"])
def providers_list():
    ollama_models = list_ollama_models()
    return jsonify({"tgpt_providers": VALID_TGPT_PROVIDERS, "ollama_models": ollama_models})

# -------------------- RUN --------------------
if __name__ == "__main__":
    init_tables()
    app.run(host=HOST, port=PORT, debug=DEBUG)
