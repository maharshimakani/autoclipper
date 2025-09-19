#!/usr/bin/env python3
# Auto-Clipper for Viral Hooks — Streamlit UI (hardened + yt-dlp caption fallback)
# Developer: Maharshi PATEL

# --- Force Transformers to use PyTorch only (avoid TF/Keras import path) ---
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import io
import json
import time
import math
import shutil
import tempfile
import subprocess
from typing import List, Tuple, Optional, Iterable, Dict
from urllib.parse import urlparse, parse_qs
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Audio & CV
import librosa
import cv2

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

# YouTube subs + translate
from youtube_transcript_api import (
    YouTubeTranscriptApi, NoTranscriptFound,
    TranscriptsDisabled, CouldNotRetrieveTranscript
)
from deep_translator import GoogleTranslator

# yt-dlp (Python API)
from yt_dlp import YoutubeDL

# HTTP for .vtt fallback
import requests

# ----------------------------- Config -----------------------------

DEFAULT_WINDOW = 45   # seconds per short
DEFAULT_STRIDE = 12   # sliding step
DEFAULT_TOPK  = 3     # number of highlights
MIN_GAP_SEC   = 25    # spacing between picks
SAMPLE_RATE   = 16000 # audio sample rate for analysis
BATCH_SIZE    = 16    # HF pipeline batch size

HOOK_PHRASES = [
    "the secret", "here's why", "watch this", "you need to", "let me show you",
    "in 3 steps", "pro tip", "must know", "the trick", "do this", "what you should",
    "important", "key idea", "pay attention", "this is why", "that’s why",
    "hack", "tip", "quick tip", "short answer", "here’s how", "listen to this",
    "the truth is", "nobody tells you", "game changer"
]

HF_MODELS = {
    "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
    "summarizer": "sshleifer/distilbart-cnn-12-6",
    "ner": "dslim/bert-base-NER"
}

SAMPLE_VIDEO = "https://www.youtube.com/watch?v=iH_6av3Yj0g"  # reliable captions

# ----------------------------- Utils -----------------------------

def ensure_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

def extract_video_id(url_or_id: str) -> Optional[str]:
    s = (url_or_id or "").strip()
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", s):
        return s
    try:
        u = urlparse(s)
        host = (u.netloc or "").lower()
        path = u.path or ""
        qs = parse_qs(u.query or "")
        if "v" in qs and re.fullmatch(r"[0-9A-Za-z_-]{11}", qs["v"][0]): return qs["v"][0]
        if "youtu.be" in host:
            m = re.match(r"^/([0-9A-Za-z_-]{11})", path);  return m.group(1) if m else None
        m = re.match(r"^/shorts/([0-9A-Za-z_-]{11})", path)
        if m: return m.group(1)
        m = re.match(r"^/embed/([0-9A-Za-z_-]{11})", path)
        if m: return m.group(1)
        m = re.search(r"([0-9A-Za-z_-]{11})", s)
        return m.group(1) if m else None
    except Exception:
        return None

def translate_lines_safe(lines: List[dict], src_lang: str, target="en") -> List[dict]:
    """Translate line-by-line (safer alignment), with a progress bar."""
    if not lines or (src_lang or "").lower().startswith("en"):
        return lines
    out = []
    prog = st.progress(0, text=f"Translating {len(lines)} lines {src_lang}→{target}…")
    for i, ln in enumerate(lines, 1):
        txt = ln.get("text", "")
        try:
            tr = GoogleTranslator(source=src_lang or "auto", target=target).translate(txt) if txt else txt
        except Exception:
            tr = txt
        out.append({**ln, "text": tr})
        if i % 5 == 0 or i == len(lines):
            prog.progress(i/len(lines))
    prog.empty()
    return out

def to_dataframe(lines: List[dict]) -> pd.DataFrame:
    rows = []
    for ln in lines:
        start = float(ln.get("start", 0.0))
        dur   = float(ln.get("duration", 0.0))
        txt   = str(ln.get("text",""))
        txt = txt.replace("\n"," ").strip()
        if txt:
            rows.append({"start": start, "end": start+dur, "text": txt})
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)

def window_iter(df: pd.DataFrame, win: float, stride: float, total: Optional[float]) -> Iterable[Tuple[float,float,pd.DataFrame]]:
    if total is None:
        total = df["end"].max() if not df.empty else 0.0
    t = 0.0
    while t < max(total - 1e-6, 0):
        wstart = t
        wend   = min(t + win, total)
        chunk  = df[(df["start"] < wend) & (df["end"] > wstart)]
        yield (wstart, wend, chunk)
        t += stride
        if wend >= total: break

def zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0: return x
    m, s = np.nanmean(x), np.nanstd(x) + 1e-8
    return (x - m) / s

def non_max_suppression(picks: List[Tuple[float,float,float]], min_gap: float) -> List[Tuple[float,float,float]]:
    picks_sorted = sorted(picks, key=lambda x: x[0], reverse=True)
    kept = []
    for s, a, b in picks_sorted:
        overlap = any((ka <= a <= kb) or (ka <= b <= kb) or (a <= ka and b >= kb) for _, ka, kb in kept)
        close   = any((abs(a - ka) < min_gap) or (abs(b - kb) < min_gap) for _, ka, kb in kept)
        if not overlap and not close:
            kept.append((s, a, b))
    return sorted(kept, key=lambda x: x[1])

def hhmmss(t: float) -> str:
    t = max(0, float(t))
    h = int(t // 3600); m = int((t % 3600)//60); s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h>0 else f"{m:02d}:{s:02d}"

# ------------------ NEW: WebVTT parsing + yt-dlp captions --------

def parse_webvtt(vtt_text: str):
    """
    Minimal WebVTT parser -> list[{"start": float, "duration": float, "text": str}]
    Supports typical YouTube .vtt from yt-dlp.
    """
    def to_seconds(ts: str) -> float:
        # 00:01:23.456  or  01:23.456
        parts = ts.strip().split(":")
        if len(parts) == 3:
            h, m, s = parts
        else:
            h, m, s = "0", parts[0], parts[1]
        return int(h)*3600 + int(m)*60 + float(s.replace(",", "."))
    lines = []
    text_buf = []
    start = end = None
    for raw in vtt_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "-->" in line:
            # flush previous cue
            if start is not None and text_buf:
                txt = " ".join(text_buf).strip()
                if txt:
                    lines.append({"start": start, "duration": max(0.0, end - start), "text": txt})
            # new cue
            text_buf = []
            m = re.search(r"([0-9:\.,]+)\s*-->\s*([0-9:\.,]+)", line)
            if not m:
                start = end = None
                continue
            start = to_seconds(m.group(1))
            end   = to_seconds(m.group(2))
        else:
            if line.upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
                continue
            text_buf.append(line)
    # last cue
    if start is not None and text_buf:
        txt = " ".join(text_buf).strip()
        if txt:
            lines.append({"start": start, "duration": max(0.0, end - start), "text": txt})
    return lines

def ytdlp_fetch_captions(url: str, prefer_langs=("en","en-US","en-GB","fr","es")):
    """
    Use yt-dlp to fetch captions (manual first, then auto). Returns (list_of_lines, language_code or '').
    """
    # 1) Probe metadata to see what captions exist
    info = None
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        info = None

    def try_download(sub_map_key: str):
        if not info or sub_map_key not in info or not isinstance(info[sub_map_key], dict):
            return [], ""
        subs_map = info[sub_map_key]  # {lang: [{ext:'vtt', url:...}, ...]}
        for lang in list(prefer_langs) + list(subs_map.keys()):
            tracks = subs_map.get(lang)
            if not tracks:
                continue
            track = next((t for t in tracks if t.get("ext") == "vtt"), tracks[0])
            try:
                r = requests.get(track["url"], timeout=15)
                if r.ok:
                    vtt = r.text
                    lines = parse_webvtt(vtt)
                    if lines:
                        return lines, lang
            except Exception:
                continue
        return [], ""

    # Try manual captions first, then auto
    lines, lang = try_download("subtitles")
    if lines:
        return lines, lang
    lines, lang = try_download("automatic_captions")
    return lines, lang

# ---------------------- Audio, Vision, NLP -----------------------

def download_media_ytdlp(video_url: str, workdir: Path, need_video: bool) -> Tuple[Optional[Path], Optional[Path], Optional[float]]:
    """
    Download audio (wav) and optionally a small mp4 using yt-dlp Python API.
    Returns (wav_path, mp4_path, duration_sec).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    duration = None

    # metadata
    try:
        with YoutubeDL({"quiet": True, "skip_download": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            duration = float(info.get("duration") or 0.0)
    except Exception:
        pass

    # audio → wav
    wav_path = None
    ydl_opts_audio = {
        "quiet": True, "no_warnings": True,
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "format": "bestaudio/best",
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"}
        ]
    }
    try:
        with YoutubeDL(ydl_opts_audio) as ydl:
            info_a = ydl.extract_info(video_url, download=True)
            vid_id = info_a.get("id")
            candidate = list(workdir.glob(f"{vid_id}.wav"))
            if candidate:
                wav_path = candidate[0]
    except Exception:
        wav_path = None

    mp4_path = None
    if need_video:
        ydl_opts_video = {
            "quiet": True, "no_warnings": True,
            "outtmpl": str(workdir / "%(id)s.%(ext)s"),
            "format": "mp4[height<=360]/bv*[height<=360]+ba/best[ext=mp4]/best",
        }
        try:
            with YoutubeDL(ydl_opts_video) as ydl:
                info_v = ydl.extract_info(video_url, download=True)
                vid_id = info_v.get("id")
                candidate = list(workdir.glob(f"{vid_id}.mp4"))
                if candidate:
                    mp4_path = candidate[0]
        except Exception:
            mp4_path = None

    return wav_path, mp4_path, duration

def audio_energy_series(wav_path: Path, hop_s: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    hop_length = int(sr * hop_s)
    frame_length = int(sr * 0.5)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).flatten()
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    if len(rms) >= 5:
        rms = np.convolve(rms, np.ones(5)/5, mode="same")
    return times, rms

@st.cache_resource(show_spinner=False)
def get_pipes():
    ensure_nltk()
    stw = set(stopwords.words("english"))
    sent = pipeline("sentiment-analysis",
                    model=HF_MODELS["sentiment"],
                    framework="pt",
                    device=-1)
    summ = pipeline("summarization",
                    model=HF_MODELS["summarizer"],
                    framework="pt",
                    device=-1)
    ner  = pipeline("token-classification",
                    model=HF_MODELS["ner"],
                    aggregation_strategy="simple",
                    framework="pt",
                    device=-1)
    return stw, sent, summ, ner

def batch_predict_sentiment(sent_pipe, texts: List[str]) -> List[float]:
    scores = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = [t[:512] for t in texts[i:i+BATCH_SIZE]]
        try:
            out = sent_pipe(chunk)
        except Exception:
            out = [{"label":"POSITIVE", "score":0.0} for _ in chunk]
        for r in out:
            scores.append(float(r.get("score", 0.0)))
    return scores

def batch_predict_ner(ner_pipe, texts: List[str]) -> List[int]:
    counts = []
    for i in range(0, len(texts), BATCH_SIZE):
        chunk = texts[i:i+BATCH_SIZE]
        try:
            out = ner_pipe(chunk)
        except Exception:
            out = [[] for _ in chunk]
        for ents in out:
            counts.append(len(ents) if isinstance(ents, list) else 0)
    return counts

def face_emotion_series_lazy(mp4_path: Path, step_s: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Import FER + TF only when needed."""
    try:
        from fer import FER  # lazy import (will trigger TF)
    except Exception:
        return np.array([]), np.array([])
    detector = FER(mtcnn=False)
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        return np.array([]), np.array([])
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(step_s * fps))
    idx = 0; ts_list = []; score_list = []
    while True:
        ret = cap.grab()
        if not ret: break
        if idx % step == 0:
            ret2, frame = cap.retrieve()
            if not ret2: break
            t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
            try:
                emos = detector.detect_emotions(frame) or []
                emo_score = 0.0
                if emos:
                    em = emos[0]["emotions"]
                    emo_score = float(max(em.get("happy",0), em.get("surprise",0)))
                ts_list.append(t); score_list.append(emo_score)
            except Exception:
                ts_list.append(t); score_list.append(0.0)
        idx += 1
    cap.release()
    ts = np.array(ts_list, dtype=float)
    sc = np.array(score_list, dtype=float)
    if sc.size >= 5:
        sc = np.convolve(sc, np.ones(5)/5, mode="same")
    return ts, sc

# ---------------------- Scoring windows --------------------------

def score_features_bulk(texts: List[str], stw: set, sent_pipe, ner_pipe) -> Dict[str, List[float]]:
    # Hooks
    hooks = [sum(1 for h in HOOK_PHRASES if h in t.lower()) for t in texts]
    # Sentiment magnitude
    sent_mag = batch_predict_sentiment(sent_pipe, texts)
    # Keyword density
    kw_density = []
    for t in texts:
        low = t.lower()
        toks = [w for w in word_tokenize(low) if w.isalnum() and w not in stw]
        kw_density.append(min(len(toks)/max(len(t.split()),1), 1.0))
    # Entities count
    entities = batch_predict_ner(ner_pipe, texts)
    # Punctuation bonus
    punct = [0.2 * (("?" in t) or ("!" in t)) for t in texts]
    return {"hooks": hooks, "sent_mag": sent_mag, "kw_density": kw_density, "entities": entities, "punct": punct}

def aggregate_scores(windows_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["audio", "face", "hooks", "sent_mag", "kw_density", "entities", "punct"]
    for c in cols:
        if c not in windows_df.columns:
            windows_df[c] = 0.0
        arr = windows_df[c].to_numpy(dtype=float)
        windows_df[c+"_z"] = zscore(arr) if np.isfinite(arr).any() else 0.0
    w = {
        "audio_z": 0.9,
        "face_z":  0.6,
        "hooks_z": 1.1,
        "sent_mag_z": 0.9,
        "kw_density_z": 0.7,
        "entities_z": 0.4,
        "punct_z": 0.3
    }
    windows_df["score"] = sum(w.get(c,0)*windows_df.get(c,0.0) for c in w.keys())
    return windows_df

# ---------------------- Clip selection --------------------------

def pick_top_clips(windows_df: pd.DataFrame, topk: int, min_gap: float) -> List[Tuple[float,float,float]]:
    picks = [(float(r["score"]), float(r["start"]), float(r["end"])) for _, r in windows_df.iterrows()]
    kept = non_max_suppression(picks, min_gap=min_gap)
    if len(kept) > topk:
        kept = sorted(kept, key=lambda x: x[0], reverse=True)[:topk]
        kept = sorted(kept, key=lambda x: x[1])
    return kept

def title_for_clip(text: str, summarizer) -> str:
    text = (text or "").strip()
    if not text:
        return "Highlight"
    if len(text) < 80:
        return text[:80]
    try:
        out = summarizer(text[:800], max_length=18, min_length=6, do_sample=False)
        if out and isinstance(out, list):
            t = out[0].get("summary_text","").strip()
            return t[:80] if t else text[:80]
    except Exception:
        pass
    # Fallback: first sentence
    m = re.split(r"[.!?]", text)
    first = (m[0] if m else text).strip()
    return first[:80] if first else "Highlight"

# ---------------------- FFmpeg helpers --------------------------

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def clip_with_ffmpeg(src_mp4: Path, out_mp4: Path, start: float, end: float, vertical: bool = True):
    if not has_ffmpeg():
        raise RuntimeError("FFmpeg not found on PATH.")
    dur = max(0.5, end - start)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    vf = "crop=in_h*9/16:in_h:(in_w-in_h*9/16)/2:0" if vertical else "null"
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(src_mp4),
        "-t", f"{dur:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(out_mp4)
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# --------------------------- Streamlit UI ------------------------

# Minimal, enterprise-style header (no page icon, no emojis)
st.set_page_config(page_title="Auto-Clipper for Viral Hooks", layout="wide")

# Global style: subtle card layout, tighter spacing, professional tables
theme_css = """
    <style>
        /* Base */
        .stApp { background: #0b0d0f; }
        [data-testid="stHeader"] { background: transparent; }
        .app-title { font-weight: 650; letter-spacing: 0.2px; color: #e8edf2; margin-bottom: 0.25rem; }
        .app-subtitle { color: #9aa4af; font-size: 0.95rem; margin-top: 0.25rem; }

        /* Containers as cards */
        .card { background: #111418; border: 1px solid #1a1f24; border-radius: 14px; padding: 18px 18px; box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 8px 24px rgba(0,0,0,0.25); }
        .section-title { color: #d7dee6; font-weight: 600; margin: 0 0 12px 0; }

        /* Sidebar */
        section[data-testid="stSidebar"] { background: #0f1216; border-right: 1px solid #1a1f24; }
        .sidebar-group { border: 1px solid #1a1f24; background: #0c0f13; border-radius: 12px; padding: 12px; margin-bottom: 10px; }
        .sidebar-group h4 { color: #c9d3dc; font-size: 0.95rem; margin: 0 0 10px; font-weight: 600; }
        .help-text { color: #8b95a1; font-size: 0.85rem; margin-top: 6px; }

        /* Tables */
        .stDataFrame table { font-size: 0.92rem; }
        .stDataFrame [data-baseweb="table"] { background: #0f1216 !important; }
        .stDataFrame thead tr { background: #0f1216; }
        .stDataFrame tbody tr { border-bottom: 1px solid #1a1f24; }

        /* Buttons */
        .stButton>button { width: 100%; border-radius: 10px; border: 1px solid #2a3138; background: #151a20; color: #e8edf2; padding: 0.6rem 0.9rem; }
        .stButton>button:hover { border-color: #3a434c; background: #1a2027; }

        /* Inputs */
        .stTextInput input, .stNumberInput input { background: #0c0f13; color: #dee5ee; border-radius: 10px; border: 1px solid #1c2229; }
        .stCheckbox, .stRadio { color: #c9d3dc; }
        .stSelectbox div[data-baseweb="select"] { background: #0c0f13; border-radius: 10px; border: 1px solid #1c2229; }

        /* Info boxes */
        .stAlert { border-radius: 10px; border: 1px solid #1a1f24; }

        /* Spacing tweaks */
        .block-container { padding-top: 1.6rem; }
    </style>
"""
st.markdown(theme_css, unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div class="card">
        <div class="app-title">Auto-Clipper for Viral Hooks</div>
        <div class="app-subtitle">Paste a YouTube link to extract the most short-worthy 45s clips with titles, timecodes, and optional media analysis.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-group"><h4>Source</h4>', unsafe_allow_html=True)
    url = st.text_input("Video URL", placeholder=SAMPLE_VIDEO, help="YouTube link or ID. Supports standard, Shorts, and embed URLs.")
    use_sample = st.checkbox("Use sample video", value=False, help="Quickly test with a reliable video containing captions.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group"><h4>Scoring Window</h4>', unsafe_allow_html=True)
    num_clips = st.number_input("How many shorts", 1, 10, value=DEFAULT_TOPK, step=1, help="Number of top windows to return after suppression.")
    win = st.number_input("Clip length (sec)", 10, 120, value=DEFAULT_WINDOW, step=5, help="Duration of each candidate window.")
    stride = st.number_input("Stride (sec)", 5, 60, value=DEFAULT_STRIDE, step=1, help="Slide step between windows.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group"><h4>Analysis</h4>', unsafe_allow_html=True)
    include_audio = st.checkbox("Use audio energy", value=True, help="Leverages RMS energy peaks to improve picks.")
    include_face  = st.checkbox("Use face emotion if visible", value=False, help="Runs a lightweight FER model (slower, requires TF).")
    vertical_crop = st.checkbox("Vertical 9:16 crop for Shorts/Reels", value=True, help="Applied if you render clips locally with ffmpeg.")
    st.markdown('<div class="help-text">Tip: Face emotion analysis requires a low‑res MP4 fetch and may be CPU‑intensive.</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-group"><h4>Run</h4>', unsafe_allow_html=True)
    run_btn = st.button("Find Highlights")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

if not run_btn:
    st.info("Use the sample video in the sidebar to try the workflow, then paste your own link.")
    st.stop()

# --------------------------- Main run ---------------------------

url_to_use = SAMPLE_VIDEO if use_sample or not (url or "").strip() else url.strip()
vid = extract_video_id(url_to_use)
if not vid:
    st.error("Could not parse a valid YouTube video ID from the URL.")
    st.stop()

with st.spinner("Loading NLP models…"):
    stw, sentiment_pipe, summarizer, ner_pipe = get_pipes()

# Fetch transcript: prefer YouTubeTranscriptApi; fallback to yt-dlp .vtt
st.info("Fetching transcript…")
lines = []
lang = ""

# First: try YouTubeTranscriptApi for preferred languages
try:
    for pref in ("en","en-US","en-GB","fr","es"):
        try:
            tr = YouTubeTranscriptApi.get_transcript(vid, languages=[pref], cookies=None)
            if tr:
                lines, lang = tr, pref
                break
        except NoTranscriptFound:
            continue
        except Exception:
            time.sleep(0.5)
            continue
except TranscriptsDisabled:
    st.error("Transcripts are disabled by the uploader.")
    st.stop()
except CouldNotRetrieveTranscript:
    pass
except Exception:
    pass

# If API didn’t work, use yt-dlp captions (manual -> auto)
if not lines:
    st.info("Falling back to yt-dlp captions…")
    try:
        lines, lang = ytdlp_fetch_captions(url_to_use, prefer_langs=("en","en-US","en-GB","fr","es"))
    except Exception as e:
        lines, lang = [], ""
        st.warning(f"yt-dlp caption fallback failed: {e}")

if not lines:
    st.error("No subtitles available or access blocked. Try a different video or add cookies.txt if needed.")
    st.stop()

if lang and not str(lang).lower().startswith("en"):
    st.info(f"Translating subtitles from {lang} to English…")
    lines = translate_lines_safe(lines, lang, "en")

sub_df = to_dataframe(lines)
if sub_df.empty:
    st.error("Transcript is empty after parsing.")
    st.stop()

# Download media if needed
tempdir = Path(tempfile.mkdtemp(prefix="autoclipper_"))
wav_path = mp4_path = None
duration = float(sub_df["end"].max())

# keep variable and flow intact; UI toggle removed
make_mp4 = False
need_video = bool(include_face or make_mp4)

if include_audio or need_video:
    st.info("Downloading media via yt-dlp (audio/video)…")
    wav_path, mp4_path, dur_meta = download_media_ytdlp(url_to_use, tempdir, need_video=need_video)
    if isinstance(dur_meta, float) and dur_meta > 0:
        duration = max(duration, dur_meta)

# Audio energy series
audio_t = np.array([]); audio_r = np.array([])
if include_audio and (wav_path and wav_path.exists()):
    st.info("Analyzing audio energy…")
    audio_t, audio_r = audio_energy_series(wav_path)
    if audio_r.size:
        audio_r = (audio_r - audio_r.min()) / (audio_r.max() - audio_r.min() + 1e-8)

# Face emotion series (lazy TF/FER import)
face_t = np.array([]); face_s = np.array([])
if include_face:
    if not (mp4_path and mp4_path.exists()):
        st.warning("Face emotion requested, but low‑res MP4 was not available — skipping face analysis.")
    else:
        st.info("Analyzing face emotions…")
        face_t, face_s = face_emotion_series_lazy(mp4_path)
        if face_s.size:
            face_s = (face_s - face_s.min()) / (face_s.max() - face_s.min() + 1e-8)

# Build sliding windows
st.info("Scoring windows…")
rows = []
texts_for_batch = []
boundaries = []
for wstart, wend, chunk in window_iter(sub_df, win=float(win), stride=float(stride), total=duration):
    if chunk.empty:
        continue
    txt = " ".join(chunk["text"].tolist())
    rows.append({"start": wstart, "end": wend, "text": txt})
    texts_for_batch.append(txt)
    boundaries.append((wstart, wend))

if not rows:
    st.error("Not enough transcript content to score.")
    st.stop()

# Batch compute NLP features
features = score_features_bulk(texts_for_batch, stw, sentiment_pipe, ner_pipe)

# Audio/Face features aligned per window
audio_scores = []
face_scores  = []
for (wstart, wend) in boundaries:
    # audio
    asc = 0.0
    if audio_t.size:
        mask = (audio_t >= wstart) & (audio_t <= wend)
        if mask.any():
            asc = float(np.percentile(audio_r[mask], 90))
    audio_scores.append(asc)
    # face
    fsc = 0.0
    if face_t.size:
        maskf = (face_t >= wstart) & (face_t <= wend)
        if maskf.any():
            fsc = float(np.percentile(face_s[maskf], 90))
    face_scores.append(fsc)

windows_df = pd.DataFrame(rows)
windows_df["audio"] = audio_scores
windows_df["face"]  = face_scores
for k, v in features.items():
    windows_df[k] = v

windows_df = aggregate_scores(windows_df)
picks = pick_top_clips(windows_df, topk=int(num_clips), min_gap=MIN_GAP_SEC)

# Build outputs
st.markdown('<div class="card"><h3 class="section-title">Suggested Shorts</h3>', unsafe_allow_html=True)
out_rows = []
out_dir = Path("shorts_output")
out_dir.mkdir(exist_ok=True)

for i, (score, start, end) in enumerate(picks, 1):
    seg_text = " ".join(sub_df[(sub_df["start"]<end)&(sub_df["end"]>start)]["text"].tolist())
    title = title_for_clip(seg_text, summarizer)
    out_rows.append({
        "rank": i,
        "start_sec": round(start,2),
        "end_sec": round(end,2),
        "start_hhmmss": hhmmss(start),
        "end_hhmmss": hhmmss(end),
        "score": round(score,3),
        "title": title
    })

out_df = pd.DataFrame(out_rows)
st.dataframe(out_df, use_container_width=True)

# Save CSV + TXT, and show download buttons
csv_path = out_dir / "highlights.csv"
txt_path = out_dir / "highlights.txt"
out_df.to_csv(csv_path, index=False, encoding="utf-8")

with open(txt_path, "w", encoding="utf-8") as f:
    for r in out_rows:
        f.write(f"[{r['rank']}] {r['start_hhmmss']} → {r['end_hhmmss']} | {r['title']} (score {r['score']})\n")

st.success(f"Saved: {csv_path} and {txt_path}")

with open(csv_path, "rb") as f:
    st.download_button("Download CSV", data=f.read(), file_name="highlights.csv", mime="text/csv", use_container_width=True)
with open(txt_path, "rb") as f:
    st.download_button("Download TXT", data=f.read(), file_name="highlights.txt", mime="text/plain", use_container_width=True)

# Render MP4 clips (UI switch removed; variable kept for flow integrity)
if make_mp4:
    if not (mp4_path and mp4_path.exists()):
        st.error("Cannot render MP4: source MP4 not available. Disable face emotion or try again.")
    elif not has_ffmpeg():
        st.error("FFmpeg not found on PATH. Install it and retry.")
    else:
        st.info("Rendering MP4 clips with ffmpeg…")
        clips_dir = out_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        made = []
        prog = st.progress(0.0, text="Rendering…")
        for idx, r in enumerate(out_rows, 1):
            out_mp4 = clips_dir / f"clip_{r['rank']:02d}_{r['start_hhmmss']}_{r['end_hhmmss']}.mp4"
            try:
                clip_with_ffmpeg(mp4_path, out_mp4, r["start_sec"], r["end_sec"], vertical=vertical_crop)
                made.append(out_mp4)
            except Exception as e:
                st.warning(f"Clip {r['rank']} failed: {e}")
            prog.progress(idx/len(out_rows))
        prog.empty()
        if made:
            st.success(f"Generated {len(made)} MP4 files in {clips_dir}")
            for m in made:
                st.write(str(m))
                st.video(str(m))

st.markdown('</div>', unsafe_allow_html=True)
