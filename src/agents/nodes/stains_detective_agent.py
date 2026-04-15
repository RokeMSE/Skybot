"""
Stains Detective agent node.

Handles queries that require defect origin traceback across manufacturing
process images.  The node can operate in two modes:

1. Full traceback  — uploads_dir is known (state field, explicit path in query,
                     or resolved from a VID/lot number via local dirs + cloud share).
                     Runs the complete VLM pipeline and annotates the scratchpad
                     with the origin summary + panel images.

2. Alignment-only  — user asks to align / preprocess images without running
                     the VLM traceback.  Calls align_and_preprocess_images and
                     returns diagnostic overlay URLs.

Directory resolution priority (first hit wins):
  1. state["traceback_uploads_dir"]           — caller set it explicitly
  2. Explicit local/UNC path found in query   — e.g. C:\\data\\lot123
  3. VID/lot number → local output dir        — e.g. src/agents/stains_detective/src/output/{vid}
  4. VID/lot number → cloud share             — \\\\VNATSHFS…\\result\\{vid}
"""
import logging
import os
import re

from langchain_core.messages import AIMessage

from ...config import TRACEBACK_CLOUD_ROOT
from ..llm import get_chat_model
from ..state import AgentState
from ..tools import align_and_preprocess_images, run_defect_traceback

log = logging.getLogger(__name__)

# Matches real filesystem paths only — Windows drive (C:\...) or UNC (\\server\share)
# Intentionally excludes bare Unix /word patterns which appear in LLM-generated text
_PATH_RE = re.compile(r'(?:[A-Za-z]:[\\\/]|\\\\)[\w\\/\.\-]+')

# Matches a typical Intel fab lot/VID identifier: U6P22X1603318, U6UQ657500716
VID_RE = re.compile(r'\b([A-Z][A-Z0-9]{9,19})\b')

_DEFAULT_OUTPUT_DIR = os.path.join("static", "images")


def _isdir_safe(path: str) -> bool:
    """os.path.isdir that won't hang on unreachable UNC paths."""
    try:
        return os.path.isdir(path)
    except OSError:
        return False


def _extract_path(text: str) -> str | None:
    """Return the first plausible filesystem path found in *text*, or None."""
    m = _PATH_RE.search(text)
    return m.group(0).strip() if m else None


def extract_vid(text: str) -> str | None:
    """Return the first VID/lot-number token found in *text*, or None.

    Exported so the orchestrator can call it for routing decisions.
    """
    m = VID_RE.search(text)
    return m.group(1) if m else None


def find_uploads_dir(vid: str) -> str | None:
    """Resolve a VID to its directory of images + CSVs on the cloud share.

    Only checks TRACEBACK_CLOUD_ROOT — there is no local fallback.
    """
    if not TRACEBACK_CLOUD_ROOT:
        log.warning("TRACEBACK_CLOUD_ROOT is not configured.")
        return None
    cloud = os.path.join(TRACEBACK_CLOUD_ROOT, vid)
    if _isdir_safe(cloud):
        log.info("Resolved %s → cloud: %s", vid, cloud)
        return cloud
    log.warning("VID %s not found on cloud share: %s", vid, cloud)
    return None


def stains_detective_node(state: AgentState) -> dict:
    """Runs defect traceback or image alignment based on the current sub_query."""
    iteration = state.get("iteration", 0)
    sub_query = state.get("sub_query", "")
    user_query = state.get("user_query", "")
    combined_text = sub_query + " " + user_query

    # ------------------------------------------------------------------
    # 1. Resolve uploads directory
    # ------------------------------------------------------------------
    uploads_dir: str | None = state.get("traceback_uploads_dir")

    # Only extract explicit paths from the original user query — not from sub_query,
    # which is LLM-generated text and can contain false path-like fragments.
    if not uploads_dir:
        uploads_dir = _extract_path(user_query)

    if not uploads_dir:
        vid = extract_vid(combined_text)
        if vid:
            uploads_dir = find_uploads_dir(vid)
            if not uploads_dir:
                log.warning("VID %s not found on cloud share.", vid)

    output_dir: str = state.get("traceback_output_dir") or _DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Decide mode: alignment-only vs full traceback
    #
    # When a VID resolved the directory we always do full traceback —
    # alignment-only only makes sense when the user explicitly asks for it
    # on an explicit folder path without any traceback intent.
    # ------------------------------------------------------------------
    vid_in_query = extract_vid(user_query) is not None
    if vid_in_query or state.get("traceback_uploads_dir"):
        alignment_only = False
    else:
        lower = combined_text.lower()
        alignment_only = any(
            kw in lower for kw in ("align", "preprocess", "preprocessing", "overlay", "diagnostic")
        ) and not any(
            kw in lower for kw in ("traceback", "trace back", "origin", "defect origin", "csv")
        )

    # ------------------------------------------------------------------
    # 3. No directory found — ask the user
    # ------------------------------------------------------------------
    if not uploads_dir:
        model = get_chat_model(temperature=0)
        clarification = model.invoke([
            {
                "role": "system",
                "content": (
                    "You are a semiconductor defect analysis assistant. "
                    "The user wants to run a defect traceback but no lot/VID number was found. "
                    "Ask them concisely for the VID (e.g. U6P22X1603318) — "
                    "the system will look it up on the cloud share automatically."
                ),
            },
            {"role": "user", "content": user_query},
        ])
        note = f"[Stains Detective] Missing uploads directory — {clarification.content}"
        log.warning("Stains detective: no uploads_dir or VID found.")
        return {
            "scratchpad": (
                state.get("scratchpad", "")
                + f"\n\n--- [Stains Detective — Step {iteration}] ---\n{note}"
            ).strip(),
            "messages": [AIMessage(content=note)],
        }

    # ------------------------------------------------------------------
    # 4. Run the appropriate tool
    # ------------------------------------------------------------------
    if alignment_only:
        log.info("Stains detective: alignment-only mode on %s", uploads_dir)
        result = align_and_preprocess_images(uploads_dir, uploads_dir, output_dir)
        if result.get("error"):
            note = f"[Stains Detective] Alignment failed: {result['error']}"
        else:
            ok_count = sum(1 for v in result["aligned"].values() if v["ok"])
            total = len(result["aligned"])
            lines = [
                f"Alignment complete ({uploads_dir}): {ok_count}/{total} images aligned successfully."
            ]
            for fname, info in result["aligned"].items():
                status = "OK" if info["ok"] else "FAIL"
                lines.append(
                    f"  {fname}: {status} | inliers={info['inliers']} | "
                    f"reproj_p95={info['reproj_p95']} | method={info['method']}"
                )
            note = "\n".join(lines)
        new_image_urls = result.get("diagnostics", [])

    else:
        log.info("Stains detective: full traceback on %s", uploads_dir)
        result = run_defect_traceback(uploads_dir, output_dir)
        if result.get("error"):
            note = f"[Stains Detective] Traceback failed: {result['error']}"
        else:
            lines = [f"Defect traceback complete (source: {uploads_dir}).\n"]
            origins = result.get("origin_summary", {})
            if origins:
                lines.append("Origin summary:")
                for defect_id, origin in origins.items():
                    lines.append(f"  • {defect_id} → {origin}")
            else:
                lines.append("No defect origins identified.")
            report = result.get("report_text", "")
            if report:
                lines.append("\nDetailed report:\n" + report)
            note = "\n".join(lines)
        new_image_urls = result.get("output_images", [])

    # ------------------------------------------------------------------
    # 5. Update state
    # ------------------------------------------------------------------
    scratchpad = state.get("scratchpad", "")
    scratchpad = (
        scratchpad + f"\n\n--- [Stains Detective — Step {iteration}] ---\n{note}"
    ).strip()

    existing_urls: list = state.get("image_urls", [])
    combined_urls = existing_urls + [u for u in new_image_urls if u not in existing_urls]

    log.info("Stains detective complete. Images: %d", len(new_image_urls))

    msg = note[:300] + "…" if len(note) > 300 else note
    return {
        "scratchpad": scratchpad,
        "image_urls": combined_urls,
        "messages": [AIMessage(content=f"[Stains Detective — Step {iteration}] {msg}")],
    }
