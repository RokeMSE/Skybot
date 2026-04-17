"""
Shared retrieval helper used by all agent nodes.

Extracts and centralises the ChromaDB query + hybrid image retrieval
logic that previously lived inside RAGEngine.query(). Both the existing
RAGEngine and the new agent nodes can import from here.

Also exposes:
  - align_and_preprocess_images: aligns process images to an OG reference
  - run_defect_traceback: full VLM-assisted defect origin traceback pipeline
  - retrieve_lot_unit_info: fetches lot & unit XML from the network share
"""
import logging
import os
import re
from typing import Optional

from ..storage.vectordb import get_vector_db
from ..config import LOT_UNIT_DIR

log = logging.getLogger(__name__)


def retrieve_from_knowledge_base(
    query: str,
    channel: Optional[str] = None,
    n_results: int = 5,
) -> dict:
    """
    Queries ChromaDB and performs hybrid image retrieval for matching pages.

    Returns:
        {
            "context": str,        # formatted text for LLM prompt
            "docs":    list[dict], # raw {text, metadata} pairs
            "images":  list[str],  # /static/images/<filename> URLs
            "citations": list[dict]
        }
    """
    collection = get_vector_db()

    query_kwargs: dict = {"query_texts": [query], "n_results": n_results}
    if channel:
        query_kwargs["where"] = {"channel": channel}

    results = collection.query(**query_kwargs)

    context_parts: list[str] = []
    docs: list[dict] = []
    image_urls: list[str] = []
    citations: list[dict] = []
    seen_images: set[str] = set()
    schema_pages: set[tuple] = set()

    if results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            source_tag = f"[Source: {meta.get('source', 'Unknown')}, Page {meta.get('page', '?')}]"
            context_parts.append(f"\n--- {source_tag} ---\n{doc}")
            docs.append({"text": doc, "metadata": meta})
            citations.append(meta)

            # Track text-bearing pages for hybrid image lookup
            if meta.get("type") == "text":
                src = meta.get("source")
                page = meta.get("page")
                if src and page:
                    schema_pages.add((src, page))

            # Directly retrieved image chunks
            if meta.get("type") == "image_cad" and meta.get("image_path"):
                img_url = f"/static/images/{os.path.basename(meta['image_path'])}"
                if img_url not in seen_images:
                    image_urls.append(img_url)
                    seen_images.add(img_url)

    # Hybrid retrieval — fetch images from the same pages as text matches
    for source, page in schema_pages:
        try:
            img_results = collection.get(
                where={
                    "$and": [
                        {"type": "image_cad"},
                        {"source": source},
                        {"page": page},
                    ]
                }
            )
            for meta in (img_results.get("metadatas") or []):
                if meta.get("image_path"):
                    img_url = f"/static/images/{os.path.basename(meta['image_path'])}"
                    if img_url not in seen_images:
                        image_urls.insert(0, img_url)
                        seen_images.add(img_url)
        except Exception as e:
            print(f"Hybrid retrieval error for {source} p{page}: {e}")

    return {
        "context": "\n".join(context_parts),
        "docs": docs,
        "images": image_urls,
        "citations": citations,
    }


def get_image_url_from_metadata(meta: dict) -> Optional[str]:
    if meta.get("type") == "image_cad" and meta.get("image_path"):
        return f"/static/images/{os.path.basename(meta['image_path'])}"
    return None


# ---------------------------------------------------------------------------
# Stains Detective tools
# ---------------------------------------------------------------------------

def _imread_safe(path: str):
    """cv2.imread that works with UNC paths (\\\\server\\share) on Windows.

    cv2's C backend cannot open UNC paths directly; reading via Python's
    file API and decoding with imdecode works around this limitation.
    """
    try:
        import cv2          # type: ignore[import-untyped]
        import numpy as np  # type: ignore[import-untyped]
        with open(path, "rb") as fh:
            data = np.frombuffer(fh.read(), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def align_and_preprocess_images(
    og_image_path: str,
    process_images_dir: str,
    output_dir: str,
) -> dict:
    """Align all process images in *process_images_dir* to an OG reference
    image and apply contrast normalisation.

    Wraps AxisAligner + the preprocessing helpers from stains_detective without
    invoking the VLM or the PySide6 GUI.

    Args:
        og_image_path:       Absolute path to the OG/reference image.
        process_images_dir:  Directory that contains the process step images.
        output_dir:          Where to write diagnostic overlay images.

    Returns:
        {
            "aligned":   {filename: {"ok": bool, "inliers": int,
                                     "reproj_p95": float, "method": str}},
            "diagnostics": ["/static/images/<filename>"],  # overlay URLs
            "error":     str or None
        }
    """
    try:
        import cv2  # type: ignore[import-untyped]  # needed for draw_diagnostics + warpAffine
        from .stains_detective.src.alignment_validation import AxisAligner, draw_diagnostics
        from .stains_detective.src.defect_traceback_vlm import (
            normalize_contrast,
            auto_match_og_to_process,
        )
    except ImportError as exc:
        return {"aligned": {}, "diagnostics": [], "error": f"Import error: {exc}"}
    _ = cv2  # suppress unused-import hint — cv2 is used by draw_diagnostics internally

    os.makedirs(output_dir, exist_ok=True)

    og_bgr = _imread_safe(og_image_path)
    if og_bgr is None:
        return {"aligned": {}, "diagnostics": [], "error": f"Cannot read OG image: {og_image_path}"}

    proc_pattern = re.compile(r"(?:\w+_)?\d+_(In|Out)\.(jpg|jpeg|png)$", re.IGNORECASE)
    proc_files = [
        f for f in sorted(os.listdir(process_images_dir))
        if proc_pattern.match(f)
    ]
    if not proc_files:
        return {"aligned": {}, "diagnostics": [], "error": "No process images found (expected *_In.jpg / *_Out.jpg pattern)."}

    proc_imgs = {}
    for fname in proc_files:
        img = _imread_safe(os.path.join(process_images_dir, fname))
        if img is not None:
            proc_imgs[fname] = img

    # Preprocess: normalise contrast on OG to match process polarity
    proc_sample = next(iter(proc_imgs.values()))
    og_norm, og_was_inverted = auto_match_og_to_process(og_bgr, proc_sample)
    log.info("OG contrast inversion detected: %s", og_was_inverted)

    aligner = AxisAligner()
    aligned_info = {}
    diagnostic_urls = []

    for fname, proc_bgr in proc_imgs.items():
        proc_norm = normalize_contrast(proc_bgr, clip_limit=3.0)
        affine = aligner.align(og_bgr, proc_bgr)

        aligned_info[fname] = {
            "ok": affine.ok,
            "inliers": affine.inliers,
            "reproj_p95": round(affine.reproj_p95, 2),
            "method": affine.method,
        }

        # Write diagnostic overlays to output_dir; checker path is returned directly
        if affine.ok:
            diag_label = os.path.splitext(fname)[0]
            diag_paths = draw_diagnostics(og_norm, proc_norm, affine, output_dir, label=diag_label)
            checker_path = diag_paths.get("checker")
            if checker_path and os.path.isfile(checker_path):
                diagnostic_urls.append(f"/static/images/{os.path.basename(checker_path)}")

        log.info(
            "Aligned %s — ok=%s inliers=%d reproj_p95=%.1f method=%s",
            fname, affine.ok, affine.inliers, affine.reproj_p95, affine.method,
        )

    return {"aligned": aligned_info, "diagnostics": diagnostic_urls, "error": None}


class _LangChainVLM:
    """VLMService adapter that reuses the main project's LangChain client.

    Avoids creating a separate OpenAI SDK client with different SSL/proxy
    settings — the LangChain client is already known to reach the endpoint.
    """

    def analyze_images(self, images: list, prompt: str) -> str:
        import base64
        import io as _io
        from langchain_core.messages import HumanMessage
        from .llm import get_chat_model

        content: list = [{"type": "text", "text": prompt}]
        for img in images:
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })

        try:
            llm = get_chat_model(temperature=0.2)
            response = llm.invoke([HumanMessage(content=content)])
            return str(response.content)
        except Exception as exc:
            log.error("LangChainVLM error: %s", exc)
            return f"ERROR: {exc}"


def run_defect_traceback(
    uploads_dir: str,
    output_dir: str,
    ref_image_key: Optional[str] = None,
) -> dict:
    """Run the full VLM-assisted defect origin traceback pipeline.

    Reads images and *DVI_box_data.csv* from *uploads_dir*, aligns all process
    images to the OG reference, then uses the configured VLM (Azure OpenAI /
    OpenAI) to determine for each defect which process step introduced it.

    The PySide6 GUI layer is intentionally excluded — this function is the
    pure library entry-point.

    Args:
        uploads_dir:    Directory with OG images, process images, DVI_box_data.csv,
                        and optionally vid_data.csv.
        output_dir:     Where traceback panels and the text report are written.
        ref_image_key:  Filename of the OG image to use as anchor (auto-selects
                        FRAME2 when None).

    Returns:
        {
            "report_text":    str,           # full text summary
            "output_images":  list[str],     # /static/images/<fname> URLs
            "origin_summary": dict,          # defect_id → origin step
            "error":          str or None
        }
    """
    try:
        from .stains_detective.src.defect_traceback_vlm import run_traceback
    except ImportError as exc:
        return {"report_text": "", "output_images": [], "origin_summary": {}, "error": f"Import error: {exc}"}

    result = run_traceback(
        uploads_dir=uploads_dir,
        outdir=output_dir,
        ref_image_key=ref_image_key,
        vlm=_LangChainVLM(),
    )

    if result.get("error"):
        return {"report_text": "", "output_images": [], "origin_summary": {}, "error": result["error"]}

    # Build URL list for output images.
    # run_traceback already writes panels into output_dir; if that is already
    # static/images we just build URLs — no copy needed (and copying a file
    # onto itself causes a PermissionError on Windows).
    import shutil
    static_img_dir = os.path.abspath(os.path.join("static", "images"))
    os.makedirs(static_img_dir, exist_ok=True)

    # Only expose PANEL_ images to the agent — individual TB_*/OG_*/ZONE_* crops
    # are internal diagnostics and not needed in the chat report.
    image_urls = []
    for img_path in result.get("output_images", []):
        if not os.path.isfile(img_path):
            continue
        fname = os.path.basename(img_path)
        if not fname.startswith("PANEL_"):
            continue
        dest_path = os.path.join(static_img_dir, fname)
        if os.path.abspath(img_path) != dest_path:
            shutil.copy2(img_path, dest_path)
        image_urls.append(f"/static/images/{fname}")

    origin_summary = {
        str(entry[0].dr_sub_item): entry[2]
        for entry in result.get("all_results", [])
    }

    return {
        "report_text": result.get("report_text", ""),
        "output_images": image_urls,
        "origin_summary": origin_summary,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Aries Oracle DB — unit test results
# ---------------------------------------------------------------------------

def query_unit_test_aries(
    lot: Optional[str] = None,
    operation: Optional[str] = None,
    tester_id: Optional[str] = None,
    visual_id: Optional[str] = None,
    interface_bin: Optional[int] = None,
    row_limit: int = 5000,
    test_start_date_time: Optional[str] = None,
    device_start_date_time: Optional[str] = None,
    device_end_date_time: Optional[str] = None,
) -> dict:
    """Query unit test data from Aries Oracle database.

    Returns:
        {
            "records": list[dict],   # row dicts from the query
            "count":   int,
            "error":   str or None,
        }
    """
    try:
        from ..tools.unit_test_arias_tool import UnitTestAriasTool
        from ..tools.base_tool import ToolConfig
    except ImportError as e:
        return {"records": [], "count": 0, "error": f"Aries DB unavailable: {e}"}

    tool = UnitTestAriasTool(config=ToolConfig(
        name="unit_test_aries",
        description="Query Aries unit test data",
        timeout_seconds=180,
    ))

    params: dict = {"row_limit": row_limit}
    if lot:
        params["lot"] = lot
    if operation:
        params["operation"] = operation
    if tester_id:
        params["tester_id"] = tester_id
    if visual_id:
        params["visual_id"] = visual_id
    if interface_bin is not None:
        params["interface_bin"] = interface_bin
    if test_start_date_time:
        params["test_start_date_time"] = test_start_date_time
    if device_start_date_time:
        params["device_start_date_time"] = device_start_date_time
    if device_end_date_time:
        params["device_end_date_time"] = device_end_date_time

    try:
        result = tool.execute(params)
        records = result.get("data", [])
        return {"records": records, "count": len(records), "error": None}
    except Exception as e:
        return {"records": [], "count": 0, "error": f"Aries query failed: {e}"}


# ---------------------------------------------------------------------------
# Lot & Unit info tools  (XML from network share)
# ---------------------------------------------------------------------------

def retrieve_lot_unit_info(
    lot_id: str,
    operation: str,
    base_dir: Optional[str] = None,
) -> dict:
    """Look up lotinfo.xml and unitinfo.xml for a given lot+operation.

    Searches ``{base_dir}/{lot_id}_{operation}/`` for the XML files and
    returns compact, token-efficient summaries ready for LLM prompts.

    Returns:
        {
            "lot_summary":  str,   # compact lot info (or error message)
            "unit_summary": str,   # compact unit info (or error message)
            "error":        str or None,
        }
    """
    from ..extractors.xml import XMLExtractor
    from ..tools.lot_info_tool import format_lot_info
    from ..tools.unit_info_tool import format_unit_info

    search_dir = base_dir or LOT_UNIT_DIR
    lot_dir = os.path.join(search_dir, f"{lot_id}_{operation}")

    if not os.path.isdir(lot_dir):
        return {
            "lot_summary": "",
            "unit_summary": "",
            "error": f"Directory not found: {lot_dir}",
        }

    extractor = XMLExtractor()
    lot_summary = ""
    unit_summary = ""
    tester_id = None
    errors = []

    lot_path = os.path.join(lot_dir, "lotinfo.xml")
    if os.path.isfile(lot_path):
        try:
            lot_dict = extractor.xml_to_dict(lot_path)
            lot_summary = format_lot_info(lot_dict)
            # Extract tester ID (SysId) from the parsed XML
            _lot = lot_dict.get("LotInfo", {}).get("LotInfo", {})
            if not _lot:
                _lot = lot_dict.get("LotInfo", {})
            sys_id = _lot.get("SysId", {})
            if isinstance(sys_id, dict):
                tester_id = sys_id.get("#text")
            elif isinstance(sys_id, str) and sys_id:
                tester_id = sys_id
        except Exception as e:
            errors.append(f"lotinfo.xml parse error: {e}")
    else:
        errors.append(f"lotinfo.xml not found in {lot_dir}")

    unit_path = os.path.join(lot_dir, "unitinfo.xml")
    if os.path.isfile(unit_path):
        try:
            unit_dict = extractor.xml_to_dict(unit_path)
            unit_summary = format_unit_info(unit_dict)
        except Exception as e:
            errors.append(f"unitinfo.xml parse error: {e}")
    else:
        errors.append(f"unitinfo.xml not found in {lot_dir}")

    return {
        "lot_summary": lot_summary,
        "unit_summary": unit_summary,
        "tester_id": tester_id,
        "error": "; ".join(errors) if errors else None,
    }


def _list_recent_lots_aries(n: int = 10) -> dict:
    """Query Aries Oracle for recently tested lots (fallback when network share is unavailable)."""
    try:
        import oracledb as cx
        import pandas as pd
    except ImportError as e:
        return {"lots": [], "error": f"Aries DB packages unavailable: {e}"}

    from ..config import ARIES_DB_USER, ARIES_DB_PASSWORD, ARIES_DB_DSN

    if not ARIES_DB_USER or not ARIES_DB_PASSWORD:
        return {"lots": [], "error": "Aries DB credentials not configured"}

    sql = f"""
    SELECT * FROM (
        SELECT v0.lot,
               v0.operation,
               v0.tester_id,
               TO_CHAR(MAX(v0.test_end_date_time), 'YYYY-MM-DD HH24:MI') AS latest_test,
               COUNT(*) AS session_count
        FROM A_Testing_Session v0
        WHERE v0.test_end_date_time >= SYSDATE - 1
        AND (v0.operation LIKE '6%' OR v0.operation LIKE '7%')
        AND v0.tester_id LIKE '%HXV%'
        GROUP BY v0.lot, v0.operation, v0.tester_id
        ORDER BY MAX(v0.test_end_date_time) DESC
    )
    WHERE ROWNUM <= {int(n)}
    """

    conn = None
    try:
        conn = cx.connect(user=ARIES_DB_USER, password=ARIES_DB_PASSWORD, dsn=ARIES_DB_DSN)
        df = pd.read_sql(sql, conn)
        lots = []
        for _, row in df.iterrows():
            lots.append({
                "folder": None,
                "lot_id": str(row["LOT"]),
                "operation": str(row["OPERATION"]),
                "modified": str(row["LATEST_TEST"]),
                "summary": f"{row['SESSION_COUNT']} test sessions on tester {row['TESTER_ID']}",
                "tester_id": str(row["TESTER_ID"]),
            })
        return {"lots": lots, "error": None}
    except Exception as e:
        return {"lots": [], "error": f"Aries query failed: {e}"}
    finally:
        if conn is not None:
            conn.close()


def list_recent_lots(
    n: int = 10,
    base_dir: Optional[str] = None,
) -> dict:
    """List the most recently modified lot folders and parse their summaries.

    Tries the network share first (``LOT_UNIT_DIR``). If that is unavailable,
    falls back to querying Aries Oracle for recently tested lots.

    Returns:
        {
            "lots": [
                {"folder": str, "lot_id": str, "operation": str,
                 "modified": str, "summary": str, "tester_id": str|None},
                ...
            ],
            "error": str or None,
        }
    """
    from datetime import datetime
    from ..extractors.xml import XMLExtractor
    from ..tools.lot_info_tool import format_lot_info

    search_dir = base_dir or LOT_UNIT_DIR
    if not os.path.isdir(search_dir):
        log.info("Network share unavailable (%s), falling back to Aries Oracle", search_dir)
        return _list_recent_lots_aries(n)

    # Pattern: {lotID}_{operation}  (e.g. 4V56656R_5274)
    folder_re = re.compile(r"^([A-Z0-9]{2}[A-Z0-9]{5,7})_(\d{3,5})$", re.IGNORECASE)

    entries = []
    try:
        for name in os.listdir(search_dir):
            m = folder_re.match(name)
            if not m:
                continue
            full_path = os.path.join(search_dir, name)
            if not os.path.isdir(full_path):
                continue
            mtime = os.path.getmtime(full_path)
            entries.append((name, m.group(1).upper(), m.group(2), mtime, full_path))
    except OSError as e:
        return {"lots": [], "error": f"Cannot list directory: {e}"}

    # Sort by modification time, newest first
    entries.sort(key=lambda x: x[3], reverse=True)
    entries = entries[:n]

    extractor = XMLExtractor()
    lots = []
    for folder_name, lot_id, operation, mtime, full_path in entries:
        modified_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        summary = ""
        tester_id = None

        lot_path = os.path.join(full_path, "lotinfo.xml")
        if os.path.isfile(lot_path):
            try:
                lot_dict = extractor.xml_to_dict(lot_path)
                summary = format_lot_info(lot_dict)
                # Extract tester ID
                _lot = lot_dict.get("LotInfo", {}).get("LotInfo", {})
                if not _lot:
                    _lot = lot_dict.get("LotInfo", {})
                sys_id = _lot.get("SysId", {})
                if isinstance(sys_id, dict):
                    tester_id = sys_id.get("#text")
                elif isinstance(sys_id, str) and sys_id:
                    tester_id = sys_id
            except Exception as e:
                summary = f"(parse error: {e})"

        lots.append({
            "folder": folder_name,
            "lot_id": lot_id,
            "operation": operation,
            "modified": modified_str,
            "summary": summary,
            "tester_id": tester_id,
        })

    return {"lots": lots, "error": None}
