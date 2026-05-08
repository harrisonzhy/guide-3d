"""
GUIDE-3D unified ranking UI with embedded 3DGS viewer.

Run with:
    streamlit run app.py

Discovers scenes by intersecting:
    - gaussian-splatting/models/<scene>/   (trained 3DGS models, must contain
                                            point_cloud/iteration_30000/point_cloud.ply
                                            and cameras.json)
    - metrics/<scene>.pkl                  (pre-computed per-view metrics)

Each metrics pickle is a pandas DataFrame with columns:
    image_name, object, L1, SSIM, PSNR_dB, total_pixels
"""

from __future__ import annotations

import http.server
import json
import re
import socket
import socketserver
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).parent.resolve()
METRICS_DIR = ROOT / "metrics"
MODELS_DIR = ROOT / "gaussian-splatting" / "models"

LINEUP_CSS_URL = "https://unpkg.com/lineupjs/build/LineUpJS.css"
LINEUP_JS_URL = "https://unpkg.com/lineupjs/build/LineUpJS.js"
GS3D_VERSION = "0.4.7"
GS3D_MODULE_URL = (
    f"https://cdn.jsdelivr.net/npm/@mkkellogg/gaussian-splats-3d@{GS3D_VERSION}"
    "/build/gaussian-splats-3d.module.js"
)
THREE_VERSION = "0.160.0"
THREE_MODULE_URL = f"https://cdn.jsdelivr.net/npm/three@{THREE_VERSION}/build/three.module.js"

st.set_page_config(
    page_title="GUIDE-3D Object Quality Ranking",
    layout="wide",
)


# ---------- file server ----------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _RootHandler(http.server.SimpleHTTPRequestHandler):
    """Serves files from the workspace root with CORS enabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=86400")
        super().end_headers()

    def log_message(self, *_args, **_kwargs):
        pass


@st.cache_resource(show_spinner=False)
def start_static_server() -> str:
    """Spin up a daemonized HTTP server serving ROOT. Returns base URL."""
    port = _free_port()
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), _RootHandler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, name="static-server", daemon=True)
    thread.start()
    return f"http://127.0.0.1:{port}"


# ---------- data loading ----------


def discover_scenes() -> list[str]:
    if not METRICS_DIR.exists() or not MODELS_DIR.exists():
        return []
    pkl_scenes = {p.stem for p in METRICS_DIR.glob("*.pkl")}
    model_scenes = {
        d.name
        for d in MODELS_DIR.iterdir()
        if d.is_dir() and (d / "cameras.json").exists()
    }
    return sorted(pkl_scenes & model_scenes)


@st.cache_data(show_spinner=False)
def load_metrics(scene: str) -> pd.DataFrame:
    """Load per-view per-object metrics. Tolerates absent optional columns
    (LPIPS, L1, depth_err) and de-duplicates accidentally repeated rows."""
    df = pd.read_pickle(METRICS_DIR / f"{scene}.pkl")
    rename = {
        "image_name": "view",
        "PSNR_dB": "psnr",
        "SSIM": "ssim",
        "LPIPS": "lpips",
        "L1": "l1",
        "depth_err": "depth_err",
        "total_pixels": "pixel_count",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df = df.drop_duplicates(subset=["object", "view"]).reset_index(drop=True)
    canonical = ["object", "view", "psnr", "ssim", "lpips", "l1", "depth_err", "pixel_count"]
    return df[[c for c in canonical if c in df.columns]]


@st.cache_data(show_spinner=False)
def load_cameras(scene: str) -> dict:
    """Read cameras.json and convert poses to Three.js conventions.

    cameras.json from gaussian-splatting:
      - position: camera world position (xyz)
      - rotation: 3x3 camera-to-world rotation. The camera frame uses the
        OpenCV convention (X right, Y down, Z forward).

    Three.js wants:
      - position (world)
      - lookAt (world point one unit in front of the camera)
      - up (world vector)

    Forward in camera frame (OpenCV) = (0, 0, 1)  -> world: R @ (0, 0, 1)
    Up in camera frame              = (0,-1, 0)  -> world: R @ (0,-1, 0)
    """
    path = MODELS_DIR / scene / "cameras.json"
    raw = json.loads(path.read_text())
    out = {}
    for cam in raw:
        R = np.asarray(cam["rotation"], dtype=float)
        pos = np.asarray(cam["position"], dtype=float)
        forward_world = R @ np.array([0.0, 0.0, 1.0])
        up_world = R @ np.array([0.0, -1.0, 0.0])
        look_at = pos + forward_world

        # Convert FoV (vertical) for Three.js PerspectiveCamera (in degrees).
        fy = float(cam["fy"])
        height = float(cam["height"])
        fov_y_deg = float(np.degrees(2.0 * np.arctan(0.5 * height / fy)))

        img = cam["img_name"]
        out[img] = {
            "position": pos.tolist(),
            "lookAt": look_at.tolist(),
            "up": up_world.tolist(),
            "fovY": fov_y_deg,
        }
    return out


def find_ply(scene: str) -> Path | None:
    base = MODELS_DIR / scene / "point_cloud"
    if not base.exists():
        return None
    candidates = sorted(
        (p for p in base.glob("iteration_*/point_cloud.ply") if p.exists()),
        key=lambda p: int(p.parent.name.split("_")[1]),
        reverse=True,
    )
    return candidates[0] if candidates else None


# Candidate roots holding the original (ground-truth) capture images per scene.
# The exact resolution subfolder (e.g. images_4) is taken from the trained
# model's cfg_args so the GT matches what the metrics were computed against.
_GT_DATA_ROOTS = [
    ROOT / "gaussian-splatting" / "data" / "images1",
    ROOT / "gaussian-splatting" / "data" / "images2",
]
_GT_IMG_EXTS = (".JPG", ".jpg", ".jpeg", ".JPEG", ".png", ".PNG")


def _gt_subdir_from_cfg(scene: str) -> str:
    """Read the trained-model cfg_args to find which images_* subfolder
    was used at training time. Falls back to plain 'images' if missing."""
    cfg = MODELS_DIR / scene / "cfg_args"
    if cfg.exists():
        m = re.search(r"images=['\"]([^'\"]+)['\"]", cfg.read_text())
        if m:
            return m.group(1)
    return "images"


def find_gt_dir(scene: str) -> Path | None:
    subdir = _gt_subdir_from_cfg(scene)
    for root in _GT_DATA_ROOTS:
        p = root / scene / subdir
        if p.is_dir():
            return p
    return None


def build_gt_urls(scene: str, cameras: dict, base_url: str) -> dict[str, str]:
    """Map view name -> URL of the matching ground-truth image. Empty if
    GT images aren't present for this scene."""
    gt_dir = find_gt_dir(scene)
    if gt_dir is None:
        return {}
    rel = gt_dir.relative_to(ROOT).as_posix()
    out: dict[str, str] = {}
    for view in cameras:
        for ext in _GT_IMG_EXTS:
            if (gt_dir / f"{view}{ext}").exists():
                out[view] = f"{base_url}/{rel}/{view}{ext}"
                break
    return out


# ---------- aggregation ----------


def _angular_coverage(views: list[str], cameras: dict) -> float:
    """Directional dispersion of view directions in [0, 1].
    0 = all views look in the same direction, 1 = perfectly dispersed."""
    dirs = []
    for v in views:
        cam = cameras.get(v) or cameras.get(v.split(".")[0])
        if cam is None:
            continue
        d = np.asarray(cam["lookAt"]) - np.asarray(cam["position"])
        n = np.linalg.norm(d)
        if n > 0:
            dirs.append(d / n)
    if not dirs:
        return 0.0
    arr = np.stack(dirs)
    return float(1.0 - np.linalg.norm(arr.mean(axis=0)))


def aggregate_objects(df: pd.DataFrame, cameras: dict, total_cams: int) -> pd.DataFrame:
    """Per-object summary derived from the per-view table.

    Minimal, non-redundant column set (see correlation analysis):
        PSNR (mean)       - photometric quality, primary
        PSNR (worst)      - reconstruction-failure signal
        LPIPS (mean)      - perceptual quality (decorrelated from PSNR)
        depth err (mean)  - geometric quality (independent axis)
        visibility        - fraction of cameras that saw the object
        angular cov.      - diversity of viewing directions
        pixels (mean)     - object size context

    Columns only appear if the underlying metric is present in `df`."""
    has = {col: col in df.columns for col in ["psnr", "lpips", "depth_err"]}

    if df.empty:
        return pd.DataFrame(columns=["object", "visibility"])

    rows = []
    for obj, grp in df.groupby("object"):
        n = len(grp)
        row: dict = {"object": obj}

        if has["psnr"]:
            psnr = grp["psnr"]
            row["PSNR (mean)"] = round(float(psnr.mean()), 2)
            row["PSNR (worst)"] = round(float(psnr.min()), 2)
        if has["lpips"]:
            row["LPIPS (mean)"] = round(float(grp["lpips"].mean()), 4)
        if has["depth_err"]:
            row["depth err (mean)"] = round(float(grp["depth_err"].mean()), 4)

        row["visibility"] = round(n / total_cams, 3) if total_cams else 0.0
        row["angular cov."] = round(
            _angular_coverage(grp["view"].tolist(), cameras), 3
        )
        row["pixels (mean)"] = int(grp["pixel_count"].mean())
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_col = "PSNR (mean)" if "PSNR (mean)" in out.columns else "visibility"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


def per_view_display(df_obj: pd.DataFrame) -> pd.DataFrame:
    """Per-view table for one object, with friendly column names."""
    rename = {
        "view": "view",
        "psnr": "PSNR",
        "ssim": "SSIM",
        "lpips": "LPIPS",
        "l1": "L1",
        "depth_err": "depth err",
        "pixel_count": "pixels",
    }
    cols = [c for c in rename if c in df_obj.columns]
    out = df_obj[cols].rename(columns=rename).copy()
    sort_col = "PSNR" if "PSNR" in out.columns else "view"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


# ---------- HTML helpers ----------


def lineup_html(df: pd.DataFrame, height_px: int) -> str:
    records = json.loads(df.to_json(orient="records"))
    payload = json.dumps(records)
    inner = max(height_px - 8, 80)
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8" />
<link href="{LINEUP_CSS_URL}" rel="stylesheet" />
<script src="{LINEUP_JS_URL}"></script>
<style>html,body{{margin:0;padding:0;height:100%;background:transparent;}}
#lineup{{width:100%;height:{inner}px;}}</style></head>
<body><div id="lineup"></div>
<script>LineUpJS.asLineUp(document.getElementById('lineup'), {payload});</script>
</body></html>"""


def viewer_html(
    ply_url: str,
    cameras: dict,
    object_views: dict[str, list[str]],
    default_object: str | None,
    scene: str,
    height_px: int,
    gt_urls: dict[str, str] | None = None,
) -> str:
    """Build the 3DGS viewer iframe HTML.

    Inputs are scene-scoped only (no filter state) so that the rendered HTML
    is byte-stable across Streamlit reruns within a scene. That keeps the
    iframe mounted and avoids re-downloading the multi-hundred-MB .ply on
    every filter change. Filter-driven focus is delivered via a sibling
    'focus bridge' iframe that posts cross-origin messages.

    Args:
        ply_url: URL the iframe will fetch the .ply from.
        cameras: dict of img_name -> {position, lookAt, up, fovY}.
        object_views: full object_name -> ordered list of img_names (best→worst by PSNR).
        default_object: object to focus on at very first paint, before any
            bridge message arrives.
        scene: used as a BroadcastChannel + localStorage namespace so that
            switching scenes doesn't cross-contaminate focus state.
    """
    initial_views = object_views.get(default_object, []) if default_object else []
    initial_view = initial_views[0] if initial_views else next(iter(cameras), None)
    if initial_view is None:
        return "<div style='padding:20px;color:#c00'>No cameras available.</div>"

    cam_payload = json.dumps(cameras)
    obj_payload = json.dumps(object_views)
    initial_view_js = json.dumps(initial_view)
    initial_obj_js = json.dumps(default_object)
    gt_payload = json.dumps(gt_urls or {})
    channel_id = json.dumps(f"guide3d:{scene}")
    storage_key = json.dumps(f"guide3d:focus:{scene}")
    body_h = max(height_px - 60, 200)

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8" />
<style>
  html, body {{
    margin: 0; padding: 0; background: #0d0d10; color: #ddd;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  }}
  #toolbar {{
    display: flex; align-items: center; gap: 8px;
    padding: 6px 10px; background: #16161b; border-bottom: 1px solid #2a2a32;
    font-size: 13px; height: 44px; box-sizing: border-box;
  }}
  #toolbar label {{ color: #a8a8b3; }}
  #toolbar select, #toolbar button {{
    background: #232329; color: #eee; border: 1px solid #3a3a44;
    border-radius: 4px; padding: 4px 8px; font-size: 12px;
    cursor: pointer;
  }}
  #toolbar select {{ min-width: 170px; }}
  #toolbar button:hover {{ background: #2c2c34; }}
  #toolbar .spacer {{ flex: 1; }}
  #status {{ color: #888; font-size: 11px; }}
  #stage {{
    display: flex; flex-direction: row; width: 100%; height: {body_h}px;
    background: #000;
  }}
  #viewerHost {{
    position: relative; flex: 1 1 auto; min-width: 0; height: 100%;
    background: #000; overflow: hidden;
  }}
  #gtPanel {{
    flex: 0 0 38%; max-width: 50%; min-width: 240px; height: 100%;
    background: #0a0a0d; border-left: 1px solid #2a2a32;
    display: flex; flex-direction: column;
    transition: flex-basis 0.18s ease, min-width 0.18s ease,
                border-left-color 0.18s ease;
    overflow: hidden;
  }}
  #gtPanel.collapsed {{
    flex-basis: 0px; min-width: 0; border-left-color: transparent;
  }}
  #gtHeader {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 4px 8px; background: #16161b; border-bottom: 1px solid #2a2a32;
    font-size: 11px; color: #a8a8b3; flex: 0 0 auto;
  }}
  #gtHeader .gtTitle {{ color: #ddd; font-weight: 600; }}
  #gtHeader .gtView {{ color: #89c0ff; font-family: ui-monospace, monospace; }}
  #gtImgWrap {{
    flex: 1 1 auto; display: flex; align-items: center; justify-content: center;
    overflow: hidden; padding: 6px; box-sizing: border-box;
  }}
  #gtImg {{
    max-width: 100%; max-height: 100%; object-fit: contain;
    background: #000; display: block;
  }}
  #gtEmpty {{
    color: #777; font-size: 12px; padding: 12px; text-align: center;
  }}
  #gtBtn[aria-pressed="true"] {{ background: #2a3a55; border-color: #4d6fa3; }}
</style>
<script type="importmap">
{{
  "imports": {{
    "three": "{THREE_MODULE_URL}",
    "@mkkellogg/gaussian-splats-3d": "{GS3D_MODULE_URL}"
  }}
}}
</script>
</head>
<body>
  <div id="toolbar">
    <label for="viewSel">Camera view</label>
    <select id="viewSel"></select>
    <button id="bestBtn" title="Jump to highest-PSNR view for the focused object">Best view</button>
    <button id="worstBtn" title="Jump to lowest-PSNR view for the focused object">Worst view</button>
    <button id="gtBtn" aria-pressed="false" title="Show ground-truth photo for the selected camera view">Ground truth</button>
    <span id="focusedObj" style="margin-left:12px;color:#89c0ff;font-size:12px;"></span>
    <span class="spacer"></span>
    <span id="status">loading splats…</span>
  </div>
  <div id="stage">
    <div id="viewerHost"></div>
    <aside id="gtPanel" class="collapsed" aria-hidden="true">
      <div id="gtHeader">
        <span class="gtTitle">Ground truth</span>
        <span class="gtView" id="gtView"></span>
      </div>
      <div id="gtImgWrap">
        <img id="gtImg" alt="Ground-truth photo for the selected camera view" />
        <div id="gtEmpty" style="display:none">No ground-truth image available for this view.</div>
      </div>
    </aside>
  </div>

  <script type="module">
    import * as THREE from 'three';
    import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

    const cameras = {cam_payload};
    const objectViews = {obj_payload};
    const gtUrls = {gt_payload};
    const initialView = {initial_view_js};
    let currentObject = {initial_obj_js};

    const viewNames = Object.keys(cameras);
    const viewSel = document.getElementById('viewSel');
    for (const v of viewNames) {{
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = v;
      if (v === initialView) opt.selected = true;
      viewSel.appendChild(opt);
    }}

    const status = document.getElementById('status');
    const host = document.getElementById('viewerHost');
    const initial = cameras[initialView] || cameras[viewNames[0]];

    const viewer = new GaussianSplats3D.Viewer({{
      rootElement: host,
      cameraUp: initial.up,
      initialCameraPosition: initial.position,
      initialCameraLookAt: initial.lookAt,
      sharedMemoryForWorkers: false,
      useBuiltInControls: true,
      sphericalHarmonicsDegree: 1,
    }});

    function applyCamera(viewName) {{
      const c = cameras[viewName];
      if (!c) return;
      const cam = viewer.camera;
      cam.up.fromArray(c.up).normalize();
      cam.position.fromArray(c.position);
      cam.lookAt(new THREE.Vector3().fromArray(c.lookAt));
      if (c.fovY && cam.isPerspectiveCamera) {{
        cam.fov = c.fovY;
        cam.updateProjectionMatrix();
      }}
      const controls = viewer.controls || viewer.cameraControls;
      if (controls && controls.target) {{
        controls.target.fromArray(c.lookAt);
        if (controls.update) controls.update();
      }}
      // Keep the GT side panel in sync no matter who triggered the change
      // (dropdown, best/worst buttons, focus bridge). updateGt is hoisted via
      // the const binding by the time any caller fires.
      if (typeof updateGt === 'function'
          && gtPanel && !gtPanel.classList.contains('collapsed')) {{
        updateGt(viewName);
      }}
    }}

    // ----- ground-truth side panel -----
    const gtPanel = document.getElementById('gtPanel');
    const gtImg = document.getElementById('gtImg');
    const gtEmpty = document.getElementById('gtEmpty');
    const gtViewLabel = document.getElementById('gtView');
    const gtBtn = document.getElementById('gtBtn');
    const hasAnyGt = Object.keys(gtUrls).length > 0;
    if (!hasAnyGt) {{
      gtBtn.disabled = true;
      gtBtn.title = 'No ground-truth images found for this scene';
      gtBtn.style.opacity = 0.5;
      gtBtn.style.cursor = 'not-allowed';
    }}

    function updateGt(viewName) {{
      gtViewLabel.textContent = viewName || '';
      const url = viewName ? gtUrls[viewName] : null;
      if (url) {{
        if (gtImg.getAttribute('src') !== url) gtImg.src = url;
        gtImg.style.display = '';
        gtEmpty.style.display = 'none';
      }} else {{
        gtImg.removeAttribute('src');
        gtImg.style.display = 'none';
        gtEmpty.style.display = '';
      }}
    }}

    function setGtOpen(open) {{
      const willOpen = !!open && hasAnyGt;
      gtPanel.classList.toggle('collapsed', !willOpen);
      gtPanel.setAttribute('aria-hidden', willOpen ? 'false' : 'true');
      gtBtn.setAttribute('aria-pressed', willOpen ? 'true' : 'false');
      if (willOpen) updateGt(viewSel.value);
      // Let the GS3D viewer pick up the new canvas size after layout settles.
      requestAnimationFrame(() => {{
        window.dispatchEvent(new Event('resize'));
        setTimeout(() => window.dispatchEvent(new Event('resize')), 220);
      }});
    }}

    gtBtn.addEventListener('click', () => {{
      if (gtBtn.disabled) return;
      setGtOpen(gtPanel.classList.contains('collapsed'));
    }});

    viewSel.addEventListener('change', e => applyCamera(e.target.value));

    const focusedLabel = document.getElementById('focusedObj');

    function focusObject(obj) {{
      if (!obj || !objectViews[obj] || !objectViews[obj].length) return false;
      currentObject = obj;
      const best = objectViews[obj][0];
      viewSel.value = best;
      applyCamera(best);
      focusedLabel.textContent = '→ ' + obj;
      return true;
    }}

    function pickFromObject(which) {{
      if (!currentObject || !objectViews[currentObject]) return;
      const ordered = objectViews[currentObject];
      const view = which === 'best' ? ordered[0] : ordered[ordered.length - 1];
      if (!view) return;
      viewSel.value = view;
      applyCamera(view);
    }}
    document.getElementById('bestBtn').addEventListener('click', () => pickFromObject('best'));
    document.getElementById('worstBtn').addEventListener('click', () => pickFromObject('worst'));

    // Cross-iframe focus bridge: a sibling 'focus bridge' iframe (re-mounted
    // on filter change) broadcasts {{type:'focus', obj}} here. If the bridge
    // posted before we set up the listener, we fall back to a localStorage
    // snapshot once the splat scene is ready.
    let splatReady = false;
    let pendingFocus = null;
    function maybeFocus(obj) {{
      if (!obj) return;
      if (!splatReady) {{ pendingFocus = obj; return; }}
      focusObject(obj);
    }}
    try {{
      const ch = new BroadcastChannel({channel_id});
      ch.addEventListener('message', e => {{
        if (e && e.data && e.data.type === 'focus') maybeFocus(e.data.obj);
      }});
    }} catch (e) {{ console.warn('BroadcastChannel unavailable', e); }}

    status.textContent = 'downloading .ply (this can take a while)…';
    viewer.addSplatScene({json.dumps(ply_url)}, {{
      progressiveLoad: true,
      showLoadingUI: true,
      splatAlphaRemovalThreshold: 5,
    }})
    .then(() => {{
      viewer.start();
      splatReady = true;
      status.textContent = `ready · ${{viewNames.length}} cameras`;
      let target = pendingFocus;
      if (!target) {{
        try {{ target = localStorage.getItem({storage_key}); }} catch (e) {{}}
      }}
      if (!target) target = currentObject;
      focusObject(target);
    }})
    .catch(err => {{
      status.textContent = 'load failed: ' + (err && err.message ? err.message : err);
      console.error(err);
    }});
  </script>
</body></html>"""


def focus_bridge_html(scene: str, target_obj: str | None) -> str:
    """Tiny hidden iframe. Streamlit remounts it whenever `target_obj` changes
    (because its HTML content differs), which executes the script and:
      1. writes a localStorage snapshot the viewer can read on first paint
      2. broadcasts a {type:'focus', obj} message on a scene-scoped channel

    The viewer iframe listens on the same channel; because ITS HTML is
    byte-stable per scene, it stays mounted and doesn't re-download the .ply."""
    payload = {
        "obj": target_obj or "",
        "channel": f"guide3d:{scene}",
        "storage": f"guide3d:focus:{scene}",
    }
    return f"""<!DOCTYPE html><html><body style="margin:0"><script>
(function () {{
  const p = {json.dumps(payload)};
  if (!p.obj) return;
  try {{ localStorage.setItem(p.storage, p.obj); }} catch (e) {{}}
  try {{ new BroadcastChannel(p.channel).postMessage({{type: 'focus', obj: p.obj}}); }} catch (e) {{}}
}})();
</script></body></html>"""


# ---------- filter helper ----------


def parse_filter(text: str, all_objects: list[str]) -> tuple[list[str], list[str]]:
    tokens = [t.strip() for t in text.split(",") if t.strip()]
    if not tokens:
        return all_objects, []

    lower_to_orig = {o.lower(): o for o in all_objects}
    matched: list[str] = []
    missing: list[str] = []
    for tok in tokens:
        tl = tok.lower()
        if tl in lower_to_orig:
            matched.append(lower_to_orig[tl])
            continue
        subs = [orig for low, orig in lower_to_orig.items() if tl in low]
        if subs:
            matched.extend(subs)
        else:
            missing.append(tok)

    seen: set[str] = set()
    return [m for m in matched if not (m in seen or seen.add(m))], missing


# ---------- sidebar ----------

st.sidebar.title("GUIDE-3D")
st.sidebar.caption("Object reconstruction quality ranking")

scenes = discover_scenes()
if not scenes:
    st.error(
        "No scenes available.\n\n"
        f"Need both `{MODELS_DIR}/<scene>/` (with `cameras.json`) "
        f"and `{METRICS_DIR}/<scene>.pkl`."
    )
    st.stop()

scene = st.sidebar.selectbox("Scene", scenes)

df_view_all = load_metrics(scene)
all_objects = sorted(df_view_all["object"].unique().tolist())

filter_text = st.sidebar.text_input(
    "Objects (comma-separated)",
    placeholder=", ".join(all_objects[:2]) if all_objects else "",
    help="Leave blank to show all objects in this scene. Substring match, case-insensitive.",
)

st.sidebar.caption(
    "Build your own ranking inside the LineUp tables: drag column headers, "
    "create weighted-sum columns, change sort direction, and filter by ranges."
)

with st.sidebar.expander("Available objects in this scene", expanded=False):
    st.write(all_objects)

# ---------- filter + aggregate ----------

matched, missing = parse_filter(filter_text, all_objects)
if missing:
    st.sidebar.warning("Not in this scene: " + ", ".join(missing))
if not matched:
    st.sidebar.error("No matching objects; falling back to all.")
    matched = all_objects

df_view = df_view_all[df_view_all["object"].isin(matched)].reset_index(drop=True)

# ---------- viewer setup ----------

ply_path = find_ply(scene)
if ply_path is None:
    st.error(f"No point_cloud.ply found under `{MODELS_DIR / scene / 'point_cloud'}`.")
    st.stop()

base_url = start_static_server()
ply_url = f"{base_url}/{ply_path.relative_to(ROOT).as_posix()}"

cameras = load_cameras(scene)
gt_urls = build_gt_urls(scene, cameras, base_url)


def img_to_view(img_name: str) -> str:
    """cameras.json uses 'DSCF0656', metrics use 'DSCF0656.JPG'. Bridge them."""
    return img_name if img_name in cameras else img_name.split(".")[0]


df_objects = aggregate_objects(df_view, cameras, total_cams=len(cameras))

# Filter-independent object -> ordered-views map. Computed from the
# unfiltered df_view_all so that the viewer iframe's HTML is byte-stable
# across filter changes and doesn't re-download the .ply every time.
object_views_all: dict[str, list[str]] = {}
for obj, grp in df_view_all.groupby("object"):
    sort_col = "psnr" if "psnr" in grp.columns else "view"
    ordered = grp.sort_values(sort_col, ascending=False)["view"].tolist()
    object_views_all[obj] = [img_to_view(v) for v in ordered if img_to_view(v) in cameras]

# Scene-default object for the viewer's very first paint.
_df_objects_all = aggregate_objects(df_view_all, cameras, total_cams=len(cameras))
default_object = (
    _df_objects_all["object"].iloc[0]
    if len(_df_objects_all)
    else next(iter(object_views_all), None)
)

# Current target object driven by the filter textbox. This is what the
# focus-bridge iframe will broadcast to the viewer.
target_object = (
    df_objects["object"].iloc[0]
    if len(df_objects)
    else (matched[0] if matched else default_object)
)

# ---------- main layout ----------

st.markdown(
    f"### Scene: `{scene}` "
    f"<span style='color:#888;font-weight:normal;font-size:0.8em'>"
    f"· {len(df_view)} measurements · {df_view['object'].nunique()} object(s) · "
    f"{df_view['view'].nunique()} views · ply {ply_path.stat().st_size / 1024**2:.0f} MB"
    f"</span>",
    unsafe_allow_html=True,
)

components.html(
    viewer_html(
        ply_url=ply_url,
        cameras=cameras,
        object_views=object_views_all,
        default_object=default_object,
        scene=scene,
        height_px=720,
        gt_urls=gt_urls,
    ),
    height=730,
    scrolling=False,
)

# Sibling hidden iframe that, on every filter change, broadcasts the new
# target object to the viewer above. Because the viewer's HTML is stable,
# this is the only iframe that remounts on filter change → camera jump
# happens without re-downloading the .ply.
components.html(focus_bridge_html(scene, target_object), height=0)

st.caption(
    "Camera control: left-drag to orbit · right-drag to pan · scroll to zoom. "
    "Typing an object in the sidebar jumps the camera to that object's best view. "
    "Use the toolbar's best/worst buttons to inspect extremes for the focused object."
)

with st.expander(f"Object ranking  ·  {len(df_objects)} object(s)", expanded=True):
    obj_h = min(140 + 32 * max(len(df_objects), 1), 480)
    components.html(lineup_html(df_objects, obj_h), height=obj_h + 20, scrolling=False)

with st.expander("Per-view ranking", expanded=False):
    chosen = st.selectbox(
        "Object",
        options=matched,
        index=matched.index(target_object) if target_object in matched else 0,
    )
    df_drill = per_view_display(df_view[df_view["object"] == chosen])
    drill_h = 520
    components.html(lineup_html(df_drill, drill_h), height=drill_h + 20, scrolling=False)
    st.caption(
        f"{len(df_drill)} views · sorted by PSNR. The 'best/worst view' buttons in the "
        f"viewer use this object's PSNR ranking."
    )
