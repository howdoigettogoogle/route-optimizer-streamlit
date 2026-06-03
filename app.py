from __future__ import annotations

import io
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import folium
import pandas as pd
import requests
import streamlit as st
from folium.features import DivIcon
from ortools.sat.python import cp_model
from streamlit_folium import st_folium


# -------------------------- Configuration Defaults --------------------------

WAREHOUSE_ADDR = "In Schalmen 15, 78056 Villingen-Schwenningen"
DEFAULT_OSRM_BASE_URL = "https://router.project-osrm.org"
DEFAULT_OSRM_BLOCK = 50
HTTP_TIMEOUT = 30.0
RETRY_MAX = 5
BACKOFF_BASE = 1.5
BACKOFF_CAP = 60.0


# ------------------------------- Utilities ----------------------------------

def sleep_backoff(attempt: int) -> None:
    """Exponential backoff with jitter."""
    delay = min(BACKOFF_CAP, (BACKOFF_BASE ** attempt) + random.random())
    time.sleep(delay)


def is_latlon_string(s: str) -> Optional[Tuple[float, float]]:
    """Try to parse 'lat, lon' or 'lat lon' into floats."""
    try:
        sep = "," if "," in s else " "
        parts = [p.strip() for p in s.split(sep) if p.strip()]
        if len(parts) != 2:
            return None
        lat = float(parts[0])
        lon = float(parts[1])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None
        return lat, lon
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def nominatim_geocode(query: str, lang: str = "en") -> Optional[Tuple[float, float, str]]:
    """Geocode an address using Nominatim. Cached to avoid repeated lookups."""
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        # Replace the contact string with your own email if you deploy this publicly.
        "User-Agent": "route_optimizer_streamlit/1.0 (contact: micool@duck.com)",
        "Accept-Language": lang or "en",
    }
    params = {"q": query, "format": "json", "limit": 1}

    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                arr = resp.json()
                if arr:
                    lat = float(arr[0]["lat"])
                    lon = float(arr[0]["lon"])
                    disp = arr[0].get("display_name", query)
                    return lat, lon, disp
                return None
            if resp.status_code in (429, 502, 503, 504):
                sleep_backoff(attempt)
            else:
                return None
        except requests.RequestException:
            sleep_backoff(attempt)
    return None


def parse_latlon_or_geocode(value: str) -> Tuple[float, float, str]:
    """Return (lat, lon, display_name) from either coordinates or an address."""
    value = value.strip()
    if not value:
        raise ValueError("Enter coordinates or an address.")

    parsed = is_latlon_string(value)
    if parsed:
        lat, lon = parsed
        return lat, lon, f"{lat:.6f}, {lon:.6f}"

    res = nominatim_geocode(value)
    if not res:
        raise ValueError(f"Could not resolve this address: {value}")
    return res


def resolve_location(choice: str, custom_value: str) -> Tuple[float, float, str]:
    if choice == "Warehouse":
        res = nominatim_geocode(WAREHOUSE_ADDR)
        if not res:
            raise ValueError("Could not geocode the warehouse address. Use Custom and enter coordinates instead.")
        return res
    return parse_latlon_or_geocode(custom_value)


def to_float_coord(x) -> float:
    try:
        return float(str(x).replace(",", ".").strip())
    except Exception:
        return math.nan


def read_csv_bytes(data: bytes, source_name: str) -> pd.DataFrame:
    """Read a semicolon-delimited CSV from bytes with UTF-8 fallback handling."""
    last_exc: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(data), delimiter=";", dtype=str, encoding=encoding)
        except Exception as exc:  # try the next encoding
            last_exc = exc
    raise ValueError(f"Could not read {source_name} as a semicolon CSV: {last_exc}")


def clean_stops_dataframe(df_all: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean one concatenated stops dataframe."""
    if "Latitude" not in df_all.columns or "Longitude" not in df_all.columns:
        raise ValueError("Missing required columns: Latitude and/or Longitude.")

    before = len(df_all)
    df_all = df_all.copy()
    df_all["Latitude"] = df_all["Latitude"].apply(to_float_coord)
    df_all["Longitude"] = df_all["Longitude"].apply(to_float_coord)
    df_all = df_all.dropna(subset=["Latitude", "Longitude"]).copy()
    dropped = before - len(df_all)

    if df_all.empty:
        raise ValueError("No valid stop coordinates found after cleaning Latitude/Longitude.")

    df_all.attrs["dropped_rows"] = dropped
    return df_all


def load_stops_from_upload(uploaded_file) -> pd.DataFrame:
    """Read one uploaded CSV file. Kept single-file for Android reliability."""
    if uploaded_file is None:
        raise ValueError("Upload a CSV file or paste CSV text.")
    data = uploaded_file.getvalue()
    df = read_csv_bytes(data, getattr(uploaded_file, "name", "uploaded file"))
    df["__source_file__"] = getattr(uploaded_file, "name", "uploaded.csv")
    return clean_stops_dataframe(df)


def load_stops_from_paste(csv_text: str) -> pd.DataFrame:
    """Read semicolon-delimited CSV text pasted into the app."""
    if not csv_text.strip():
        raise ValueError("Paste CSV text or upload a CSV file.")
    df = pd.read_csv(io.StringIO(csv_text.strip()), delimiter=";", dtype=str)
    df["__source_file__"] = "pasted_csv"
    return clean_stops_dataframe(df)


def detect_status_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Status", "status", "Vehicle status", "vehicle_status", "vehicleStatus"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def osrm_table(
    base_url: str,
    coords_all: List[Tuple[float, float]],
    src_idx: List[int],
    dst_idx: List[int],
) -> List[List[Optional[float]]]:
    """Call OSRM table for given source/destination indices. Returns durations in seconds."""
    coords_str = ";".join([f"{lon:.6f},{lat:.6f}" for lat, lon in coords_all])
    sources_str = ";".join(str(i) for i in src_idx)
    dest_str = ";".join(str(j) for j in dst_idx)
    url = f"{base_url.rstrip('/')}/table/v1/driving/{coords_str}"
    params = {
        "annotations": "duration",
        "sources": sources_str,
        "destinations": dest_str,
    }

    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("durations")
            if resp.status_code in (429, 502, 503, 504):
                sleep_backoff(attempt)
            else:
                raise RuntimeError(f"OSRM table error {resp.status_code}: {resp.text[:300]}")
        except requests.RequestException as exc:
            if attempt == RETRY_MAX:
                raise RuntimeError(f"OSRM table request failed: {exc}") from exc
            sleep_backoff(attempt)

    raise RuntimeError("OSRM table failed after retries.")


@st.cache_data(show_spinner=False)
def build_duration_matrix_cached(
    coords_tuple: Tuple[Tuple[float, float], ...],
    block: int,
    base_url: str,
) -> List[List[int]]:
    coords = list(coords_tuple)
    n = len(coords)
    mat: List[List[Optional[int]]] = [[None for _ in range(n)] for _ in range(n)]

    row_tiles = [(i, min(i + block, n)) for i in range(0, n, block)]
    col_tiles = [(j, min(j + block, n)) for j in range(0, n, block)]

    for i0, i1 in row_tiles:
        for j0, j1 in col_tiles:
            src_idx = list(range(i0, i1))
            dst_idx = list(range(j0, j1))
            tile = osrm_table(base_url, coords, src_idx, dst_idx)
            if tile is None:
                raise RuntimeError("OSRM returned no durations for a matrix tile.")
            if len(tile) != len(src_idx) or any(len(row) != len(dst_idx) for row in tile):
                raise RuntimeError("OSRM tile dimension mismatch.")

            for a, src in enumerate(src_idx):
                for b, dst in enumerate(dst_idx):
                    val = tile[a][b]
                    if val is None:
                        raise RuntimeError(f"No route between nodes {src} -> {dst}. Cannot guarantee optimality.")
                    mat[src][dst] = max(0, int(round(val)))

    if any(mat[i][j] is None for i in range(n) for j in range(n)):
        raise RuntimeError("Incomplete duration matrix after OSRM tiling.")

    return mat  # type: ignore[return-value]


def solve_exact_path(
    duration: List[List[int]],
    start_idx: int,
    end_idx: int,
    time_limit_sec: Optional[float],
) -> Tuple[List[int], int]:
    """Solve exact shortest s->t Hamiltonian path using OR-Tools CP-SAT with MTZ constraints."""
    n = len(duration)
    if n < 2:
        return list(range(n)), 0

    all_nodes = range(n)
    model = cp_model.CpModel()

    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    u = [model.NewIntVar(0, n - 1, f"u_{i}") for i in all_nodes]

    for i in all_nodes:
        out_sum = [x[(i, j)] for j in all_nodes if i != j]
        in_sum = [x[(j, i)] for j in all_nodes if i != j]

        if i == start_idx:
            model.Add(sum(out_sum) == 1)
            model.Add(sum(in_sum) == 0)
        elif i == end_idx:
            model.Add(sum(in_sum) == 1)
            model.Add(sum(out_sum) == 0)
        else:
            model.Add(sum(out_sum) == 1)
            model.Add(sum(in_sum) == 1)

    model.Add(u[start_idx] == 0)
    model.Add(u[end_idx] == n - 1)

    big_m = n
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                model.Add(u[i] + 1 <= u[j] + big_m * (1 - x[(i, j)]))

    model.Minimize(
        sum(duration[i][j] * x[(i, j)] for i in all_nodes for j in all_nodes if i != j)
    )

    solver = cp_model.CpSolver()
    if time_limit_sec is not None and time_limit_sec > 0:
        solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = 8
    solver.parameters.stop_after_first_solution = False

    status = solver.Solve(model)
    if status != cp_model.OPTIMAL:
        status_name = solver.StatusName(status)
        if status == cp_model.FEASIBLE:
            raise RuntimeError(
                "The solver found a route but could not prove it is the absolute shortest within the time limit. "
                "Increase the time limit or reduce the number of stops. "
                f"Solver status: {status_name}."
            )
        raise RuntimeError(f"Solver did not prove an optimal route. Solver status: {status_name}.")

    succ = {i: None for i in all_nodes}
    for i in all_nodes:
        for j in all_nodes:
            if i != j and solver.BooleanValue(x[(i, j)]):
                succ[i] = j

    order = [start_idx]
    while order[-1] != end_idx:
        nxt = succ[order[-1]]
        if nxt is None:
            raise RuntimeError("Broken successor chain while reconstructing the route.")
        if nxt in order:
            raise RuntimeError("Cycle detected while reconstructing the route.")
        order.append(nxt)

    if len(order) != n:
        raise RuntimeError(f"Unexpected path length {len(order)} != {n}.")

    total_seconds = int(round(solver.ObjectiveValue()))
    return order, total_seconds


@st.cache_data(show_spinner=False)
def fetch_osrm_polyline_cached(
    base_url: str,
    coords_ordered_tuple: Tuple[Tuple[float, float], ...],
) -> List[Tuple[float, float]]:
    coords_ordered = list(coords_ordered_tuple)
    coords_str = ";".join([f"{lon:.6f},{lat:.6f}" for lat, lon in coords_ordered])
    url = f"{base_url.rstrip('/')}/route/v1/driving/{coords_str}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
    }

    for attempt in range(1, RETRY_MAX + 1):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                routes = data.get("routes", [])
                if not routes:
                    raise RuntimeError("OSRM returned no route for the optimized order.")
                geom = routes[0]["geometry"]["coordinates"]
                return [(lat, lon) for lon, lat in geom]
            if resp.status_code in (429, 502, 503, 504):
                sleep_backoff(attempt)
            else:
                raise RuntimeError(f"OSRM route error {resp.status_code}: {resp.text[:300]}")
        except requests.RequestException as exc:
            if attempt == RETRY_MAX:
                raise RuntimeError(f"OSRM route request failed: {exc}") from exc
            sleep_backoff(attempt)

    raise RuntimeError("OSRM route failed after retries.")


def make_map(
    start: Tuple[float, float, str],
    end: Tuple[float, float, str],
    stops_df: pd.DataFrame,
    visit_order: List[int],
    polyline: List[Tuple[float, float]],
    status_col: Optional[str],
) -> folium.Map:
    all_nodes: List[Tuple[float, float, str]] = [(start[0], start[1], start[2])]
    for _, row in stops_df.iterrows():
        all_nodes.append((float(row["Latitude"]), float(row["Longitude"]), ""))
    all_nodes.append((end[0], end[1], end[2]))

    avg_lat = sum(lat for lat, _, _ in all_nodes) / len(all_nodes)
    avg_lon = sum(lon for _, lon, _ in all_nodes) / len(all_nodes)

    m = folium.Map(
        location=(avg_lat, avg_lon),
        zoom_start=12,
        control_scale=True,
        tiles="CartoDB Voyager",
    )

    folium.Marker(
        location=(start[0], start[1]),
        popup=folium.Popup(f"<b>Start</b><br>{start[2]}", max_width=300),
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    folium.Marker(
        location=(end[0], end[1]),
        popup=folium.Popup(f"<b>End</b><br>{end[2]}", max_width=300),
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    stop_seq_by_index: Dict[int, int] = {}
    seq = 1
    for idx in visit_order:
        if 1 <= idx <= len(all_nodes) - 2:
            stop_seq_by_index[idx] = seq
            seq += 1

    for i in range(1, len(all_nodes) - 1):
        lat, lon, _ = all_nodes[i]
        seqnum = stop_seq_by_index.get(i)
        row = stops_df.iloc[i - 1]
        parts = [f"<b>Stop #{seqnum if seqnum is not None else i}</b>"]

        for label, candidates in [
            ("Vehicle", ["Vehicle number", "Vehicle", "Vehicle Number", "vehicle_number"]),
            ("Battery", ["Vehicle battery", "Battery", "battery"]),
            ("Last ride", ["Last ride", "last_ride", "Last Ride"]),
        ]:
            for col in candidates:
                if col in stops_df.columns and pd.notna(row.get(col)):
                    parts.append(f"{label}: {row.get(col)}")
                    break

        if status_col and status_col in stops_df.columns:
            val = row.get(status_col)
            if pd.notna(val):
                parts.append(f"Status: {val}")

        parts.append(
            f'<a target="_blank" href="https://www.google.com/maps?q={lat:.6f},{lon:.6f}">Open in Google Maps</a>'
        )
        popup_html = "<br>".join(parts)

        label = str(seqnum if seqnum is not None else i)
        folium.map.Marker(
            [lat, lon],
            icon=DivIcon(
                icon_size=(24, 24),
                icon_anchor=(12, 12),
                html=(
                    '<div style="background:#2b6cb0;color:white;border-radius:50%;width:24px;height:24px;'
                    'display:flex;align-items:center;justify-content:center;'
                    f'font-weight:bold;font-size:12px;">{label}</div>'
                ),
            ),
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(m)

    folium.PolyLine(locations=polyline, weight=5, opacity=0.8).add_to(m)
    return m


def seconds_to_hms(total_seconds: int) -> str:
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:d}h {m:02d}m {s:02d}s"


def route_table(order: List[int], stops_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stop_num = 1
    for node_idx in order:
        if node_idx == 0:
            rows.append({"Visit order": 0, "Type": "Start", "Latitude": "", "Longitude": "", "Source file": ""})
        elif node_idx == len(stops_df) + 1:
            rows.append({"Visit order": len(order) - 1, "Type": "End", "Latitude": "", "Longitude": "", "Source file": ""})
        else:
            row = stops_df.iloc[node_idx - 1]
            rows.append(
                {
                    "Visit order": stop_num,
                    "Type": "Stop",
                    "Latitude": row["Latitude"],
                    "Longitude": row["Longitude"],
                    "Source file": row.get("__source_file__", ""),
                }
            )
            stop_num += 1
    return pd.DataFrame(rows)


# -------------------------------- Streamlit UI -------------------------------

st.set_page_config(
    page_title="Route Optimizer",
    page_icon="🗺️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {
          padding-top: 0.7rem;
          padding-left: 0.75rem;
          padding-right: 0.75rem;
          max-width: 760px;
      }
      h1 { font-size: 1.65rem !important; line-height: 1.15 !important; }
      h2, h3 { margin-top: 0.75rem !important; }
      [data-testid="stFileUploader"] section {
          padding: 0.65rem !important;
      }
      [data-testid="stFileUploader"] button {
          width: 100% !important;
          min-height: 2.75rem !important;
      }
      .stButton > button, .stDownloadButton > button {
          width: 100%;
          min-height: 2.9rem;
          font-weight: 700;
      }
      div[data-testid="stRadio"] label { min-height: 2.2rem; }
      div[data-testid="stExpander"] details { border-radius: 0.7rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Route Optimizer")
st.caption("Mobile-first version: upload or paste stops, choose start/end, then calculate the proven shortest driving-time route.")

st.info(
    "Android tip: use one CSV file. If Android will not show your CSV in the picker, use **Paste CSV** below instead."
)

input_mode = st.radio(
    "Stops input",
    ["Upload file", "Paste CSV"],
    horizontal=True,
)

uploaded_file = None
csv_text = ""

if input_mode == "Upload file":
    uploaded_file = st.file_uploader(
        "Upload one semicolon-delimited CSV",
        # No type filter: Android file pickers can hide .csv files when accept/type filters are present.
        type=None,
        accept_multiple_files=False,
        help="CSV must contain Latitude and Longitude columns. Files renamed to .txt also work if the content is semicolon-delimited CSV.",
    )
    st.caption("File not visible? Open your file manager and choose **All files**, or switch to **Paste CSV**.")
else:
    csv_text = st.text_area(
        "Paste semicolon-delimited CSV text",
        height=220,
        placeholder="Latitude;Longitude;Vehicle number;Vehicle battery;Status\n48.059123;8.458123;V001;72%;Available",
    )

st.subheader("Start")
start_choice = st.radio("Start location", ["Warehouse", "Custom"], horizontal=True, key="start_choice")
start_custom = ""
if start_choice == "Custom":
    start_custom = st.text_input("Start address or coordinates", placeholder="48.059123, 8.458123")

st.subheader("End")
end_choice = st.radio("End location", ["Warehouse", "Custom"], horizontal=True, key="end_choice")
end_custom = ""
if end_choice == "Custom":
    end_custom = st.text_input("End address or coordinates", placeholder="In Schalmen 15, 78056 Villingen-Schwenningen")

with st.expander("Advanced settings"):
    osrm_base_url = st.text_input("OSRM base URL", value=DEFAULT_OSRM_BASE_URL)
    osrm_block = st.number_input("OSRM matrix block size", min_value=5, max_value=100, value=DEFAULT_OSRM_BLOCK, step=5)
    solver_time_limit = st.number_input(
        "Solver time limit in seconds",
        min_value=0,
        max_value=3600,
        value=120,
        step=30,
        help="0 means no limit. With a limit, the app only returns a route if optimality is proven inside the limit.",
    )

with st.expander("CSV format"):
    st.write("Required columns:")
    st.code("Latitude;Longitude", language="text")
    st.write("Optional columns used in popups: Vehicle number, Vehicle battery, Last ride, Status.")
    st.write("For Android reliability, this version accepts one CSV at a time. Merge multiple CSVs into one file if needed.")

run = st.button("Optimize route", type="primary")

if not run:
    st.stop()

try:
    with st.status("Preparing route...", expanded=True) as status:
        if input_mode == "Upload file":
            df = load_stops_from_upload(uploaded_file)
        else:
            df = load_stops_from_paste(csv_text)

        dropped_rows = df.attrs.get("dropped_rows", 0)
        status.write(f"Loaded {len(df)} valid stops. Dropped {dropped_rows} rows with invalid coordinates.")

        start = resolve_location(start_choice, start_custom)
        end = resolve_location(end_choice, end_custom)
        status.write(f"Start: {start[2]}")
        status.write(f"End: {end[2]}")

        coords: List[Tuple[float, float]] = [(start[0], start[1])]
        coords += list(zip(df["Latitude"].astype(float), df["Longitude"].astype(float)))
        coords.append((end[0], end[1]))

        if len(coords) > 75:
            st.warning(
                "This is a large exact optimization problem. Public OSRM and the exact solver may be slow or may reject very large requests."
            )

        status.update(label="Building driving-time matrix with OSRM...", state="running")
        duration = build_duration_matrix_cached(tuple(coords), int(osrm_block), osrm_base_url)
        status.write("Driving-time matrix built.")

        status.update(label="Solving exact shortest path...", state="running")
        time_limit = None if solver_time_limit == 0 else float(solver_time_limit)
        order, total_seconds = solve_exact_path(duration, 0, len(coords) - 1, time_limit)
        status.write(f"Optimal route proven. Total driving time: {seconds_to_hms(total_seconds)}.")

        status.update(label="Fetching display route geometry...", state="running")
        coords_ordered = [coords[i] for i in order]
        polyline = fetch_osrm_polyline_cached(osrm_base_url, tuple(coords_ordered))
        status.update(label="Done.", state="complete")

    status_col = detect_status_column(df)
    fmap = make_map(start=start, end=end, stops_df=df, visit_order=order, polyline=polyline, status_col=status_col)

    st.success(f"Optimal total driving time: {seconds_to_hms(total_seconds)} ({total_seconds:,} seconds)")

    st_folium(fmap, height=500, use_container_width=True)

    html = fmap.get_root().render()
    st.download_button(
        "Download map as HTML",
        data=html,
        file_name="optimal_route_osrm_exact.html",
        mime="text/html",
    )

    with st.expander("Route order", expanded=False):
        ordered_df = route_table(order, df)
        st.dataframe(ordered_df, use_container_width=True, hide_index=True)

except Exception as exc:
    st.error(str(exc))
    st.stop()
