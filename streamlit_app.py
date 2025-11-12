import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
import tempfile, os

# Import the scheduler core (expects ark_scheduler.py in the same repo)
import ark_scheduler as ark

st.set_page_config(page_title="ARK Scheduler", layout="wide")
st.title("ARK Production Scheduler (Web)")

st.markdown("Upload your **Dictionary CSV** and **Forecast CSV**, review the JSON config, then click **Run Scheduler**.")

col1, col2 = st.columns(2)
with col1:
    dict_file = st.file_uploader("Production Hour Dictionary CSV", type=["csv"], key="dict")
with col2:
    forecast_file = st.file_uploader("Forecast CSV", type=["csv"], key="forecast")

st.markdown("### Configuration (JSON)")
default_cfg_text = "{\n  \"window\": {\n    \"start\": \"2025-11-12 08:00\",\n    \"end\": \"2025-11-23 23:59\"\n  },\n  \"rules\": {\n    \"gap_after_finish_hours\": 2,\n    \"gap_before_assembly_hours\": 12,\n    \"assembly_earliest_hour\": 9\n  },\n  \"employees\": [],\n  \"priorities\": {\n    \"customers\": {},\n    \"targets\": []\n  },\n  \"special_projects\": []\n}"
cfg_text = st.text_area("Edit as needed", value=default_cfg_text, height=320)

run = st.button("Run Scheduler", type="primary", disabled=not(dict_file and forecast_file))

if run:
    try:
        cfg = json.loads(cfg_text)
    except Exception as e:
        st.error(f"Config JSON parse error: {e}")
        st.stop()

    # Save uploads to temp files that pandas can read
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_dict:
        tmp_dict.write(dict_file.read()); dict_path = tmp_dict.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_fore:
        tmp_fore.write(forecast_file.read()); forecast_path = tmp_fore.name

    try:
        service_blocks, service_stage_orders = ark.load_service_blocks(dict_path)
        jobs, unsched = ark.build_job_instances(forecast_path, service_blocks, service_stage_orders)
        df = ark.schedule_jobs(cfg, jobs, service_stage_orders)
        v2, v12 = ark.validate_schedule(df, cfg["rules"]["gap_after_finish_hours"], cfg["rules"]["gap_before_assembly_hours"])
    except Exception as e:
        st.error(f"Scheduling error: {e}")
        raise

    st.success(f"Schedule built. 2h-gap violations={v2}, 12h-before-assembly violations={v12}")

    if not df.empty:
        st.markdown("### Hours by worker")
        st.dataframe(df.groupby("Assigned To")["Hours"].sum().round(2).reset_index())

        st.markdown("### Preview (first 200 rows)")
        st.dataframe(df.head(200))

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download schedule.csv", data=csv_bytes, file_name="schedule.csv", mime="text/csv")

    if unsched:
        st.warning("Some rows were not schedulable (showing up to 10):")
        import itertools
        subset = list(itertools.islice(unsched, 10))
        st.json(subset)
