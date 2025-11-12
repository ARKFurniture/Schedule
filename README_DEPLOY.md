# ARK Scheduler — Web App (Streamlit)

This repository lets you run the production scheduler **entirely online** using Streamlit Community Cloud.

## Deploy to Streamlit Cloud

1. Create a new **GitHub** repo and add these files to the root:
   - `ark_scheduler.py`  ← the core scheduler (provided above)
   - `streamlit_app.py`  ← the web UI
   - `requirements.txt`  ← Python dependencies
   - (optional) `ark_config.json` as a default config

2. Go to [streamlit.io](https://streamlit.io/cloud), click **"New app"**, select your repo and branch,
   and set **app file** to `streamlit_app.py`.

3. When the app is live, upload your **Dictionary CSV** and **Forecast CSV**, paste or edit the config JSON,
   and click **Run Scheduler**. A **schedule.csv** download button will appear.

### Notes
- The app imports `ark_scheduler.py` directly, so keep it in the repo root next to `streamlit_app.py`.
- You can customize employees, abilities, shifts, off-days, priorities, and delivery targets in the config JSON.
- Dry-time rules are in the config (`gap_after_finish_hours`, `gap_before_assembly_hours`).

## Local development (optional)
If you later have a Python environment:
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
