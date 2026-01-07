import streamlit as st
import requests
import json
from datetime import datetime

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Bot Detection Dashboard", layout="wide")
st.title("ML Bot & Abuse Detection Dashboard")

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", API_URL)
    api_key = st.text_input("API Key (X-API-Key)", type="password")
    st.caption("Ensure the FastAPI server is running at this address.")

mode = st.radio("Mode", ["Single Event", "Session", "Explain", "About"], horizontal=True)

if mode == "Single Event":
    st.subheader("Score Single Activity Event")
    with st.form("single_form"):
        user_id = st.text_input("user_id", value="demo_user")
        timestamp = st.text_input("timestamp (ISO8601)", value=datetime.utcnow().isoformat() + "Z")
        request_path = st.text_input("request_path", value="/home")
        request_duration = st.number_input("request_duration (s)", min_value=0.0, value=0.25, step=0.01)
        typing_speed = st.number_input("typing_speed (CPM)", min_value=0.0, value=0.0, step=1.0)
        mouse_movements = st.text_area("mouse_movements (JSON list)", value="[]")
        click_pattern = st.text_area("click_pattern (JSON list)", value="[]")
        scroll_behavior = st.text_area("scroll_behavior (JSON)", value="{\n  \"speed\": 5,\n  \"direction\": \"down\"\n}")
        submitted = st.form_submit_button("Score Event")
    if submitted:
        try:
            payload = {
                "user_id": user_id,
                "timestamp": timestamp,
                "request_path": request_path,
                "request_duration": float(request_duration),
                "typing_speed": float(typing_speed),
                "mouse_movements": json.loads(mouse_movements or "[]"),
                "click_pattern": json.loads(click_pattern or "[]"),
                "scroll_behavior": json.loads(scroll_behavior or "{}"),
            }
            headers = {"X-API-Key": api_key} if api_key else None
            resp = requests.post(f"{api_url}/detect-bot", json=payload, headers=headers, timeout=15)
            if resp.ok:
                data = resp.json()
                st.success("Scored successfully")
                st.json(data)
                st.metric("Bot Probability", f"{data.get('bot_probability', 0):.2f}", help=str(data.get("debug")))
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.exception(e)

elif mode == "Explain":
    st.subheader("Explain Predictions")
    explain_tab = st.radio("Explain Type", ["Activity", "Session"], horizontal=True)
    if explain_tab == "Activity":
        with st.form("explain_activity_form"):
            user_id = st.text_input("user_id", value="explain_user")
            timestamp = st.text_input("timestamp (ISO8601)", value=datetime.utcnow().isoformat() + "Z")
            request_path = st.text_input("request_path", value="/home")
            request_duration = st.number_input("request_duration (s)", min_value=0.0, value=0.25, step=0.01)
            typing_speed = st.number_input("typing_speed (CPM)", min_value=0.0, value=0.0, step=1.0)
            mouse_movements = st.text_area("mouse_movements (JSON list)", value="[]")
            click_pattern = st.text_area("click_pattern (JSON list)", value="[]")
            scroll_behavior = st.text_area("scroll_behavior (JSON)", value="{\n  \"speed\": 5,\n  \"direction\": \"down\"\n}")
            submitted = st.form_submit_button("Explain Activity")
        if submitted:
            try:
                payload = {
                    "user_id": user_id,
                    "timestamp": timestamp,
                    "request_path": request_path,
                    "request_duration": float(request_duration),
                    "typing_speed": float(typing_speed),
                    "mouse_movements": json.loads(mouse_movements or "[]"),
                    "click_pattern": json.loads(click_pattern or "[]"),
                    "scroll_behavior": json.loads(scroll_behavior or "{}"),
                }
                headers = {"X-API-Key": api_key} if api_key else None
                resp = requests.post(f"{api_url}/explain-activity", json=payload, headers=headers, timeout=30)
                if resp.ok:
                    data = resp.json()
                    st.json(data.get("top_contributions", []))
                    if data.get("plot_base64"):
                        st.image("data:image/png;base64," + data["plot_base64"])
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.exception(e)
    else:
        with st.form("explain_session_form"):
            user_id = st.text_input("user_id", value="session_user")
            session_id = st.text_input("session_id", value="s1")
            events_json = st.text_area("events (JSON list of activities)", height=300, value=json.dumps([
                {
                    "user_id": "session_user",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "request_path": "/home",
                    "request_duration": 0.25,
                    "mouse_movements": [],
                    "click_pattern": [],
                    "typing_speed": 0,
                    "scroll_behavior": {"speed": 5, "direction": "down"}
                },
                {
                    "user_id": "session_user",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "request_path": "/product",
                    "request_duration": 0.32,
                    "mouse_movements": [],
                    "click_pattern": [],
                    "typing_speed": 0,
                    "scroll_behavior": {"speed": 6, "direction": "down"}
                }
            ], indent=2))
            submitted = st.form_submit_button("Explain Session")
        if submitted:
            try:
                events = json.loads(events_json or "[]")
                payload = {"user_id": user_id, "session_id": session_id, "events": events}
                headers = {"X-API-Key": api_key} if api_key else None
                resp = requests.post(f"{api_url}/explain-session", json=payload, headers=headers, timeout=30)
                if resp.ok:
                    data = resp.json()
                    st.json(data.get("top_contributions", []))
                    if data.get("plot_base64"):
                        st.image("data:image/png;base64," + data["plot_base64"])
                else:
                    st.error(f"Error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.exception(e)

elif mode == "Session":
    st.subheader("Score Session (Multiple Events)")
    with st.form("session_form"):
        user_id = st.text_input("user_id", value="session_user")
        session_id = st.text_input("session_id", value="s1")
        events_json = st.text_area("events (JSON list of activities)", height=300, value=json.dumps([
            {
                "user_id": "session_user",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_path": "/home",
                "request_duration": 0.25,
                "mouse_movements": [],
                "click_pattern": [],
                "typing_speed": 0,
                "scroll_behavior": {"speed": 5, "direction": "down"}
            },
            {
                "user_id": "session_user",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_path": "/product",
                "request_duration": 0.32,
                "mouse_movements": [],
                "click_pattern": [],
                "typing_speed": 0,
                "scroll_behavior": {"speed": 6, "direction": "down"}
            }
        ], indent=2))
        submitted = st.form_submit_button("Score Session")
    if submitted:
        try:
            events = json.loads(events_json or "[]")
            payload = {"user_id": user_id, "session_id": session_id, "events": events}
            headers = {"X-API-Key": api_key} if api_key else None
            resp = requests.post(f"{api_url}/score-session", json=payload, headers=headers, timeout=30)
            if resp.ok:
                data = resp.json()
                st.success("Scored successfully")
                st.json(data)
                st.metric("Fused Bot Probability", f"{data.get('bot_probability', 0):.2f}")
                if "debug" in data:
                    st.caption("Debug signals")
                    st.json(data["debug"])
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
        except Exception as e:
            st.exception(e)

else:
    st.subheader("About")
    st.write("This dashboard calls the FastAPI service to score events and sessions using a fused ML approach (supervised, anomaly, and sequence models).")
