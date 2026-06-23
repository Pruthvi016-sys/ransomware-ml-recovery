"""
dashboard/app.py
Streamlit 4-tab unified investigation dashboard.

Tab 1 — Timeline View        : anomaly scores across snapshots, attack marked
Tab 2 — Blast Radius View    : directory heatmap, file type breakdown
Tab 3 — Recovery Recommendation : clean snapshot card, MTTR, affected files
Tab 4 — Event Log            : full snapshot history, exportable

Run with:
  streamlit run dashboard/app.py
"""

import os
import csv
import json
import sys
from collections import defaultdict

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Path Setup ────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

def data_path(filename):
    return os.path.join(BASE_DIR, "data", "dataset", filename)

# ── Data Loaders (cached) ─────────────────────────────────────────────────────


@st.cache_data
def load_csv(filename):
    path = data_path(filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp1252")

@st.cache_data
def load_recovery_report():
    path = data_path("recovery_report.csv")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except UnicodeDecodeError:
        with open(path, newline="", encoding="cp1252") as f:
            rows = list(csv.DictReader(f))
    return rows[0] if rows else {}
# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ransomware Recovery — ML Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cyber-security.png", width=64)
    st.title("🛡️ Ransomware Recovery")
    st.caption("ML-Powered Detection & Recovery Pipeline")
    st.divider()

    st.markdown("**Pipeline Stages**")
    st.markdown("✅ Stage 1 — Anomaly Detection")
    st.markdown("✅ Stage 2A — Mass Move Filter")
    st.markdown("✅ Stage 2B — Encryption Detection")
    st.markdown("✅ Stage 3A — Blast Radius Analysis")
    st.markdown("✅ Stage 3B — Recovery Point ID")
    st.divider()

    # Quick stats
    s1_df = load_csv("stage1_results.csv")
    if not s1_df.empty:
        n_flagged = int(s1_df["anomaly_flag"].sum())
        n_ransomware = int((s1_df["event_type"] == "ransomware").sum())
        st.metric("Snapshots Analysed", len(s1_df))
        st.metric("Anomalies Detected", n_flagged)
        st.metric("Ransomware Snapshots", n_ransomware)

    report = load_recovery_report()
    if report:
        st.divider()
        st.markdown("**Recovery Status**")
        st.success(f"Clean point: Snapshot [{report.get('clean_recovery_snapshot','?')}]")
        st.info(f"MTTR saved: {report.get('hours_saved','?')} hrs")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Timeline View",
    "💥 Blast Radius",
    "♻️ Recovery Recommendation",
    "📋 Event Log",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — TIMELINE VIEW
# ════════════════════════════════════════════════════════════════════════════════

with tab1:
    st.header("Snapshot Timeline & Anomaly Detection")

    s1_df  = load_csv("stage1_results.csv")
    s2b_df = load_csv("stage2b_results.csv")

    if s1_df.empty:
        st.warning("Run the pipeline first to generate stage1_results.csv")
    else:
        # Merge Stage 1 + Stage 2B
        if not s2b_df.empty:
            merged = s1_df.merge(
                s2b_df[["snapshot_id", "encryption_prob"]],
                on="snapshot_id", how="left"
            )
        else:
            merged = s1_df.copy()
            merged["encryption_prob"] = 0.0

        merged["encryption_prob"] = merged["encryption_prob"].fillna(0.0)

        # Color by event type
        color_map = {
            "clean":      "#2ecc71",
            "mass_move":  "#f39c12",
            "ransomware": "#e74c3c",
        }
        merged["color"] = merged["event_type"].map(color_map).fillna("#95a5a6")

        col1, col2 = st.columns(2)

        # ── Anomaly Score Timeline ────────────────────────────────────────────
        with col1:
            st.subheader("Stage 1 — Anomaly Scores")
            fig = go.Figure()

            for etype, color in color_map.items():
                mask = merged["event_type"] == etype
                subset = merged[mask]
                if subset.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=subset["snapshot_id"],
                    y=subset["anomaly_score"],
                    mode="markers+lines",
                    name=etype.replace("_", " ").title(),
                    marker=dict(color=color, size=8,
                                symbol="circle",
                                line=dict(width=1, color="white")),
                    line=dict(color=color, width=1.5, dash="dot"),
                    hovertemplate=(
                        "Snapshot %{x}<br>"
                        "Score: %{y:.4f}<br>"
                        f"Event: {etype}"
                    ),
                ))

            # Threshold line
            fig.add_hline(y=0.0, line_dash="dash",
                          line_color="white", opacity=0.4,
                          annotation_text="Anomaly threshold",
                          annotation_position="top left")

            # Attack marker
            attack_row = merged[merged["event_type"] == "ransomware"]
            if not attack_row.empty:
                first_attack = attack_row["snapshot_id"].min()
                fig.add_vline(x=first_attack, line_dash="dash",
                              line_color="#e74c3c", opacity=0.8,
                              annotation_text=f"⚠ Attack @ [{first_attack:03d}]",
                              annotation_font_color="#e74c3c")

            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Snapshot ID",
                yaxis_title="Anomaly Score (higher = more normal)",
                legend_title="Event Type",
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Encryption Probability Timeline ───────────────────────────────────
        with col2:
            st.subheader("Stage 2B — Encryption Probability")
            fig2 = go.Figure()

            for etype, color in color_map.items():
                mask = merged["event_type"] == etype
                subset = merged[mask]
                if subset.empty:
                    continue
                fig2.add_trace(go.Bar(
                    x=subset["snapshot_id"],
                    y=subset["encryption_prob"],
                    name=etype.replace("_", " ").title(),
                    marker_color=color,
                    opacity=0.85,
                    hovertemplate=(
                        "Snapshot %{x}<br>"
                        "Enc Prob: %{y:.4f}<br>"
                        f"Event: {etype}"
                    ),
                ))

            fig2.add_hline(y=0.5, line_dash="dash",
                           line_color="yellow", opacity=0.6,
                           annotation_text="Decision threshold (0.5)",
                           annotation_position="top left")

            fig2.update_layout(
                template="plotly_dark",
                xaxis_title="Snapshot ID",
                yaxis_title="Encryption Probability",
                barmode="overlay",
                legend_title="Event Type",
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Snapshot Detail ───────────────────────────────────────────────────
        st.subheader("Snapshot Detail")
        selected = st.slider("Select snapshot to inspect",
                             min_value=int(merged["snapshot_id"].min()),
                             max_value=int(merged["snapshot_id"].max()),
                             value=35)

        row = merged[merged["snapshot_id"] == selected]
        if not row.empty:
            r = row.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Event Type",    r["event_type"].replace("_", " ").title())
            c2.metric("Anomaly Score", f"{r['anomaly_score']:.4f}")
            c3.metric("Anomaly Flag",  "🚨 YES" if r["anomaly_flag"] else "✓ NO")
            c4.metric("Enc Probability", f"{r['encryption_prob']:.4f}")

            status_color = {
                "clean": "✅ Clean",
                "mass_move": "⚠️ Mass Move (False Positive)",
                "ransomware": "🔴 RANSOMWARE CONFIRMED",
            }.get(r["event_type"], "Unknown")
            st.markdown(f"### Status: {status_color}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — BLAST RADIUS
# ════════════════════════════════════════════════════════════════════════════════

with tab2:
    st.header("💥 Blast Radius Analysis")

    blast_df   = load_csv("blast_radius.csv")
    affected_df = load_csv("affected_files.csv")

    if blast_df.empty:
        st.warning("Run blast_radius.py first.")
    else:
        # Top metrics
        total_enc  = int(blast_df["encrypted"].sum())
        total_del  = int(blast_df["deleted"].sum())
        dirs_hit   = int((blast_df["total_affected"] > 0).sum())
        severity   = blast_df.loc[blast_df["impact_ratio"].idxmax(), "severity"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files Encrypted",  total_enc)
        c2.metric("Files Deleted",    total_del)
        c3.metric("Directories Hit",  f"{dirs_hit} / {len(blast_df)}")
        c4.metric("Severity",         f"🔴 {severity}" if severity == "CRITICAL"
                                       else f"🟡 {severity}")

        st.divider()
        col1, col2 = st.columns(2)

        # ── Directory Heatmap ─────────────────────────────────────────────────
        with col1:
            st.subheader("Directory Impact Heatmap")
            fig = go.Figure(go.Bar(
                x=blast_df["encrypted"],
                y=blast_df["directory"],
                orientation="h",
                marker=dict(
                    color=blast_df["impact_ratio"],
                    colorscale="RdYlGn_r",
                    showscale=True,
                    colorbar=dict(title="Impact Ratio"),
                ),
                text=[f"{r:.1%}" for r in blast_df["impact_ratio"]],
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Encrypted: %{x}<br>"
                    "Impact: %{text}"
                ),
            ))
            fig.update_layout(
                template="plotly_dark",
                xaxis_title="Files Encrypted",
                yaxis_title="",
                height=420,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── File Type Breakdown ───────────────────────────────────────────────
        with col2:
            st.subheader("Affected File Types")
            if not affected_df.empty:
                ext_counts = (affected_df[affected_df["change_type"] == "encrypted"]
                              ["extension"]
                              .value_counts()
                              .reset_index())
                ext_counts.columns = ["extension", "count"]

                fig2 = px.pie(
                    ext_counts.head(8),
                    names="extension",
                    values="count",
                    color_discrete_sequence=px.colors.sequential.RdBu_r,
                    hole=0.4,
                )
                fig2.update_layout(
                    template="plotly_dark",
                    height=420,
                    margin=dict(l=10, r=10, t=20, b=10),
                )
                st.plotly_chart(fig2, use_container_width=True)

        # ── Encrypted vs Deleted per Directory ───────────────────────────────
        st.subheader("Encrypted vs Deleted per Directory")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name="Encrypted",
            x=blast_df["directory"],
            y=blast_df["encrypted"],
            marker_color="#e74c3c",
        ))
        fig3.add_trace(go.Bar(
            name="Deleted",
            x=blast_df["directory"],
            y=blast_df["deleted"],
            marker_color="#e67e22",
        ))
        fig3.update_layout(
            template="plotly_dark",
            barmode="group",
            xaxis_title="Directory",
            yaxis_title="File Count",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # ── Affected Files Table ──────────────────────────────────────────────
        if not affected_df.empty:
            st.subheader("Affected Files (sample)")
            sample = affected_df.head(50)[
                ["file_path", "directory", "extension", "change_type",
                 "size_bytes", "entropy"]
            ]
            st.dataframe(sample, use_container_width=True, height=250)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — RECOVERY RECOMMENDATION
# ════════════════════════════════════════════════════════════════════════════════

with tab3:
    st.header("♻️ Recovery Recommendation")

    report    = load_recovery_report()
    diff_df   = load_csv("recovery_diff.csv")

    if not report:
        st.warning("Run recovery_point.py first.")
    else:
        # ── Recovery Card ─────────────────────────────────────────────────────
        clean_id  = report.get("clean_recovery_snapshot", "?")
        attack_id = report.get("attack_snapshot", "?")

        st.success(f"""
        ## ✅ Restore from Snapshot [{clean_id}]
        **Last known clean state** before the ransomware attack at Snapshot [{attack_id}].
        """)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Data Loss Window",    f"{report.get('data_loss_hours','?')} hrs")
        c2.metric("MTTR with ML",        f"{report.get('ml_mttr_hours','?')} hrs")
        c3.metric("MTTR without ML",     f"{report.get('manual_mttr_hours','?')} hrs")
        c4.metric("Time Saved",
                  f"{report.get('hours_saved','?')} hrs",
                  delta=f"{report.get('mttr_reduction_pct','?')}% reduction",
                  delta_color="normal")

        st.divider()
        col1, col2 = st.columns(2)

        # ── MTTR Comparison Chart ─────────────────────────────────────────────
        with col1:
            st.subheader("MTTR Comparison")
            fig = go.Figure(go.Bar(
                x=["Without ML\n(Manual)", "With ML Pipeline"],
                y=[float(report.get("manual_mttr_hours", 72)),
                   float(report.get("ml_mttr_hours", 8))],
                marker_color=["#e74c3c", "#2ecc71"],
                text=[f"{float(report.get('manual_mttr_hours',72)):.0f} hrs",
                      f"{float(report.get('ml_mttr_hours',8)):.0f} hrs"],
                textposition="outside",
                width=0.4,
            ))
            fig.update_layout(
                template="plotly_dark",
                yaxis_title="Hours to Recover",
                height=380,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(range=[0, float(report.get("manual_mttr_hours",72)) * 1.2]),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Data Loss Timeline ────────────────────────────────────────────────
        with col2:
            st.subheader("Recovery Timeline")
            clean_time  = report.get("clean_snapshot_time", "")
            attack_time = report.get("attack_snapshot_time", "")

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=[clean_time, attack_time],
                y=[1, 1],
                mode="markers+lines+text",
                marker=dict(size=[18, 18],
                            color=["#2ecc71", "#e74c3c"],
                            symbol=["circle", "x"]),
                line=dict(color="#e74c3c", width=3, dash="dot"),
                text=[f"✓ Snapshot [{clean_id}]\nClean",
                      f"⚠ Snapshot [{attack_id}]\nAttack"],
                textposition=["bottom center", "bottom center"],
                textfont=dict(size=12),
            ))
            fig2.update_layout(
                template="plotly_dark",
                xaxis_title="Time",
                yaxis=dict(visible=False),
                height=380,
                margin=dict(l=10, r=10, t=20, b=80),
                showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True)

        # ── Recovery Diff Table ───────────────────────────────────────────────
        st.subheader("Files to Restore / Delete")
        if not diff_df.empty:
            status_filter = st.multiselect(
                "Filter by status",
                options=diff_df["status"].unique().tolist(),
                default=diff_df["status"].unique().tolist(),
            )
            filtered = diff_df[diff_df["status"].isin(status_filter)]

            # Color rows by status
            def highlight_status(row):
                if row["status"] == "encrypted":
                    return ["background-color: #5d1a1a"] * len(row)
                elif row["status"] == "deleted":
                    return ["background-color: #4a3800"] * len(row)
                elif row["status"] == "ransom_note_added":
                    return ["background-color: #1a1a5d"] * len(row)
                return [""] * len(row)

            st.dataframe(
                filtered.style.apply(highlight_status, axis=1),
                use_container_width=True,
                height=300,
            )

            # Download
            csv_data = filtered.to_csv(index=False)
            st.download_button(
                "📥 Download Recovery Diff CSV",
                data=csv_data,
                file_name="recovery_diff.csv",
                mime="text/csv",
            )

        # ── Action Checklist ──────────────────────────────────────────────────
        st.subheader("Recovery Checklist")
        st.markdown(f"""
        - [ ] **Isolate** affected systems from network
        - [ ] **Restore** all files from Snapshot **[{clean_id}]**
        - [ ] **Delete** {report.get('diff_ransom_notes','?')} ransom note files
        - [ ] **Verify** restored data integrity
        - [ ] **Investigate** infection vector before reconnecting
        - [ ] **Patch** exploited vulnerability
        - [ ] **Re-enable** monitoring and alerting
        """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVENT LOG
# ════════════════════════════════════════════════════════════════════════════════

with tab4:
    st.header("📋 Full Event Log")

    s1_df  = load_csv("stage1_results.csv")
    s2b_df = load_csv("stage2b_results.csv")
    s2a_df = load_csv("stage2a_results.csv")

    if s1_df.empty:
        st.warning("Run the pipeline first.")
    else:
        if not s2b_df.empty:
            log_df = s1_df.merge(
                s2b_df[["snapshot_id", "encryption_prob", "ransomware_confirmed"]],
                on="snapshot_id", how="left"
            )
        else:
            log_df = s1_df.copy()
            log_df["encryption_prob"] = 0.0
            log_df["ransomware_confirmed"] = 0

        log_df["encryption_prob"]      = log_df["encryption_prob"].fillna(0.0)
        log_df["ransomware_confirmed"] = log_df["ransomware_confirmed"].fillna(0).astype(int)

        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            event_filter = st.multiselect(
                "Filter by event type",
                options=log_df["event_type"].unique().tolist(),
                default=log_df["event_type"].unique().tolist(),
            )
        with col2:
            flag_filter = st.checkbox("Show only flagged snapshots", value=False)

        filtered_log = log_df[log_df["event_type"].isin(event_filter)]
        if flag_filter:
            filtered_log = filtered_log[filtered_log["anomaly_flag"] == 1]

        # Display
        display_cols = [
            "snapshot_id", "event_type", "label",
            "anomaly_score", "anomaly_flag",
            "encryption_prob", "ransomware_confirmed", "correct"
        ]
        st.dataframe(
            filtered_log[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=400,
        )

        # ── Summary stats ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("Summary Statistics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Snapshots",   len(log_df))
        c2.metric("Clean",             int((log_df["event_type"] == "clean").sum()))
        c3.metric("Mass Move",         int((log_df["event_type"] == "mass_move").sum()))
        c4.metric("Ransomware",        int((log_df["event_type"] == "ransomware").sum()))
        c5.metric("Correctly Labelled",int(log_df["correct"].sum()))

        # ── Stage 2A events ───────────────────────────────────────────────────
        if not s2a_df.empty:
            st.subheader("Stage 2A — Mass Move Filter Decisions")
            st.dataframe(s2a_df, use_container_width=True, height=200)

        # ── Export ─────────────────────────────────────────────────────────────
        csv_export = log_df.to_csv(index=False)
        st.download_button(
            "📥 Download Full Event Log",
            data=csv_export,
            file_name="event_log.csv",
            mime="text/csv",
        )