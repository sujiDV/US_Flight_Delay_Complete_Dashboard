
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path

st.set_page_config(
    page_title="U.S. Flight Delay Explorer",
    page_icon="✈️",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

TIME_ORDER = [
    "Overnight", "Early Morning",
    "Morning", "Afternoon", "Evening"
]

WEEK_ORDER = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

CAUSE_LABELS = {
    "CarrierDelay": "Carrier",
    "WeatherDelay": "Weather",
    "NASDelay": "National Aviation System",
    "SecurityDelay": "Security",
    "LateAircraftDelay": "Late Aircraft"
}

CAUSE_COLS = list(CAUSE_LABELS)

@st.cache_data
def load_data():
    return (
        pd.read_csv(BASE / "patterns.csv"),
        pd.read_csv(BASE / "conditions.csv"),
        pd.read_csv(BASE / "causes.csv"),
        pd.read_parquet(BASE / "severity.parquet"),
        pd.read_parquet(BASE / "propagation.parquet")
    )

patterns, conditions, causes, severity, propagation = load_data()

def by_month(df, month):
    if month == "All Months":
        return df.copy()
    return df[df["Month"] == int(month)].copy()

def aggregate(df, dimensions):
    if df.empty:
        return pd.DataFrame()

    out = df.groupby(
        dimensions, as_index=False
    ).agg(
        Flights=("Flights", "sum"),
        DelayedFlights=("DelayedFlights", "sum")
    )

    out["DelayRate"] = (
        out["DelayedFlights"] / out["Flights"] * 100
    )
    return out

def no_data():
    st.info("No records match the selected filters.")

st.title("✈️ U.S. Flight Delay Explorer")
st.write(
    "Explore where and when U.S. flights are delayed, "
    "how patterns differ across travel conditions, "
    "how severe delays are, and how they relate to "
    "subsequent aircraft operations."
)

available_months = sorted(
    patterns["Month"].unique().tolist()
)

month_names = {
    1:"January", 2:"February", 3:"March",
    4:"April", 5:"May", 6:"June",
    7:"July", 8:"August", 9:"September",
    10:"October", 11:"November", 12:"December"
}

month = st.sidebar.selectbox(
    "Month",
    ["All Months"] + available_months,
    format_func=lambda x: month_names.get(x, x)
)

st.sidebar.caption(
    "BTS 2025 Reporting Carrier On-Time Performance"
)

if len(available_months) == 1:
    st.info(
        f"Pilot dataset: {month_names[available_months[0]]} "
        "2025 only. Results are not full-year estimates."
    )
else:
    st.caption(
        f"2025 data | {len(available_months)} months available"
    )

st.caption(
    "Arrival-delay rate: share of eligible flights arriving "
    "at least 15 minutes late. Cancelled and diverted flights "
    "are excluded from this denominator."
)

tabs = st.tabs([
    "1. Overall Patterns",
    "2. Travel Conditions",
    "3. Reported Causes",
    "4. Delay Severity",
    "5. Delay Propagation"
])

# ===================================================
# TAB 1 — OVERALL PATTERNS
# ===================================================

with tabs[0]:

    st.header("Where and When Do Delays Occur?")

    p = by_month(patterns, month)

    total = int(p["Flights"].sum())
    delayed = int(p["DelayedFlights"].sum())
    rate = 100 * delayed / total if total else 0

    a, b, c = st.columns(3)
    a.metric("Eligible Arrivals", f"{total:,}")
    b.metric("Delayed Arrivals", f"{delayed:,}")
    c.metric("Arrival Delay Rate", f"{rate:.1f}%")

    st.subheader("Airport Delay Rates")

    min_airport = st.number_input(
        "Minimum eligible flights per airport",
        min_value=1, value=1000, step=100,
        key="airport_threshold"
    )

    airports = aggregate(p, ["Origin"])
    airports = airports[airports["Flights"] >= min_airport]

    # Busiest airports, not a best/worst ranking
    airports = airports.nlargest(20, "Flights")
    airports = airports.sort_values("DelayRate")

    if airports.empty:
        no_data()
    else:
        fig = px.scatter(
            airports,
            x="DelayRate",
            y="Origin",
            hover_data=["Flights", "DelayedFlights"],
            labels={
                "DelayRate": "Arrival Delay Rate (%)",
                "Origin": "Departure Airport"
            },
            title="20 Busiest Eligible Departure Airports"
        )
        fig.update_traces(marker_size=11)
        fig.update_yaxes(
            categoryorder="array",
            categoryarray=airports["Origin"].tolist()
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Delay Rate by Departure Hour")

    hourly = aggregate(p, ["DepartureHour"])
    hourly = hourly.sort_values("DepartureHour")

    fig = px.line(
        hourly,
        x="DepartureHour",
        y="DelayRate",
        markers=True,
        hover_data=["Flights"],
        labels={
            "DepartureHour": "Scheduled Departure Hour",
            "DelayRate": "Arrival Delay Rate (%)"
        }
    )
    fig.update_xaxes(dtick=2)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Weekday × Departure Hour Heatmap")

    heat = aggregate(p, ["Weekday", "DepartureHour"])

    grid = heat.pivot(
        index="Weekday",
        columns="DepartureHour",
        values="DelayRate"
    ).reindex(WEEK_ORDER)

    fig = px.imshow(
        grid,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={
            "x": "Departure Hour",
            "y": "Weekday",
            "color": "Delay Rate (%)"
        }
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(available_months) > 1:
        st.subheader("Monthly Pattern")

        monthly = aggregate(patterns, ["Month"])

        fig = px.line(
            monthly.sort_values("Month"),
            x="Month",
            y="DelayRate",
            markers=True
        )
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)


# ===================================================
# TAB 2 — TRAVEL CONDITIONS
# ===================================================

with tabs[1]:

    st.header("How Do Patterns Change by Travel Conditions?")

    c = by_month(conditions, month)

    origin = st.selectbox(
        "Origin Airport",
        sorted(c["Origin"].dropna().unique()),
        key="route_origin"
    )

    destinations = sorted(
        c.loc[c["Origin"] == origin, "Dest"].dropna().unique()
    )

    dest = st.selectbox(
        "Destination Airport",
        destinations,
        key="route_dest"
    )

    route = c[
        (c["Origin"] == origin) &
        (c["Dest"] == dest)
    ].copy()

    airlines = sorted(route["Airline"].dropna().unique())

    selected = st.multiselect(
        "Airlines",
        airlines,
        default=airlines,
        key="route_airlines"
    )

    minimum = st.number_input(
        "Minimum flights per airline/time period",
        min_value=1,
        value=30,
        step=10
    )

    route = route[route["Airline"].isin(selected)]
    summary = aggregate(route, ["Airline", "TimePeriod"])

    if not summary.empty:
        summary = summary[
            summary["Flights"] >= minimum
        ].copy()

    if summary.empty:
        no_data()
    else:
        st.subheader(f"{origin} → {dest}")

        grid = summary.pivot(
            index="Airline",
            columns="TimePeriod",
            values="DelayRate"
        ).reindex(columns=TIME_ORDER)

        grid = grid.dropna(axis=1, how="all")

        fig = px.imshow(
            grid,
            aspect="auto",
            color_continuous_scale="Blues",
            labels={
                "x": "Scheduled Departure Time",
                "y": "Airline",
                "color": "Delay Rate (%)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Time-Based Comparison")

        fig = px.line(
            summary,
            x="TimePeriod",
            y="DelayRate",
            color="Airline",
            markers=True,
            category_orders={
                "TimePeriod": TIME_ORDER
            },
            hover_data=["Flights", "DelayedFlights"],
            labels={
                "TimePeriod": "Departure Time",
                "DelayRate": "Arrival Delay Rate (%)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Groups with fewer than the minimum number "
            "of flights are omitted. Missing cells are "
            "not interpreted as zero delay."
        )

        st.dataframe(summary, hide_index=True)


# ===================================================
# TAB 3 — REPORTED DELAY CAUSES
# ===================================================

with tabs[2]:

    st.header("What Are the Reported Delay Causes?")

    d = by_month(causes, month)

    airport = st.selectbox(
        "Departure Airport",
        ["All Airports"] + sorted(d["Origin"].unique()),
        key="cause_airport"
    )

    if airport != "All Airports":
        d = d[d["Origin"] == airport]

    airline_options = (
        ["All Airlines"]
        + sorted(d["Airline"].dropna().unique())
    )

    airline = st.selectbox(
        "Airline",
        airline_options,
        key="cause_airline"
    )

    if airline != "All Airlines":
        d = d[d["Airline"] == airline]

    if d.empty:
        no_data()
    else:
        st.metric(
            "Flights with Complete Cause Reports",
            f"{d['ReportedFlights'].sum():,.0f}"
        )

        summary = (
            d.groupby("TimePeriod", as_index=False)
            [CAUSE_COLS].sum()
        )

        long = summary.melt(
            id_vars="TimePeriod",
            value_vars=CAUSE_COLS,
            var_name="Cause",
            value_name="Minutes"
        )

        long["Cause"] = long["Cause"].map(CAUSE_LABELS)

        long["TotalMinutes"] = (
            long.groupby("TimePeriod")["Minutes"]
            .transform("sum")
        )

        long = long[long["TotalMinutes"] > 0].copy()

        if long.empty:
            no_data()
        else:
            long["Share"] = (
                long["Minutes"] / long["TotalMinutes"] * 100
            )

            st.subheader("Composition of Reported Delay Minutes")

            fig = px.bar(
                long,
                x="TimePeriod",
                y="Share",
                color="Cause",
                barmode="stack",
                category_orders={
                    "TimePeriod": TIME_ORDER
                },
                hover_data=["Minutes"],
                labels={
                    "Share": "Share of Reported Delay Minutes (%)",
                    "TimePeriod": "Departure Time"
                }
            )
            fig.update_yaxes(range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Reported Minutes by Cause")

            totals = (
                long.groupby("Cause", as_index=False)
                ["Minutes"].sum()
                .sort_values("Minutes")
            )

            fig = px.bar(
                totals,
                x="Minutes",
                y="Cause",
                orientation="h"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These are airline-reported delay categories, "
        "not independently established root causes. "
        "Percentages represent shares of reported minutes, "
        "not percentages of all flights."
    )


# ===================================================
# TAB 4 — DELAY SEVERITY
# ===================================================

with tabs[3]:

    st.header("How Severe Are Delays?")

    sev = by_month(severity, month)

    airport = st.selectbox(
        "Departure Airport",
        ["All Airports"] + sorted(sev["Origin"].unique()),
        key="severity_airport"
    )

    if airport != "All Airports":
        sev = sev[sev["Origin"] == airport]

    airline_options = (
        ["All Airlines"]
        + sorted(sev["Airline"].dropna().unique())
    )

    airline = st.selectbox(
        "Airline",
        airline_options,
        key="severity_airline"
    )

    if airline != "All Airlines":
        sev = sev[sev["Airline"] == airline]

    if sev.empty:
        no_data()
    else:
        a, b, c = st.columns(3)

        a.metric(
            "Median Delay",
            f"{sev['ArrDelay'].median():.1f} min"
        )
        b.metric(
            "90th Percentile",
            f"{sev['ArrDelay'].quantile(0.9):.1f} min"
        )
        c.metric(
            "Delayed Flights",
            f"{len(sev):,}"
        )

        st.subheader("Arrival Delay Distribution")

        hist = sev.copy()
        hist["DisplayDelay"] = (
            hist["ArrDelay"].clip(upper=360)
        )

        fig = px.histogram(
            hist,
            x="DisplayDelay",
            nbins=35,
            labels={
                "DisplayDelay": "Arrival Delay (min)"
            }
        )
        fig.update_layout(yaxis_title="Flights")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "For the histogram, delays above 360 minutes "
            "are grouped into the rightmost value. "
            "Median and P90 use original uncapped values."
        )

        st.subheader("Delay Duration by Airline")

        counts = sev["Airline"].value_counts()
        eligible = counts[counts >= 30].index

        box = sev[sev["Airline"].isin(eligible)].copy()

        if box.empty:
            no_data()
        else:
            # Sample only for visualization performance
            if len(box) > 100000:
                box = box.sample(100000, random_state=42)

            fig = px.box(
                box,
                x="Airline",
                y="ArrDelay",
                points=False,
                labels={
                    "ArrDelay": "Arrival Delay (min)"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Box plot may use a sample of 100,000 delayed "
                "flights for display. The headline metrics "
                "use all filtered observations."
            )


# ===================================================
# TAB 5 — DELAY PROPAGATION
# ===================================================

with tabs[4]:

    st.header("How Are Consecutive Flights Associated?")

    st.write(
        "Explore plausible consecutive flights operated "
        "by the same aircraft, with event times compared "
        "in UTC."
    )

    prop = by_month(propagation, month)

    if prop.empty:
        no_data()
    else:
        st.metric("Linked Flight Pairs", f"{len(prop):,}")

        summary = (
            prop.groupby("InboundLate", as_index=False)
            .agg(
                LinkedPairs=("OutboundLate", "size"),
                OutboundLateRate=("OutboundLate", "mean")
            )
        )

        summary["OutboundLateRate"] *= 100
        summary["PreviousArrival"] = (
            summary["InboundLate"].map({
                True: "Delayed Arrival",
                False: "Not Delayed Arrival"
            })
        )

        st.subheader(
            "Next Departure Delay by Previous Arrival Status"
        )

        fig = px.bar(
            summary,
            x="PreviousArrival",
            y="OutboundLateRate",
            text="OutboundLateRate",
            hover_data=["LinkedPairs"],
            labels={
                "PreviousArrival": "Previous Flight",
                "OutboundLateRate": "Next Departure Late (%)"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )
        fig.update_yaxes(range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(
            "Previous Arrival Delay vs Next Departure Delay"
        )

        scatter = prop
        if len(scatter) > 12000:
            scatter = scatter.sample(
                12000, random_state=42
            )

        fig = px.scatter(
            scatter,
            x="InboundArrDelay",
            y="OutboundDepDelay",
            opacity=0.35,
            hover_data=[
                "ConnectionAirport", "TurnaroundMin"
            ],
            labels={
                "InboundArrDelay": "Previous Arrival Delay (min)",
                "OutboundDepDelay": "Next Departure Delay (min)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Explore an Aircraft's Connections")

        aircraft_options = (
            prop.groupby(["FlightDate", "TailNum"])
            .size()
            .sort_values(ascending=False)
            .head(100)
        )

        if not aircraft_options.empty:
            labels = [
                f"{date} | {tail} | {n} pairs"
                for (date, tail), n in aircraft_options.items()
            ]

            choice = st.selectbox(
                "Aircraft and Departure Date",
                range(len(labels)),
                format_func=lambda i: labels[i]
            )

            selected_date, selected_tail = (
                aircraft_options.index[choice]
            )

            selected = prop[
                (prop["FlightDate"] == selected_date)
                & (prop["TailNum"] == selected_tail)
            ].copy()

            arrivals = selected[
                ["InboundArrUTC", "InboundArrDelay"]
            ].rename(columns={
                "InboundArrUTC": "EventTimeUTC",
                "InboundArrDelay": "DelayMinutes"
            })
            arrivals["Event"] = "Arrival"

            departures = selected[
                ["OutboundDepUTC", "OutboundDepDelay"]
            ].rename(columns={
                "OutboundDepUTC": "EventTimeUTC",
                "OutboundDepDelay": "DelayMinutes"
            })
            departures["Event"] = "Departure"

            timeline = pd.concat(
                [arrivals, departures],
                ignore_index=True
            ).drop_duplicates()

            timeline["EventTimeUTC"] = (
                pd.to_datetime(
                    timeline["EventTimeUTC"],
                    utc=True
                )
            )

            timeline = timeline.sort_values("EventTimeUTC")

            fig = px.scatter(
                timeline,
                x="EventTimeUTC",
                y="DelayMinutes",
                color="Event",
                hover_data=["Event"],
                labels={
                    "EventTimeUTC": "Event Time (UTC)",
                    "DelayMinutes": "Delay (minutes)"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                selected[
                    [
                        "InboundOrigin",
                        "ConnectionAirport",
                        "OutboundDest",
                        "InboundArrDelay",
                        "OutboundDepDelay",
                        "TurnaroundMin"
                    ]
                ],
                hide_index=True,
                use_container_width=True
            )

        st.caption(
            "Aircraft connections are inferred from flight "
            "records and have not been independently verified. "
            "Monthly boundaries and flights with invalid "
            "time records may be excluded. Associations "
            "must not be interpreted as causal proof."
        )

st.divider()
st.caption(
    "Source: BTS Reporting Carrier On-Time Performance, "
    "2025. Historical descriptive analysis."
)
