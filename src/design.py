# %%
import altair as alt
import folium
import pandas as pd
from folium.plugins import MarkerCluster
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

alt.data_transformers.enable("vegafusion")
alt.data_transformers.enable("default", max_rows=None)

# %%
trips_df = pd.read_csv("../data/cleaned_data/metro_trips.csv")
trips_df = trips_df.astype(
    {
        "trip_id": "int64",
        "bicycle_id": "object",
        "bike_type": "category",
        "pass_type": "category",
        "duration": "int64",
        "checkout_kiosk_id": "category",
        "return_kiosk_id": "category",
    }
)
trips_df["checkout_time"] = pd.to_datetime(trips_df["checkout_time"])
trips_df["checkout_hour"] = trips_df["checkout_hour"].astype("int32")

# %%
weather_df = pd.read_csv("../data/cleaned_data/weather.csv")
weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.round("H")
weather_df = weather_df.astype(
    {
        "temp": "float64",
        "precipitation": "float64",
        "humidity": "float64",
        "visibility": "float64",
        "wind_speed": "float64",
    }
)

# %%
kiosks_df = pd.read_csv("../data/cleaned_data/kiosk.csv")
kiosks_df = kiosks_df.astype(
    {
        "kiosk_id": "int64",
        "kiosk_name": "object",
        "status": "category",
        "address": "object",
        "latitude": "float64",
        "longitude": "float64",
    }
)

# %%
# seasonal ridership trends
trips_2023 = trips_df[trips_df["checkout_time"].dt.year == 2023]

rides_per_day = (
    trips_2023.groupby(trips_2023["checkout_time"].dt.date)
    .size()
    .reset_index(name="rides")
)
rides_per_day["checkout_time"] = pd.to_datetime(rides_per_day["checkout_time"])

chart = (
    alt.Chart(rides_per_day)
    .mark_line(point=True)
    .encode(
        x=alt.X(
            "checkout_time:T",
            title="Date",
            axis=alt.Axis(format="%b %Y", labelAngle=45),
        ),
        y=alt.Y("rides:Q", title="Number of Rides"),
        tooltip=[
            alt.Tooltip("checkout_time:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("rides:Q", title="Rides"),
        ],
    )
    .properties(title="Number of Rides per Day in 2023")
    .interactive()
)

chart

# %%
# weather impact
weather_2023 = weather_df[weather_df["date"].dt.year == 2023].copy()
weather_2023["date"] = weather_2023["date"].dt.date

daily_temp = weather_2023.groupby("date")["temp"].mean().reset_index()
rides_per_day = trips_2023.groupby("checkout_date").size().reset_index(name="rides")

daily_temp["date"] = pd.to_datetime(daily_temp["date"])
rides_per_day["checkout_date"] = pd.to_datetime(rides_per_day["checkout_date"])

merged_df = pd.merge(
    rides_per_day, daily_temp, left_on="checkout_date", right_on="date", how="inner"
)

# Use a single date column
merged_df["date"] = pd.to_datetime(merged_df["checkout_date"])

# Brush selection
brush = alt.selection_interval(encodings=["x"])

# Rides chart
rides_chart = (
    alt.Chart(merged_df)
    .mark_line(point=True, color="blue")
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("rides:Q", title="Number of Rides"),
        tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), alt.Tooltip("rides:Q")],
    )
    .add_selection(brush)
    .properties(width=600, height=200, title="Daily Rides")
)

# Temperature chart filtered by brush
temp_chart = (
    alt.Chart(merged_df)
    .mark_line(point=True, color="red")
    .encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("temp:Q", title="Average Temperature (°F)"),
        tooltip=[
            alt.Tooltip("date:T", format="%Y-%m-%d"),
            alt.Tooltip("temp:Q", title="Temperature"),
        ],
    )
    .transform_filter(brush)
    .properties(width=600, height=200, title="Daily Temperature")
)

rides_chart & temp_chart

# %%
# station accessibility
rides_per_kiosk = trips_2023.groupby("checkout_kiosk").size().reset_index(name="rides")

# Merge with kiosk coordinates
kiosk_map_df = rides_per_kiosk.merge(
    kiosks_df, left_on="checkout_kiosk", right_on="kiosk_name", how="inner"
)
m = folium.Map(
    location=[kiosk_map_df["latitude"].mean(), kiosk_map_df["longitude"].mean()],
    zoom_start=12,
)

# Add a marker cluster for better visualization
marker_cluster = MarkerCluster().add_to(m)

# Add kiosks to the map
for idx, row in kiosk_map_df.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5 + row["rides"] ** 0.5,  # radius proportional to ridership
        color="blue",
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['kiosk_name']}<br>Rides: {row['rides']}",
    ).add_to(marker_cluster)

# Display map
m

# %%
semester_periods = [
    {"start": "2023-01-15", "end": "2023-05-05", "name": "Spring 2023"},
    {"start": "2023-08-25", "end": "2023-12-10", "name": "Fall 2023"},
]

for sem in semester_periods:
    sem["start"] = pd.to_datetime(sem["start"])
    sem["end"] = pd.to_datetime(sem["end"])

trips_2023["checkout_date"] = trips_2023["checkout_time"].dt.date
trips_2023["checkout_date"] = pd.to_datetime(trips_2023["checkout_date"])


def in_semester(date):
    for sem in semester_periods:
        if sem["start"] <= date <= sem["end"]:
            return True
    return False


rides_per_day = trips_2023.groupby("checkout_date").size().reset_index(name="rides")
rides_per_day["in_semester"] = rides_per_day["checkout_date"].apply(in_semester)


boxplot = (
    alt.Chart(rides_per_day)
    .mark_boxplot(size=80)
    .encode(
        x=alt.X(
            "in_semester:N",
            title="Period",
            axis=alt.Axis(labels=True, labelAngle=0),
        ),
        y=alt.Y("rides:Q", title="Daily Ridership"),
        color=alt.Color(
            "in_semester:N",
            title="In Semester",
            scale=alt.Scale(domain=[True, False], range=["#2E8B57", "#808080"]),
            legend=None,
        ),
    )
    .properties(
        title="Ridership Distribution: Semester vs. Break", width=400, height=300
    )
)

boxplot

# %%
event_periods = [
    {"start": "2023-03-10", "end": "2023-03-19", "name": "SXSW"},
    {"start": "2023-10-06", "end": "2023-10-15", "name": "ACL Festival"},
    {"start": "2023-11-23", "end": "2023-11-26", "name": "Thanksgiving"},
    {"start": "2023-12-22", "end": "2023-12-31", "name": "Winter Holidays"},
]

for ev in event_periods:
    ev["start"] = pd.to_datetime(ev["start"])
    ev["end"] = pd.to_datetime(ev["end"])

# --- Prepare data ---
trips_2023["checkout_date"] = pd.to_datetime(trips_2023["checkout_time"].dt.date)

# (Optional) Filter to near-campus zones if available
# near_campus_zones = ["Campus West", "Campus East", "Main Quad", "Stadium"]
# trips_2023 = trips_2023[trips_2023["zone"].isin(near_campus_zones)]


# --- Label each day by event/holiday period ---
def get_event_label(date):
    for ev in event_periods:
        if ev["start"] <= date <= ev["end"]:
            return ev["name"]
    return "Non-Event"


# --- Aggregate daily ridership ---
rides_per_day = trips_2023.groupby("checkout_date").size().reset_index(name="rides")
rides_per_day["event_period"] = rides_per_day["checkout_date"].apply(get_event_label)

# --- Interactive Boxplot ---
chart = (
    alt.Chart(rides_per_day)
    .mark_boxplot(size=80)
    .encode(
        x=alt.X(
            "event_period:N",
            title="Event / Holiday Period",
            sort=[
                "SXSW",
                "ACL Festival",
                "Thanksgiving",
                "Winter Holidays",
                "Non-Event",
            ],
        ),
        y=alt.Y("rides:Q", title="Daily Ridership (Near Campus)"),
        color=alt.Color(
            "event_period:N",
            legend=None,
            scale=alt.Scale(
                domain=[
                    "SXSW",
                    "ACL Festival",
                    "Thanksgiving",
                    "Winter Holidays",
                    "Non-Event",
                ],
                range=["#FF7F0E", "#1F77B4", "#9467BD", "#2CA02C", "#808080"],
            ),
        ),
        tooltip=["event_period:N", "rides:Q"],
    )
    .properties(
        title="Impact of Major Events and Holidays on Ridership Near Campus",
        width=500,
        height=300,
    )
    .interactive()
)

chart
# %%
from sklearn.pipeline import Pipeline

# Define features and label
df_feat = rides_per_day[["event_period"]]
df_lab = rides_per_day["rides"]

# Create a pipeline that encodes categorical data, then fits the linear model
lin_model = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(drop="first", sparse_output=False)),
        ("regressor", LinearRegression()),
    ]
)

# Fit the model
lin_model_fitted = lin_model.fit(df_feat, df_lab)


# %%
# Get encoded feature names
feature_names = lin_model.named_steps["encoder"].get_feature_names_out(["event_period"])

# Combine feature names and coefficients
coef_df = pd.DataFrame(
    {"Event": feature_names, "Coefficient": lin_model.named_steps["regressor"].coef_}
)

coef_df
# %%
