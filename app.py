"""
app.py — Vercel entry point for the CO₂ & Life Satisfaction Dashboard

Repository layout expected:
  app.py
  requirements.txt
  vercel.json
  data/
    co2-emissions-per-capita.csv
    WHR25_Data_Figure_2.1v3.xlsx

Vercel runs this module as a WSGI app via @vercel/python.
`app = dash_app.server` exposes the Flask WSGI callable.
"""

import os, copy, json as _json, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from dash import Dash, dcc, html, Input, Output, State, no_update

warnings.filterwarnings("ignore")

# data paths 
_HERE     = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_HERE, "data")
CO2_CSV   = os.path.join(_DATA_DIR, "co2-emissions-per-capita.csv")
WHR_XLSX  = os.path.join(_DATA_DIR, "WHR25_Data_Figure_2.1v3.xlsx")

# color palette
CORAL      = "#E8734A"
TEAL       = "#1A9C87"
TEAL_LIGHT = "#A8D5CF"
TEAL_DARK  = "#155F56"
SLATE      = "#2E3B4E"
AMBER      = "#F5A623"
VIOLET     = "#7C3AED"
GREEN      = "#2D8B4E"
GRAY       = "#9CA3AF"
INACTIVE   = "#CCCCCC"

REGION_COLORS = {
    "Africa":       "#11A5C0",
    "Americas":     CORAL,
    "Asia-Pacific": AMBER,
    "Europe":       VIOLET,
    "Middle East":  GREEN,
    "Other":        GRAY,
}
EFFICIENCY_COLORS = {
    "High Efficiency": TEAL,
    "Global Average":  GRAY,
    "Low Efficiency":  CORAL,
}
STATUS_COLORS = {
    "Decoupled": TEAL,
    "Coupled":   CORAL,
    "Both Down": AMBER,
    "Declining": GRAY,
}
PLOT_DEFAULTS = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial", size=12, color=SLATE),
)

# data loading
def _load_data():
    KEEP_COLS = [
        "Year", "Country name", "Life evaluation (3-year average)",
        "Explained by: Log GDP per capita", "Explained by: Social support",
        "Explained by: Healthy life expectancy",
        "Explained by: Freedom to make life choices",
        "Explained by: Generosity", "Explained by: Perceptions of corruption",
    ]
    whr_raw = pd.read_excel(WHR_XLSX)
    whr_raw.columns = [c.strip() if isinstance(c, str) else c for c in whr_raw.columns]
    whr = whr_raw[KEEP_COLS].dropna(subset=["Year", "Country name"]).copy()
    whr["Year"] = whr["Year"].astype(int)
    whr = whr[(whr["Year"] >= 2014) & (whr["Year"] <= 2024)]

    co2_raw = pd.read_csv(CO2_CSV)
    co2_raw.columns = ["country", "iso_code", "Year", "co2_per_capita"]
    co2 = co2_raw.dropna(subset=["iso_code"]).copy()
    co2["iso_code"] = co2["iso_code"].str.strip()
    co2 = co2[(co2["Year"] >= 2014) & (co2["Year"] <= 2024)]

    NAME_MAP = {
        "Cote d'Ivoire": "Cote d'Ivoire",
        "DR Congo": "Democratic Republic of Congo",
        "Hong Kong SAR of China": "Hong Kong",
        "Lao PDR": "Laos",
        "Republic of Korea": "South Korea",
        "Republic of Moldova": "Moldova",
        "Russian Federation": "Russia",
        "State of Palestine": "Palestine",
        "Swaziland": "Eswatini",
        "Taiwan Province of China": "Taiwan",
        "Turkiye": "Turkey",
        "Viet Nam": "Vietnam",
        "North Cyprus": None,
        "Puerto Rico": None,
        "Somaliland Region": None,
    }
    whr["co2_key"] = whr["Country name"].map(lambda x: NAME_MAP.get(x, x))
    whr_clean = whr[whr["co2_key"].notna()].copy()

    merged = whr_clean.merge(
        co2[["country", "iso_code", "Year", "co2_per_capita"]],
        left_on=["co2_key", "Year"],
        right_on=["country", "Year"],
        how="inner",
    ).drop(columns=["country"])
    merged = merged.rename(columns={"Life evaluation (3-year average)": "life_satisfaction"})
    merged["country"] = merged["Country name"]
    merged["year"]    = merged["Year"]

    REGION_MAP = {
        **dict.fromkeys(["Algeria","Angola","Benin","Botswana","Burkina Faso","Burundi",
            "Cameroon","Central African Republic","Chad","Comoros","Congo",
            "Democratic Republic of Congo","Cote d'Ivoire","Djibouti","Egypt",
            "Ethiopia","Eswatini","Gabon","Gambia","Ghana","Guinea","Kenya","Lesotho",
            "Liberia","Libya","Madagascar","Malawi","Mali","Mauritania","Mauritius",
            "Morocco","Mozambique","Namibia","Niger","Nigeria","Rwanda","Senegal",
            "Sierra Leone","Somalia","South Africa","South Sudan","Sudan",
            "Tanzania","Togo","Tunisia","Uganda","Zambia","Zimbabwe"], "Africa"),
        **dict.fromkeys(["Argentina","Belize","Bolivia","Brazil","Canada","Chile",
            "Colombia","Costa Rica","Cuba","Dominican Republic","Ecuador","El Salvador",
            "Guatemala","Guyana","Haiti","Honduras","Jamaica","Mexico","Nicaragua",
            "Panama","Paraguay","Peru","Suriname","Trinidad and Tobago",
            "United States","Uruguay","Venezuela"], "Americas"),
        **dict.fromkeys(["Afghanistan","Australia","Bangladesh","Bhutan","Cambodia",
            "China","Hong Kong","India","Indonesia","Japan","Kazakhstan","Kyrgyzstan",
            "Laos","Malaysia","Maldives","Mongolia","Myanmar","Nepal","New Zealand",
            "Pakistan","Philippines","South Korea","Singapore","Sri Lanka","Taiwan",
            "Tajikistan","Thailand","Turkmenistan","Uzbekistan","Vietnam"], "Asia-Pacific"),
        **dict.fromkeys(["Albania","Armenia","Austria","Azerbaijan","Belarus","Belgium",
            "Bosnia and Herzegovina","Bulgaria","Croatia","Cyprus","Czechia","Denmark",
            "Estonia","Finland","France","Georgia","Germany","Greece","Hungary",
            "Iceland","Ireland","Israel","Italy","Kosovo","Latvia","Lithuania",
            "Luxembourg","Malta","Moldova","Montenegro","Netherlands","North Macedonia",
            "Norway","Poland","Portugal","Romania","Russia","Serbia","Slovakia",
            "Slovenia","Spain","Sweden","Switzerland","Turkey","Ukraine",
            "United Kingdom"], "Europe"),
        **dict.fromkeys(["Bahrain","Iran","Iraq","Jordan","Kuwait","Lebanon",
            "Oman","Qatar","Saudi Arabia","Palestine","Syria",
            "United Arab Emirates","Yemen"], "Middle East"),
    }
    merged["region"] = merged["co2_key"].map(REGION_MAP).fillna("Other")
    return merged


merged = _load_data()


# V1: animated scatter with Gaussian fit (notebook Cells 13 + 14)
def gaussian_logx(x, a, b, c, d):
    """Gaussian in log(x) space — renders as a proper bell on the log x-axis."""
    return a * np.exp(-((np.log(x) - b) ** 2) / (2 * c ** 2)) + d


_v1_df  = merged.dropna(subset=["co2_per_capita", "life_satisfaction"])
_x_data = _v1_df["co2_per_capita"].values
_y_data = _v1_df["life_satisfaction"].values
_log_x  = np.log(_x_data)

_popt, _ = curve_fit(
    gaussian_logx, _x_data, _y_data,
    p0=[max(_y_data) - min(_y_data), np.mean(_log_x), np.std(_log_x), min(_y_data)],
    bounds=([0, _log_x.min(), 0.01, 0], [10, _log_x.max(), 10, max(_y_data)]),
    maxfev=10000,
)
x_smooth_v1 = np.exp(np.linspace(np.log(_x_data.min()), np.log(_x_data.max()), 500))
y_smooth_v1 = gaussian_logx(x_smooth_v1, *_popt)
r2_gauss    = r2_score(_y_data, gaussian_logx(_x_data, *_popt))
pearson_r   = float(np.corrcoef(_x_data, _y_data)[0, 1])
peak_co2    = float(np.exp(_popt[1]))


def build_fig1():
    plot_df = merged.sort_values(["Year", "Country name"]).copy()
    fig = px.scatter(
        plot_df,
        x="co2_per_capita", y="life_satisfaction",
        animation_frame="Year", animation_group="Country name",
        color="region", color_discrete_map=REGION_COLORS,
        hover_name="Country name",
        hover_data={
            "co2_per_capita":    ":.2f",
            "life_satisfaction": ":.2f",
            "region":            True,
            "Year":              False,
            "co2_key":           False,
        },
        labels={
            "co2_per_capita":    "CO2 per Capita (tonnes/person)",
            "life_satisfaction": "Life Satisfaction (0-10)",
            "region":            "Region",
        },
        title="<b>CO2 Emissions vs. Life Satisfaction</b>  (2014-2024)",
        log_x=True, range_x=[0.025, 50], range_y=[0.8, 9],
    )
    fig.update_traces(
        marker=dict(size=9, opacity=0.78, line=dict(width=0.6, color="white")),
        selector=dict(mode="markers"),
    )

    # gaussian overlay put into base figure and every animation frame
    _gt = dict(
        x=x_smooth_v1, y=y_smooth_v1, mode="lines",
        line=dict(color=SLATE, width=2.5, dash="dot"),
        hovertemplate="CO2 = %{x:.2f} t/person<br>Gaussian fit = %{y:.2f}<extra>Gaussian fit</extra>",
    )
    fig.add_trace(go.Scatter(**_gt, name=f"Gaussian fit  (R2={r2_gauss:.2f})", showlegend=True))
    for frame in fig.frames:
        frame.data = tuple(frame.data) + (
            go.Scatter(**_gt, name="Gaussian fit", showlegend=False),
        )

    peak_y_plot = float(gaussian_logx(peak_co2, *_popt))

    fig.update_layout(
        **PLOT_DEFAULTS,
        height=580, width=960,
        margin=dict(l=70, r=210, t=70, b=70),
        legend=dict(
            title=dict(text="<b>Region</b>", font=dict(size=12)),
            x=1.01, y=1, bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E0E0E0", borderwidth=1,
        ),
        annotations=[
            dict(
                text=f"<b>Pearson r = {pearson_r:.2f}</b><br>Gaussian R2 = {r2_gauss:.2f}",
                xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
                bgcolor="rgba(255,255,255,0.88)", bordercolor="#CCCCCC", borderwidth=1,
                font=dict(size=11, family="Arial"),
            ),
            dict(
                text=f"Peak ~ {peak_co2:.1f} t/person",
                x=np.log10(peak_co2), y=peak_y_plot + 0.25,
                xref="x", yref="y",
                showarrow=True, arrowhead=2, arrowcolor=SLATE,
                ax=0, ay=-30, font=dict(size=10, color=SLATE),
            ),
            dict(
                text="<- Industrialisation phase",
                xref="paper", yref="paper", x=0.10, y=0.18,
                showarrow=False, font=dict(size=10, color="#777"),
            ),
            dict(
                text="Diminishing returns ->",
                xref="paper", yref="paper", x=0.70, y=0.95,
                showarrow=False, font=dict(size=10, color="#777"),
            ),
        ],
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F2F2F2", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F2F2F2", zeroline=False)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 900
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 400
    return fig


# V2: region-filtered dual choropleth
REGION_SCOPE = {
    "World":        "world",
    "Africa":       "africa",
    "Americas":     "world",
    "Asia-Pacific": "asia",
    "Europe":       "europe",
    "Middle East":  "asia",
}
CO2_MAX_LOG = float(np.log1p(42))
GEO_BASE = dict(
    showcoastlines=True,  coastlinecolor="#C0C0C0",
    showland=True,        landcolor="#F7F7F7",
    showocean=True,       oceancolor="#E6F2F8",
    showframe=False,      projection_type="natural earth",
    showlakes=False,
)


def _get_year_region_df(yr, region="World"):
    sub = merged[merged["Year"] == yr]
    if region != "World":
        sub = sub[sub["region"] == region]
    return (
        sub.groupby("iso_code", as_index=False)
           .agg(
               life_satisfaction=("life_satisfaction", "mean"),
               co2_per_capita   =("co2_per_capita",    "mean"),
               country_name     =("Country name",      "first"),
           )
           .assign(co2_log=lambda d: np.log1p(d["co2_per_capita"]))
    )


def build_fig2(year=2022, region="World"):
    yd    = _get_year_region_df(year, region)
    scope = REGION_SCOPE.get(region, "world")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}]],
        subplot_titles=[
            f"<b>Life Satisfaction</b>  (0-10) -- {region}, {year}",
            f"<b>CO2 per Capita</b>  (log scale) -- {region}, {year}",
        ],
        horizontal_spacing=0.04,
    )

    fig.add_trace(
        go.Choropleth(
            locations=yd["iso_code"], z=yd["life_satisfaction"],
            text=yd["country_name"],
            colorscale="RdYlGn", zmin=1.0, zmid=5.5, zmax=8.5,
            colorbar=dict(
                title="Score", x=0.455, len=0.72, thickness=14,
                tickvals=[1, 3, 5.5, 7, 8.5], ticktext=["1", "3", "5.5", "7", "8.5"],
            ),
            hovertemplate="<b>%{text}</b><br>Life Satisfaction: %{z:.2f}<extra></extra>",
            showscale=True, geo="geo",
        ), row=1, col=1,
    )

    fig.add_trace(
        go.Choropleth(
            locations=yd["iso_code"], z=yd["co2_log"],
            text=yd["country_name"], customdata=yd["co2_per_capita"],
            colorscale="Blues", zmin=0, zmax=CO2_MAX_LOG,
            colorbar=dict(
                title="t/person", x=1.00, len=0.72, thickness=14,
                tickvals=[np.log1p(v) for v in [0, 1, 3, 8, 20, 42]],
                ticktext=["0", "1", "3", "8", "20", "42"],
            ),
            hovertemplate="<b>%{text}</b><br>CO2: %{customdata:.2f} t/person<extra></extra>",
            showscale=True, geo="geo2",
        ), row=1, col=2,
    )

    fig.update_geos(**{**GEO_BASE, "scope": scope})

    fig.update_layout(
        **PLOT_DEFAULTS,
        height=460, width=1200,
        margin=dict(l=10, r=10, t=70, b=20),
    )
    return fig


# V3: model comparison panel (notebook Cell 19)
FEATURES = [
    "Explained by: Log GDP per capita",
    "Explained by: Social support",
    "Explained by: Healthy life expectancy",
    "Explained by: Freedom to make life choices",
    "Explained by: Generosity",
    "Explained by: Perceptions of corruption",
    "co2_per_capita",
]
FEATURE_LABELS = [
    "Log GDP per capita",
    "Social support",
    "Healthy life expectancy",
    "Freedom of choice",
    "Generosity",
    "Perceptions of corruption",
    "CO2 per capita",
]

model_df = merged[FEATURES + ["life_satisfaction"]].dropna().copy()
X_all    = model_df[FEATURES].values
y_all    = model_df["life_satisfaction"].values
X_co2_   = model_df[["co2_per_capita"]].values


def _fit_poly(deg, X, y):
    pf = PolynomialFeatures(degree=deg)
    Xp = pf.fit_transform(X)
    m  = LinearRegression().fit(Xp, y)
    return r2_score(y, m.predict(Xp))


r2_lin   = _fit_poly(1, X_co2_, y_all)
r2_quad  = _fit_poly(2, X_co2_, y_all)
r2_cub   = _fit_poly(3, X_co2_, y_all)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
multi_m  = LinearRegression().fit(X_scaled, y_all)
r2_multi = r2_score(y_all, multi_m.predict(X_scaled))
coefs    = multi_m.coef_

_rng = np.random.default_rng(42)
_boot = np.array([
    LinearRegression().fit(
        X_scaled[idx := _rng.choice(len(X_scaled), len(X_scaled))], y_all[idx]
    ).coef_
    for _ in range(1000)
])
ci_lo = np.percentile(_boot, 2.5,  axis=0)
ci_hi = np.percentile(_boot, 97.5, axis=0)

_teal_grad       = [TEAL_LIGHT, TEAL, TEAL_DARK]
bar_colors_left  = _teal_grad + [CORAL]
bar_labels_left  = [
    "Linear (CO2 only)", "Quadratic (CO2 only)",
    "Cubic (CO2 only)",  "Multiple Regression",
]
bar_colors_right = [TEAL if abs(c) > 0.05 else CORAL for c in coefs]

MULTI_IDX  = 3
FEAT_START = 4

def build_fig3():
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "<b>R2 by Model</b>",
            "<b>Standardised Regression Coefficients</b><br>"
            '<span style="font-size:11px">Click a bar to toggle -- Multiple R2 updates live</span>',
        ],
        column_widths=[0.40, 0.60], horizontal_spacing=0.14,
    )

    for label, val, color in zip(bar_labels_left,
                                 [r2_lin, r2_quad, r2_cub, r2_multi],
                                 bar_colors_left):
        fig.add_trace(
            go.Bar(
                name=label, x=[label], y=[val],
                marker_color=color, marker_line=dict(color="white", width=1.5),
                text=[f"{val:.2f}"], textposition="outside",
                textfont=dict(size=12, color=SLATE),
                width=0.55, legendgroup=label,
                hovertemplate=f"<b>{label}</b><br>R2 = {val:.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

    fig.add_hline(y=r2_multi, line_dash="dash", line_color=CORAL,
                  line_width=1.8, row=1, col=1)
    fig.add_annotation(
        xref="x", yref="y", x=1.5, y=r2_multi + 0.05,
        text=f"Multiple R2 = {r2_multi:.2f}", showarrow=False,
        font=dict(size=10, color=CORAL, family="Arial"), row=1, col=1,
    )

    for i, (label, c, lo, hi, color) in enumerate(
            zip(FEATURE_LABELS, coefs, ci_lo, ci_hi, bar_colors_right)):
        fig.add_trace(
            go.Bar(
                name=f"feat_{i}", x=[c], y=[label], orientation="h",
                marker_color=color, marker_line=dict(color="white", width=0.8),
                error_x=dict(
                    type="data", symmetric=False,
                    array=[hi - c], arrayminus=[c - lo],
                    color="#666", thickness=1.8, width=5,
                ),
                text=[f"{c:+.3f}"], textposition="outside",
                textfont=dict(size=11, color=SLATE),
                width=0.6,
                hovertemplate=(
                    f"<b>{label}</b><br>beta = {c:.3f}<br>"
                    f"<i>Click to toggle</i><extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.add_vline(x=0,   line_color="#BBBBBB", line_width=1.2, row=1, col=2)
    fig.add_hline(y=3.5, line_dash="dot",  line_color="#555", line_width=1.8, row=1, col=2)
    fig.add_annotation(
        xref="x2", yref="y2", x=0.25, y="CO2 per capita",
        text="<- Small independent<br>effect", showarrow=False,
        font=dict(size=10, color=CORAL, family="Arial"),
    )
    fig.add_annotation(
        xref="paper", yref="y2", x=1.0, y=3.25,
        text="Development<br>indicators", showarrow=False,
        font=dict(size=10, color="#555", family="Arial"),
    )

    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(size=10), row=1, col=1)
    fig.update_yaxes(
        showgrid=True, gridcolor="#F0F0F0", range=[0, 1.12],
        tickfont=dict(size=11), title="R2", row=1, col=1,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#F0F0F0", zeroline=False,
        tickfont=dict(size=10), title="Standardised beta",
        range=[min(coefs) - 0.15, max(coefs) + 0.25], row=1, col=2,
    )
    fig.update_yaxes(
        showgrid=False, tickfont=dict(size=11),
        range=[-0.5, len(FEATURE_LABELS) - 0.5], row=1, col=2,
    )
    fig.update_layout(
        **PLOT_DEFAULTS,
        title=dict(
            text="<b>How Much Does CO2 Independently Explain Life Satisfaction?</b>",
            font=dict(size=15),
        ),
        height=540, width=1120, margin=dict(l=60, r=80, t=100, b=60),
        barmode="overlay",
        legend=dict(
            title=dict(text="<b>Model</b>", font=dict(size=11)),
            x=1.01, y=1, bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E0E0E0", borderwidth=1,
        ),
    )
    return fig


# V4: CO2 efficiency scatter
df_2022 = merged[merged["Year"] == 2022].copy()
df_2022 = df_2022[df_2022["co2_per_capita"] > 0].dropna(
    subset=["life_satisfaction", "co2_per_capita"]
)
df_2022["efficiency"] = df_2022["life_satisfaction"] / df_2022["co2_per_capita"]

_Xe     = np.log10(df_2022[["co2_per_capita"]])
_ye     = df_2022["efficiency"]
reg_eff = LinearRegression().fit(_Xe, _ye)
df_2022["residuals"]    = _ye - reg_eff.predict(_Xe)
df_2022["residual_pct"] = df_2022["residuals"].rank(pct=True) * 100
_ql, _qh = df_2022["residuals"].quantile([0.33, 0.66])


def _get_eff_status(res):
    if res > _qh: return "High Efficiency"
    if res < _ql: return "Low Efficiency"
    return "Global Average"

df_2022["Efficiency Status"] = df_2022["residuals"].apply(_get_eff_status)
avg_eff_val   = float(df_2022["efficiency"].mean())
x_fit_eff     = np.logspace(
    np.log10(df_2022["co2_per_capita"].min()),
    np.log10(df_2022["co2_per_capita"].max()), 300,
)
y_fit_eff     = reg_eff.predict(np.log10(x_fit_eff.reshape(-1, 1)))
EFFICIENCY_OUTLIERS = ["Malawi", "Costa Rica", "Vietnam", "Norway", "United States", "Qatar"]

def build_fig4(min_pct=0):
    filtered = df_2022[df_2022["residual_pct"] >= min_pct].copy()
    fig = px.scatter(
        filtered, x="co2_per_capita", y="efficiency",
        color="Efficiency Status", color_discrete_map=EFFICIENCY_COLORS,
        hover_name="country",
        hover_data={
            "co2_per_capita":    ":.2f",
            "life_satisfaction": ":.2f",
            "efficiency":        ":.3f",
            "residual_pct":      ":.0f",
        },
        labels={
            "co2_per_capita": "CO2 per Capita (tonnes/person, log scale)",
            "efficiency":     "Efficiency (Life Satisfaction / CO2)",
        },
        title=(
            f"<b>CO2 Efficiency vs. Life Satisfaction</b>  (2022)"
            f"<br><sup>Showing countries above the {min_pct:.0f}th efficiency percentile"
            f" -- {len(filtered)} of {len(df_2022)} countries</sup>"
        ),
        log_x=True, height=580,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0.6, color="white")))

    fig.add_trace(go.Scatter(
        x=x_fit_eff, y=y_fit_eff, mode="lines",
        name="Expected efficiency",
        line=dict(color=SLATE, width=1.8, dash="dash"),
        hovertemplate="CO2 = %{x:.2f}<br>Expected eff = %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(
        y=avg_eff_val, line_dash="dot", line_color="#888", line_width=1.5,
        annotation_text=f"Global avg = {avg_eff_val:.2f}",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color="#666"),
    )

    for _, row in filtered[filtered["country"].isin(EFFICIENCY_OUTLIERS)].iterrows():
        fig.add_annotation(
            x=row["co2_per_capita"], y=row["efficiency"],
            text=row["country"], showarrow=False,
            font=dict(size=9, color=SLATE),
            xanchor="left", yanchor="bottom", xshift=8,
        )

    fig.update_layout(
        **PLOT_DEFAULTS,
        margin=dict(l=70, r=160, t=90, b=60),
        legend=dict(
            title=dict(text="<b>Efficiency Status</b>", font=dict(size=11)),
            x=1.01, y=1, bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E0E0E0", borderwidth=1,
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F2F2F2", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#F2F2F2", zeroline=False)
    return fig


# V5: 10-Year decoupling trends
df_decade = (
    merged[merged["Year"].between(2014, 2024)]
    .dropna(subset=["co2_per_capita", "life_satisfaction"])
    .copy()
)

def _process_trends(group):
    group  = group.sort_values("year")
    start, end = group.iloc[0], group.iloc[-1]
    group["Carbon Index"]    = (group["co2_per_capita"]    / start["co2_per_capita"])    * 100
    group["Happiness Index"] = (group["life_satisfaction"] / start["life_satisfaction"]) * 100
    h_up   = end["life_satisfaction"] > start["life_satisfaction"]
    c_down = end["co2_per_capita"]    < start["co2_per_capita"]
    if   h_up and c_down:      status = "Decoupled"
    elif h_up and not c_down:  status = "Coupled"
    elif not h_up and c_down:  status = "Both Down"
    else:                      status = "Declining"
    group["Status"]      = status
    group["Facet_Title"] = f"{start['country']} | {status}"
    return group

df_indexed = df_decade.groupby("country", group_keys=False).apply(_process_trends)
top_emitters_global = (
    df_indexed[df_indexed["year"] == 2022]
    .nlargest(12, "co2_per_capita")["country"]
    .unique()
)

def build_fig5(countries):
    df_sub    = df_indexed[df_indexed["country"].isin(countries)]
    df_melted = df_sub.melt(
        id_vars=["year", "Facet_Title", "Status", "country"],
        value_vars=["Carbon Index", "Happiness Index"],
        var_name="Metric", value_name="Score",
    )
    fig = px.line(
        df_melted, x="year", y="Score", color="Metric",
        facet_col="Facet_Title", facet_col_wrap=4, markers=True,
        title="<b>10-Year Decoupling Trends</b>  (2014 = 100)",
        color_discrete_map={"Carbon Index": CORAL, "Happiness Index": TEAL},
        height=820,
        hover_data={
            "year": True, "Score": ":.1f", "Metric": False,
            "Facet_Title": False, "Status": False, "country": False,
        },
    )

    show_both      = [True]  * len(fig.data)
    show_carbon    = [t.name == "Carbon Index"    for t in fig.data]
    show_happiness = [t.name == "Happiness Index" for t in fig.data]

    fig.update_layout(
        **PLOT_DEFAULTS,
        title=dict(x=0.5, y=0.97, xanchor="center", font=dict(size=16)),
        margin=dict(t=160, b=60, r=180, l=80),
        legend=dict(
            title=dict(text="<b>Metric</b>"),
            x=1.02, y=1, bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E0E0E0", borderwidth=1,
        ),
        updatemenus=[dict(
            type="dropdown", direction="down", x=0.0, y=1.10, showactive=True,
            bgcolor="white", bordercolor="#E0E0E0",
            font=dict(family="Arial", color=SLATE),
            buttons=[
                dict(label="Compare Both",   method="update", args=[{"visible": show_both}]),
                dict(label="Carbon Only",    method="update", args=[{"visible": show_carbon}]),
                dict(label="Happiness Only", method="update", args=[{"visible": show_happiness}]),
            ],
        )],
    )

    fig.update_yaxes(
        title_text="", matches=None,
        showgrid=True, gridcolor="#F2F2F2", zeroline=False,
    )
    fig.update_xaxes(
        showgrid=False, dtick=2,
        tickvals=list(range(2014, 2025, 2)),
        ticktext=[str(y) for y in range(2014, 2025, 2)],
    )
    fig.add_annotation(
        text="Index (2014 = 100)", xref="paper", yref="paper",
        x=-0.055, y=0.5, textangle=-90, showarrow=False,
        font=dict(size=12, color=SLATE, family="Arial"),
    )
    fig.add_hline(y=100, line_dash="dot", line_color="#BBBBBB", line_width=1.2)

    def _colour_ann(ann):
        raw    = ann.text.split("=")[-1]
        parts  = [p.strip() for p in raw.split("|")]
        cname  = parts[0]
        status = parts[1] if len(parts) > 1 else ""
        ann.update(
            text=f"<b>{cname}</b>",
            font=dict(size=11, color=STATUS_COLORS.get(status, SLATE)),
        )

    fig.for_each_annotation(_colour_ann)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DASH APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
dash_app = Dash(__name__, suppress_callback_exceptions=True)
app = dash_app.server   # flask WSGI callable for vercel

YEARS_ALL   = sorted(merged["Year"].unique().tolist())
REGIONS_ALL = ["World"] + sorted([r for r in merged["region"].unique() if r != "Other"])

_tab  = dict(padding="8px 18px", fontFamily="Arial", fontSize="13px",
             color=SLATE, backgroundColor="white", borderBottom="2px solid #E8E8E8")
_tsel = dict(padding="8px 18px", fontFamily="Arial", fontSize="13px",
             fontWeight="bold", color=TEAL, backgroundColor="white",
             borderTop=f"3px solid {TEAL}", borderBottom="2px solid white")
_desc = dict(fontFamily="Arial", fontSize="12px", color="#666",
             marginBottom="10px", lineHeight="1.5")
_lbl  = dict(fontFamily="Arial", fontSize="12px", color=SLATE, marginRight="8px")

# base button style — color swapped in callbacks for play/pause state
_BTN_BASE = dict(
    fontFamily="Arial", fontSize="12px", fontWeight="bold",
    color="white", border="none", borderRadius="4px",
    padding="6px 16px", cursor="pointer", marginRight="8px",
)
_BTN_PLAY  = {**_BTN_BASE, "backgroundColor": TEAL}
_BTN_PAUSE = {**_BTN_BASE, "backgroundColor": CORAL}


def _status_legend_div():
    items = []
    for status, color in STATUS_COLORS.items():
        items.append(html.Span([
            html.Span("*", style={"color": color, "fontSize": "18px",
                                  "marginRight": "4px", "lineHeight": "1"}),
            html.Span(status, style={"marginRight": "16px"}),
        ]))
    return html.Div(
        [html.Span("Facet title colour: ",
                   style={"fontWeight": "bold", "marginRight": "8px"})] + items,
        style={"fontFamily": "Arial", "fontSize": "12px", "color": SLATE,
               "marginBottom": "10px", "display": "flex", "alignItems": "center",
               "flexWrap": "wrap", "gap": "4px"},
    )


dash_app.layout = html.Div([
    html.Div([
        html.H2("CO2 Emissions & Life Satisfaction",
                style={"margin": "0", "color": "white", "fontFamily": "Arial",
                       "fontSize": "22px", "fontWeight": "bold"}),
        html.P("Cross-National Analysis  |  2014-2024  |  Group 12",
               style={"margin": "4px 0 0 0", "color": "rgba(255,255,255,0.75)",
                      "fontFamily": "Arial", "fontSize": "13px"}),
    ], style={"backgroundColor": TEAL, "padding": "18px 32px"}),

    dcc.Tabs(id="tabs", value="v1", style={"borderBottom": "2px solid #E8E8E8"}, children=[
        dcc.Tab(label="1 - Development Paradox", value="v1", style=_tab, selected_style=_tsel),
        dcc.Tab(label="2 - Regional Patterns",   value="v2", style=_tab, selected_style=_tsel),
        dcc.Tab(label="3 - Model Comparison",    value="v3", style=_tab, selected_style=_tsel),
        dcc.Tab(label="4 - CO2 Efficiency",      value="v4", style=_tab, selected_style=_tsel),
        dcc.Tab(label="5 - 10-Year Trends",      value="v5", style=_tab, selected_style=_tsel),
    ]),

    html.Div(id="tab-content", style={"backgroundColor": "white", "padding": "20px 24px"}),
    dcc.Store(id="v3-mask", data=_json.dumps([True] * len(FEATURES))),

], style={"backgroundColor": "white", "minHeight": "100vh"})


# tab renderer
@dash_app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):

    if tab == "v1":
        return html.Div([
            html.P(
                "Animated scatter with 2D Gaussian fit in log(x) space, capturing the "
                "inverted-U / diminishing-returns relationship. Play the animation or "
                "drag the year slider.", style=_desc,
            ),
            dcc.Graph(figure=build_fig1(), config={"displayModeBar": True},
                      style={"height": "600px"}),
        ])

    if tab == "v2":
        return html.Div([
            html.P(
                "Region-filtered dual choropleth. Select a region to zoom both maps "
                "simultaneously. Use the Play button or drag the year slider to animate "
                "through 2014-2024.", style=_desc,
            ),

            # controls row
            html.Div([
                # region dropdown
                html.Div([
                    html.Label("Region:", style=_lbl),
                    dcc.Dropdown(
                        id="v2-region",
                        options=[{"label": r, "value": r} for r in REGIONS_ALL],
                        value="World", clearable=False,
                        style={"width": "180px", "fontFamily": "Arial", "fontSize": "12px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "marginRight": "20px"}),

                # play / pause button (teal = playing stopped, coral = playing)
                html.Button("Play", id="v2-play-btn", n_clicks=0, style=_BTN_PLAY),

                # year slider
                html.Div([
                    html.Label("Year:", style=_lbl),
                    dcc.Slider(
                        id="v2-year",
                        min=YEARS_ALL[0], max=YEARS_ALL[-1], step=1, value=2022,
                        marks={y: str(y) for y in YEARS_ALL},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                ], style={"flex": "1", "display": "flex", "alignItems": "center"}),

            ], style={"display": "flex", "alignItems": "center",
                      "marginBottom": "12px", "maxWidth": "980px"}),

            # interval fires every 1.1 s while playing
            dcc.Interval(id="v2-interval", interval=1100, n_intervals=0, disabled=True),

            # store: True = currently playing
            dcc.Store(id="v2-playing", data=False),

            dcc.Graph(id="v2-graph", figure=build_fig2(2022, "World"),
                      config={"displayModeBar": True}, style={"height": "480px"}),
        ])

    if tab == "v3":
        return html.Div([
            html.P(
                "Click any coefficient bar (right panel) to toggle that feature on/off. "
                "The Multiple Regression R2 bar and reference line update live.", style=_desc,
            ),
            dcc.Store(id="v3-mask-local", data=_json.dumps([True] * len(FEATURES))),
            dcc.Graph(id="fig3-graph", figure=build_fig3(),
                      config={"displayModeBar": True}, style={"height": "580px"}),
        ])

    if tab == "v4":
        return html.Div([
            html.P(
                "Efficiency = Life Satisfaction / CO2 per capita (2022). "
                "Points above the dashed regression line over-perform expected efficiency. "
                "Drag the slider to filter out lower-efficiency countries.", style=_desc,
            ),
            html.Div([
                html.Label("Minimum efficiency percentile:", style=_lbl),
                dcc.Slider(
                    id="eff-slider", min=0, max=90, step=5, value=0,
                    marks={i: f"{i}%" for i in range(0, 91, 10)},
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ], style={"maxWidth": "700px", "marginBottom": "12px"}),
            dcc.Graph(id="fig4-graph", figure=build_fig4(0),
                      config={"displayModeBar": True}, style={"height": "600px"}),
        ])

    if tab == "v5":
        return html.Div([
            html.P(
                "Base-year indexed trends (2014 = 100) for top emitters in the selected region. "
                "Teal = Happiness Index, Coral = Carbon Index.", style=_desc,
            ),
            _status_legend_div(),
            html.Div([
                html.Label("Filter by region:", style=_lbl),
                dcc.Dropdown(
                    id="v5-region",
                    options=["All"] + REGIONS_ALL,
                    value="All", clearable=False,
                    style={"width": "220px", "fontFamily": "Arial", "fontSize": "12px"},
                ),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"}),
            dcc.Graph(id="fig5-graph", figure=build_fig5(top_emitters_global),
                      config={"displayModeBar": True}, style={"height": "840px"}),
        ])

    return html.Div("Select a tab.", style={"fontFamily": "Arial", "color": SLATE})


# V2: play / pause toggle
@dash_app.callback(
    Output("v2-playing",  "data"),
    Output("v2-interval", "disabled"),
    Output("v2-play-btn", "children"),
    Output("v2-play-btn", "style"),
    Input("v2-play-btn",  "n_clicks"),
    State("v2-playing",   "data"),
    prevent_initial_call=True,
)
def toggle_play(n_clicks, is_playing):
    new_playing = not is_playing
    if new_playing:
        return True, False, "Pause", _BTN_PAUSE
    else:
        return False, True, "Play", _BTN_PLAY


# V2: advance year on each interval tick
@dash_app.callback(
    Output("v2-year",     "value"),
    Output("v2-interval", "disabled",  allow_duplicate=True),
    Output("v2-playing",  "data",      allow_duplicate=True),
    Output("v2-play-btn", "children",  allow_duplicate=True),
    Output("v2-play-btn", "style",     allow_duplicate=True),
    Input("v2-interval",  "n_intervals"),
    State("v2-year",      "value"),
    State("v2-playing",   "data"),
    prevent_initial_call=True,
)
def advance_year(n_intervals, current_year, is_playing):
    if not is_playing:
        return no_update, True, False, "Play", _BTN_PLAY

    next_year = current_year + 1

    # stop at the end of the timeline and reset the button
    if next_year > YEARS_ALL[-1]:
        return YEARS_ALL[-1], True, False, "Play", _BTN_PLAY

    return next_year, False, True, "Pause", _BTN_PAUSE


# V2: rebuild map on region or year change
@dash_app.callback(
    Output("v2-graph", "figure"),
    Input("v2-region", "value"),
    Input("v2-year",   "value"),
)
def update_v2(region, year):
    return build_fig2(year, region)


# V3: feature toggle
@dash_app.callback(
    Output("fig3-graph",    "figure"),
    Output("v3-mask-local", "data"),
    Input("fig3-graph",     "clickData"),
    State("fig3-graph",     "figure"),
    State("v3-mask-local",  "data"),
    prevent_initial_call=True,
)
def toggle_feature(click_data, current_fig, mask_json):
    if click_data is None:
        return no_update, no_update

    curve_num = click_data["points"][0]["curveNumber"]
    if curve_num < FEAT_START:
        return no_update, no_update

    feat_idx    = curve_num - FEAT_START
    active_mask = _json.loads(mask_json)
    active_mask[feat_idx] = not active_mask[feat_idx]

    active_idx = [i for i, on in enumerate(active_mask) if on]
    new_r2 = (
        r2_score(
            y_all,
            LinearRegression()
            .fit(X_scaled[:, active_idx], y_all)
            .predict(X_scaled[:, active_idx]),
        )
        if active_idx else 0.0
    )

    fig = copy.deepcopy(current_fig)
    t   = fig["data"][curve_num]
    t["marker"]["color"]   = bar_colors_right[feat_idx] if active_mask[feat_idx] else INACTIVE
    t["marker"]["opacity"] = 1.0 if active_mask[feat_idx] else 0.45

    fig["data"][MULTI_IDX]["y"]    = [new_r2]
    fig["data"][MULTI_IDX]["text"] = [f"{new_r2:.2f}"]
    fig["data"][MULTI_IDX]["hovertemplate"] = (
        f"<b>Multiple Regression</b><br>R2 = {new_r2:.3f}<extra></extra>"
    )
    fig["layout"]["shapes"][0]["y0"] = new_r2
    fig["layout"]["shapes"][0]["y1"] = new_r2

    for ann in fig["layout"]["annotations"]:
        if ann.get("text", "").startswith("Multiple R2"):
            ann["text"] = f"Multiple R2 = {new_r2:.2f}"
            ann["y"]    = new_r2 + 0.05
            break

    return fig, _json.dumps(active_mask)


# V4: efficiency percentile slider
@dash_app.callback(
    Output("fig4-graph", "figure"),
    Input("eff-slider",  "value"),
    prevent_initial_call=True,
)
def update_v4(min_pct):
    return build_fig4(min_pct)


# V5: region dropdown
@dash_app.callback(
    Output("fig5-graph", "figure"),
    Input("v5-region",   "value"),
    prevent_initial_call=True,
)
def update_v5(selected_region):
    if selected_region in (None, "All", "World"):
        countries = top_emitters_global
    else:
        pool = (
            df_indexed[
                (df_indexed["year"]   == 2022) &
                (df_indexed["region"] == selected_region)
            ]
            .nlargest(12, "co2_per_capita")["country"]
            .unique()
        )
        countries = pool if len(pool) > 0 else top_emitters_global
    return build_fig5(countries)


if __name__ == "__main__":
    dash_app.run(debug=True)