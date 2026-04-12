import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
import copy, json, warnings
warnings.filterwarnings('ignore')

# World Happiness Report data
whr_raw = pd.read_excel('WHR25_Data_Figure_2.1v3.xlsx')
whr_raw.columns = [c.strip() if isinstance(c, str) else c for c in whr_raw.columns]

KEEP_COLS = [
    'Year', 'Country name', 'Life evaluation (3-year average)',
    'Explained by: Log GDP per capita', 'Explained by: Social support',
    'Explained by: Healthy life expectancy',
    'Explained by: Freedom to make life choices',
    'Explained by: Generosity', 'Explained by: Perceptions of corruption',
]
whr = whr_raw[KEEP_COLS].dropna(subset=['Year', 'Country name']).copy()
whr['Year'] = whr['Year'].astype(int)
whr = whr[(whr['Year'] >= 2014) & (whr['Year'] <= 2024)]
print(f"WHR: {whr.shape[0]:,} rows | {whr['Country name'].nunique()} countries")
whr.head(3)

# Out World in Data: CO2 per Capita
co2_raw = pd.read_csv('co2-emissions-per-capita.csv')
co2_raw.columns = ['country', 'iso_code', 'Year', 'co2_per_capita']
co2 = co2_raw.dropna(subset=['iso_code']).copy()
co2['iso_code'] = co2['iso_code'].str.strip()
co2 = co2[(co2['Year'] >= 2014) & (co2['Year'] <= 2024)]
print(f"CO₂: {co2.shape[0]:,} rows | {co2['country'].nunique()} countries")
co2.head(3)

# Country name fix
NAME_MAP = {
    "Côte d'Ivoire"          : "Cote d'Ivoire",
    'DR Congo'                : 'Democratic Republic of Congo',
    'Hong Kong SAR of China'  : 'Hong Kong',
    'Lao PDR'                 : 'Laos',
    'Republic of Korea'       : 'South Korea',
    'Republic of Moldova'     : 'Moldova',
    'Russian Federation'      : 'Russia',
    'State of Palestine'      : 'Palestine',
    'Swaziland'               : 'Eswatini',
    'Taiwan Province of China': 'Taiwan',
    'Türkiye'                 : 'Turkey',
    'Viet Nam'                : 'Vietnam',
    'North Cyprus'            : None,   # no CO2 record
    'Puerto Rico'             : None,
    'Somaliland Region'       : None,
}

whr['co2_key'] = whr['Country name'].map(lambda x: NAME_MAP.get(x, x))
whr_clean = whr[whr['co2_key'].notna()].copy()

merged = whr_clean.merge(
    co2[['country', 'iso_code', 'Year', 'co2_per_capita']],
    left_on=['co2_key', 'Year'],
    right_on=['country', 'Year'],
    how='inner',
).drop(columns=['country'])

merged = merged.rename(columns={'Life evaluation (3-year average)': 'life_satisfaction'})

merged['country'] = merged['Country name']
merged['year']    = merged['Year']

print(f"Merged: {merged.shape[0]:,} rows | {merged['Country name'].nunique()} countries | "
      f"{merged['Year'].nunique()} years")
merged[['country', 'year', 'co2_per_capita', 'life_satisfaction', 'iso_code']].head(3)

# World to region mappng
REGION_MAP = {
    **dict.fromkeys(['Algeria','Angola','Benin','Botswana','Burkina Faso','Burundi',
        'Cameroon','Central African Republic','Chad','Comoros','Congo',
        'Democratic Republic of Congo',"Cote d'Ivoire",'Djibouti','Egypt',
        'Ethiopia','Eswatini','Gabon','Gambia','Ghana','Guinea','Kenya','Lesotho',
        'Liberia','Libya','Madagascar','Malawi','Mali','Mauritania','Mauritius',
        'Morocco','Mozambique','Namibia','Niger','Nigeria','Rwanda','Senegal',
        'Sierra Leone','Somalia','South Africa','South Sudan','Sudan',
        'Tanzania','Togo','Tunisia','Uganda','Zambia','Zimbabwe'], 'Africa'),
    **dict.fromkeys(['Argentina','Belize','Bolivia','Brazil','Canada','Chile',
        'Colombia','Costa Rica','Cuba','Dominican Republic','Ecuador','El Salvador',
        'Guatemala','Guyana','Haiti','Honduras','Jamaica','Mexico','Nicaragua',
        'Panama','Paraguay','Peru','Suriname','Trinidad and Tobago',
        'United States','Uruguay','Venezuela'], 'Americas'),
    **dict.fromkeys(['Afghanistan','Australia','Bangladesh','Bhutan','Cambodia',
        'China','Hong Kong','India','Indonesia','Japan','Kazakhstan','Kyrgyzstan',
        'Laos','Malaysia','Maldives','Mongolia','Myanmar','Nepal','New Zealand',
        'Pakistan','Philippines','South Korea','Singapore','Sri Lanka','Taiwan',
        'Tajikistan','Thailand','Turkmenistan','Uzbekistan','Vietnam'], 'Asia-Pacific'),
    **dict.fromkeys(['Albania','Armenia','Austria','Azerbaijan','Belarus','Belgium',
        'Bosnia and Herzegovina','Bulgaria','Croatia','Cyprus','Czechia','Denmark',
        'Estonia','Finland','France','Georgia','Germany','Greece','Hungary',
        'Iceland','Ireland','Israel','Italy','Kosovo','Latvia','Lithuania',
        'Luxembourg','Malta','Moldova','Montenegro','Netherlands','North Macedonia',
        'Norway','Poland','Portugal','Romania','Russia','Serbia','Slovakia',
        'Slovenia','Spain','Sweden','Switzerland','Turkey','Ukraine',
        'United Kingdom'], 'Europe'),
    **dict.fromkeys(['Bahrain','Iran','Iraq','Jordan','Kuwait','Lebanon',
        'Oman','Qatar','Saudi Arabia','Palestine','Syria',
        'United Arab Emirates','Yemen'], 'Middle East'),
}

merged['region'] = merged['co2_key'].map(REGION_MAP).fillna('Other')
print("Region distribution:")
print(merged['region'].value_counts())
merged[['country','year','co2_per_capita','life_satisfaction','iso_code','region']].describe()

# Color palette to use for all visuals
CORAL       = '#E8734A'   # carbon / high emissions / low efficiency / warm trend
TEAL        = '#1A9C87'   # life satisfaction / high efficiency / cool trend
TEAL_LIGHT  = '#A8D5CF'
TEAL_DARK   = '#155F56'
SLATE       = '#2E3B4E'   # primary text & axes
AMBER       = '#F5A623'
VIOLET      = '#7C3AED'
GREEN       = '#2D8B4E'
GRAY        = '#9CA3AF'
INACTIVE    = '#CCCCCC'   # toggled-off elements in V3

# Region colors (V1, V2, V4)
REGION_COLORS = {
    'Africa'      : '#11A5C0',
    'Americas'    : CORAL,
    'Asia-Pacific': AMBER,
    'Europe'      : VIOLET,
    'Middle East' : GREEN,
    'Other'       : GRAY,
}

# Efficiency status colors (V4) — aligned to CORAL/TEAL axis
EFFICIENCY_COLORS = {
    'High Efficiency': TEAL,
    'Global Average' : GRAY,
    'Low Efficiency' : CORAL,
}

# Decoupling status colours (V5 facet titles)
STATUS_COLORS = {
    'Decoupled': TEAL,
    'Coupled'  : CORAL,
    'Both Down': AMBER,
    'Declining': GRAY,
}

# Shared layout defaults applied to every figure
PLOT_DEFAULTS = dict(
    plot_bgcolor  = 'white',
    paper_bgcolor = 'white',
    font          = dict(family='Arial', size=12, color=SLATE),
)

X_co2 = merged[['co2_per_capita']].values
y_sat  = merged['life_satisfaction'].values

poly3       = PolynomialFeatures(degree=3, include_bias=True)
X_cubic     = poly3.fit_transform(X_co2)
cubic_model = LinearRegression().fit(X_cubic, y_sat)
r2_cub      = r2_score(y_sat, cubic_model.predict(X_cubic))
pearson_r   = np.corrcoef(X_co2.ravel(), y_sat)[0, 1]

x_smooth = np.exp(np.linspace(np.log(0.03), np.log(42), 400))
y_smooth = cubic_model.predict(poly3.transform(x_smooth.reshape(-1, 1)))

# colors
REGION_COLORS = {
    'Africa':"#11A5C0",             # teal
    'Americas':'#E8734A',           # coral
    'Asia-Pacific':'#F5A623',       # amber
    'Europe':'#7C3AED',             # violet
    'Middle East':'#2D8B4E',        # green
    'Other':'#9CA3AF',              # gray
}

plot_df = merged.sort_values(['Year', 'Country name']).copy()

# base scatter plot
fig1 = px.scatter(
    plot_df,
    x='co2_per_capita',
    y='life_satisfaction',
    animation_frame='Year',
    animation_group='Country name',
    color='region',
    color_discrete_map=REGION_COLORS,
    hover_name='Country name',
    hover_data={
        'co2_per_capita':':.2f',
        'life_satisfaction':':.2f',
        'region':True,
        'Year':False,
        'co2_key':False,
    },
    labels={
        'co2_per_capita':'CO2 per Capita (tonnes/person)',
        'life_satisfaction':'Life Satisfaction (0–10)',
        'region':'Region',
    },
    title='CO2 Emissions vs. Life Satisfaction (2014–2024)',
    log_x=True,
    range_x=[0.025, 50],
    range_y=[0.8, 9],
)

# markers
fig1.update_traces(
    marker=dict(size=9, opacity=0.78, line=dict(width=0.6, color='white')),
    selector=dict(mode='markers'),
)

# trace reg line
reg_trace = go.Scatter(
    x=x_smooth, y=y_smooth,
    mode='lines',
    name='Cubic fit (all years)',
    line=dict(color='#2E3B4E', width=2.5, dash='dot'),
    showlegend=True,
    hovertemplate='CO₂ = %{x:.2f}<br>Predicted LS = %{y:.2f}<extra>Cubic fit</extra>',
)

fig1.add_trace(reg_trace)

# repeat reg line so it's there for every animation frame
for frame in fig1.frames:
    frame.data = tuple(frame.data) + (
        go.Scatter(
            x=x_smooth, y=y_smooth,
            mode='lines',
            name='Cubic fit',
            line=dict(color='#2E3B4E', width=2.5, dash='dot'),
            showlegend=False,
        ),
    )

# annotations
annotations = [
    # stats box
    dict(
        text=(f'<b>Pearson r = {pearson_r:.2f}</b><br>'
              f'Cubic R² = {r2_cub:.2f}'),
        xref='paper', yref='paper', x=0.02, y=0.98,
        showarrow=False, align='left',
        bgcolor='rgba(255,255,255,0.88)',
        bordercolor='#CCCCCC', borderwidth=1,
        font=dict(size=11, family='Arial'),
    ),
    # industrialisation lab
    dict(
        text='← Industrialisation<br>   phase',
        xref='paper', yref='paper', x=0.09, y=0.1,
        showarrow=False, font=dict(size=10, color='#555555'),
        align='center',
    ),
    # diminishing returns
    dict(
        text='Diminishing returns →',
        xref='paper', yref='paper', x=0.72, y=1,
        showarrow=False, font=dict(size=10, color='#555555'),
    ),
]

# plot layout
fig1.update_layout(
    xaxis_title='CO2 per Capita: log scale (tonnes / person)',
    yaxis_title='Life Satisfaction (Cantril Ladder, 0–10)',
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=12, color='#2E3B4E'),
    legend=dict(
        title=dict(text='<b>Region</b>', font=dict(size=12)),
        x=1.01, y=1, xanchor='left',
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#E0E0E0', borderwidth=1,
    ),
    annotations=annotations,
    height=580, width=960,
    margin=dict(l=70, r=180, t=70, b=70),
)
fig1.update_xaxes(showgrid=True, gridcolor='#F2F2F2', zeroline=False,
                  tickfont=dict(size=11))
fig1.update_yaxes(showgrid=True, gridcolor='#F2F2F2', zeroline=False,
                  tickfont=dict(size=11))

# animation speed
fig1.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 900
fig1.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 400

fig1.show()

years = sorted(merged['Year'].unique())

# per year lookup dict
def get_year_df(yr):
    """Return a clean sub-frame for one year, one row per country."""
    g = (merged[merged['Year'] == yr]
         .groupby('iso_code', as_index=False)
         .agg(life_satisfaction=('life_satisfaction','mean'),
              co2_per_capita=('co2_per_capita','mean'),
              country_name=('Country name','first')))
    g['co2_log'] = np.log1p(g['co2_per_capita'])   # log-compress for color scale
    return g

# fig with the two geo subplots
fig2 = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'choropleth'}, {'type': 'choropleth'}]],
    subplot_titles=[
        '<b>Life Satisfaction Score</b> (0–10)',
        '<b>CO2 per Capita</b> (log-compressed scale)'
    ],
    horizontal_spacing=0.04,
)

init = get_year_df(years[0])

# left map: Life Satisfaction (diverging: red-yellow-green centered at 5.5ish. Idk scale is subjective might change)
fig2.add_trace(
    go.Choropleth(
        locations=init['iso_code'], z=init['life_satisfaction'],
        text=init['country_name'],
        colorscale='RdYlGn',
        zmin=1.0, zmid=5.5, zmax=8.5,
        colorbar=dict(
            title='Score', x=0.455, len=0.72, thickness=14,
            tickvals=[1,3,5.5,7,8.5],
            ticktext=['1','3','5.5 ★','7','8.5'],
        ),
        hovertemplate='<b>%{text}</b><br>Life Satisfaction: %{z:.2f}<extra></extra>',
        showscale=True,
        geo='geo',
    ),
    row=1, col=1,
)

# right map: CO2 (sequential blues; log-compressed so gulf states don't dominate)
CO2_MAX_LOG = np.log1p(42)   # 3.76 log scale ceiling (big math)
fig2.add_trace(
    go.Choropleth(
        locations=init['iso_code'], z=init['co2_log'],
        text=init['country_name'],
        customdata=init['co2_per_capita'],
        colorscale='Blues',
        zmin=0, zmax=CO2_MAX_LOG,
        colorbar=dict(
            title='tonnes/person', x=1.00, len=0.72, thickness=14,
            tickvals=[np.log1p(v) for v in [0, 1, 3, 8, 20, 42]],
            ticktext=['0','1','3','8','20','42'],
        ),
        hovertemplate='<b>%{text}</b><br>CO2: %{customdata:.2f} tonnes/person<extra></extra>',
        showscale=True,
        geo='geo2',
    ),
    row=1, col=2,
)

# animation frame (1 frame = 1 year)
frames = []
for yr in years:
    yrdf = get_year_df(yr)
    frames.append(go.Frame(
        data=[
            go.Choropleth(
                locations=yrdf['iso_code'], z=yrdf['life_satisfaction'],
                text=yrdf['country_name'],
                colorscale='RdYlGn', zmin=1.0, zmid=5.5, zmax=8.5,
            ),
            go.Choropleth(
                locations=yrdf['iso_code'], z=yrdf['co2_log'],
                text=yrdf['country_name'],
                customdata=yrdf['co2_per_capita'],
                colorscale='Blues', zmin=0, zmax=CO2_MAX_LOG,
            ),
        ],
        name=str(yr),
    ))
fig2.frames = frames

# slider
slider_steps = [
    dict(
        args=[[str(yr)],
              {'frame': {'duration': 700, 'redraw': True},
               'mode': 'immediate',
               'transition': {'duration': 300}}],
        label=str(yr), method='animate',
    )
    for yr in years
]

# play/pause btns
play_buttons = dict(
    type='buttons', showactive=False,
    y=-0.08, x=0.06, xanchor='right', yanchor='top',
    buttons=[
        dict(label='▶ Play',
             method='animate',
             args=[None, {'frame': {'duration': 900, 'redraw': True},
                          'fromcurrent': True,
                          'transition': {'duration': 400}}]),
        dict(label='⏸ Pause',
             method='animate',
             args=[[None], {'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate'}]),
    ],
)

# layout
fig2.update_layout(
    title=dict(
        text='Life Satisfaction & CO2 Emissions Across 162 Countries  (2014–2024)',
        font=dict(size=15, family='Arial'),
    ),
    height=480, width=1200,
    paper_bgcolor='white',
    font=dict(family='Arial', size=11, color='#2E3B4E'),
    margin=dict(l=10, r=10, t=80, b=120),
    sliders=[dict(
        steps=slider_steps,
        active=0,
        x=0.08, y=-0.04, len=0.85,
        pad={'t': 50},
        currentvalue=dict(
            prefix='Year: ', visible=True, xanchor='center',
            font=dict(size=14, color='#2E3B4E'),
        ),
        transition=dict(duration=300),
    )],
    updatemenus=[play_buttons],
    annotations=[
        dict(
            text='← Lowest satisfaction: Sub-Saharan Africa & South Asia',
            xref='paper', yref='paper', x=0.2, y=-0.2,
            showarrow=False, font=dict(size=10, color='#666'),
        ),
        dict(
            text='Highest emitters: Gulf States, North America, Australia →',
            xref='paper', yref='paper', x=0.93, y=-0.22,
            showarrow=False, xanchor='right', font=dict(size=10, color='#666'),
        ),
    ],
)

# style both geo projections the same
geo_style = dict(
    showcoastlines=True, coastlinecolor='#C0C0C0',
    showland=True, landcolor='#F7F7F7',
    showocean=True, oceancolor='#E6F2F8',
    showframe=False,
    projection_type='natural earth',
    showlakes=False,
)
fig2.update_geos(geo_style, 'geo')
fig2.update_geos(geo_style, 'geo2')

fig2.show()

# prep modelling data
FEATURES = [
    'Explained by: Log GDP per capita',
    'Explained by: Social support',
    'Explained by: Healthy life expectancy',
    'Explained by: Freedom to make life choices',
    'Explained by: Generosity',
    'Explained by: Perceptions of corruption',
    'co2_per_capita',
]
FEATURE_LABELS = [
    'Log GDP per capita',
    'Social support',
    'Healthy life expectancy',
    'Freedom of choice',
    'Generosity',
    'Perceptions of corruption',
    'CO2 per capita',
]

model_df = merged[FEATURES + ['life_satisfaction']].dropna().copy()
X_all  = model_df[FEATURES].values
y_all  = model_df['life_satisfaction'].values
X_co2_ = model_df[['co2_per_capita']].values

# CO2-only polynomial models
def fit_poly(deg, X, y):
    pf = PolynomialFeatures(degree=deg)
    Xp = pf.fit_transform(X)
    m  = LinearRegression().fit(Xp, y)
    return r2_score(y, m.predict(Xp))

r2_lin  = fit_poly(1, X_co2_, y_all)
r2_quad = fit_poly(2, X_co2_, y_all)
r2_cub  = fit_poly(3, X_co2_, y_all)

# standardised multiple reg 
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_all)
multi_m  = LinearRegression().fit(X_scaled, y_all)
r2_multi = r2_score(y_all, multi_m.predict(X_scaled))
coefs    = multi_m.coef_

# bootstrap 95% CIs (1,000 resamples)
rng = np.random.default_rng(42)
boot_coefs = np.array([
    LinearRegression().fit(
        X_scaled[idx := rng.choice(len(X_scaled), len(X_scaled))], y_all[idx]
    ).coef_
    for _ in range(1000)
])
ci_lo = np.percentile(boot_coefs, 2.5,  axis=0)
ci_hi = np.percentile(boot_coefs, 97.5, axis=0)

import copy
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# colors
TEAL_GRAD = ['#A8D5CF', '#1A9C87', '#155F56']
CORAL = '#E8734A'
TEAL_POS = '#1A9C87'
SLATE = '#2E3B4E'
INACTIVE_COLOR = '#CCCCCC'

r2_vals = [r2_lin, r2_quad, r2_cub, r2_multi]
bar_colors_left = TEAL_GRAD + [CORAL]
bar_labels_left = ['Linear (CO2 only)', 'Quadratic (CO2 only)',
                    'Cubic (CO2 only)', 'Multiple Regression']
bar_colors_right = [TEAL_POS if abs(c) > 0.05 else CORAL for c in coefs]

# figure
fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        '<b>R² by Model</b>',
        '<b>Standardised Regression Coefficients</b><br>'
        '<span style="font-size:11px">'
    ],
    column_widths=[0.40, 0.60],
    horizontal_spacing=0.14,
)

# Left panel: one trace per bar
for label, val, color in zip(bar_labels_left, r2_vals, bar_colors_left):
    fig3.add_trace(
        go.Bar(
            name=label,
            x=[label],
            y=[val],
            marker_color=color,
            marker_line=dict(color='white', width=1.5),
            text=[f'{val:.2f}'],
            textposition='outside',
            textfont=dict(size=12, color=SLATE),
            width=0.55,
            legendgroup=label,
            hovertemplate=f'<b>{label}</b><br>R² = {val:.3f}<extra></extra>',
        ),
        row=1, col=1,
    )

fig3.add_hline(
    y=r2_multi, line_dash='dash', line_color=CORAL, line_width=1.8,
    row=1, col=1,
)
fig3.add_annotation(
    xref='x', yref='y',
    x=1.5, y=r2_multi + 0.05,
    text=f'Multiple R² = {r2_multi:.2f}',
    showarrow=False,
    font=dict(size=10, color=CORAL, family='Arial'),
    row=1, col=1,
)

# Right panel: one trace per feature
for i, (label, c, lo, hi, color) in enumerate(
        zip(FEATURE_LABELS, coefs, ci_lo, ci_hi, bar_colors_right)):
    fig3.add_trace(
        go.Bar(
            name=f'feat_{i}',
            x=[c],
            y=[label],
            orientation='h',
            marker_color=color,
            marker_line=dict(color='white', width=0.8),
            error_x=dict(
                type='data', symmetric=False,
                array=[hi - c],
                arrayminus=[c - lo],
                color='#666', thickness=1.8, width=5,
            ),
            showlegend=False,
        ),
        row=1, col=2,
    )

fig3.add_vline(x=0,   line_color='#BBBBBB', line_width=1.2, row=1, col=2)
fig3.add_hline(y=3.5, line_dash='dot', line_color='#555', line_width=1.8, row=1, col=2)
fig3.add_annotation(
    xref='x2', yref='y2',
    x=0.25, y='CO2 per capita',
    text='← Small independent<br>effect',
    showarrow=False,
    font=dict(size=10, color=CORAL, family='Arial'),
)
fig3.add_annotation(
    xref='paper', yref='y2',
    x=1.0, y=3.25,
    text='Development<br>indicators',
    showarrow=False,
    font=dict(size=10, color='#555', family='Arial'),
)

fig3.update_xaxes(showgrid=False, zeroline=False,
                  tickfont=dict(size=10), row=1, col=1)
fig3.update_yaxes(showgrid=True, gridcolor='#F0F0F0',
                  range=[0, 1.12], tickfont=dict(size=11),
                  title='R²', row=1, col=1)
fig3.update_xaxes(showgrid=True, gridcolor='#F0F0F0',
                  zeroline=False, tickfont=dict(size=10),
                  title='Standardised β', row=1, col=2,
                  range=[min(coefs) - 0.15, max(coefs) + 0.25])
fig3.update_yaxes(showgrid=False, tickfont=dict(size=11), row=1, col=2,
                  range=[-0.5, len(FEATURE_LABELS) - 0.5])

fig3.update_layout(
    title=dict(
        text='<b>How Much Does CO2 Independently Explain Life Satisfaction?</b>',
        font=dict(size=15, family='Arial', color=SLATE),
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=12, color=SLATE),
    height=540, width=1120,
    margin=dict(l=60, r=80, t=100, b=60),
    barmode='overlay',
    legend=dict(
        title=dict(text='<b>Model</b>', font=dict(size=11)),
        x=1.01, y=1,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#E0E0E0', borderwidth=1,
    ),
)

df_2022 = merged[merged['Year'] == 2022].copy()
df_2022 = (df_2022[df_2022['co2_per_capita'] > 0]
           .dropna(subset=['life_satisfaction', 'co2_per_capita']))
df_2022['efficiency'] = df_2022['life_satisfaction'] / df_2022['co2_per_capita']

# Residuals from log-linear regression define efficiency status
X_eff   = np.log10(df_2022[['co2_per_capita']])
y_eff   = df_2022['efficiency']
reg_eff = LinearRegression().fit(X_eff, y_eff)
df_2022['residuals']    = y_eff - reg_eff.predict(X_eff)
df_2022['residual_pct'] = df_2022['residuals'].rank(pct=True) * 100

q_low, q_high = df_2022['residuals'].quantile([0.33, 0.66])

# FIX: return value must match EFFICIENCY_COLORS domain ('Global Average', not 'Average')
def get_eff_status(res):
    if res > q_high: return 'High Efficiency'
    if res < q_low:  return 'Low Efficiency'
    return 'Global Average'

df_2022['Efficiency Status'] = df_2022['residuals'].apply(get_eff_status)
avg_eff_val = df_2022['efficiency'].mean()

# Regression fit curve for the overlay line
x_fit_eff = np.logspace(
    np.log10(df_2022['co2_per_capita'].min()),
    np.log10(df_2022['co2_per_capita'].max()), 300
)
y_fit_eff = reg_eff.predict(np.log10(x_fit_eff.reshape(-1, 1)))

EFFICIENCY_OUTLIERS = ['Malawi', 'Costa Rica', 'Vietnam', 'Norway', 'United States', 'Qatar']

# using dash for slider
def build_fig4(min_pct=0):
    filtered = df_2022[df_2022['residual_pct'] >= min_pct].copy()

    fig = px.scatter(
        filtered,
        x='co2_per_capita', y='efficiency',
        color='Efficiency Status',
        color_discrete_map=EFFICIENCY_COLORS,
        hover_name='country',
        hover_data={
            'co2_per_capita'   : ':.2f',
            'life_satisfaction': ':.2f',
            'efficiency'       : ':.3f',
            'residual_pct'     : ':.0f',
        },
        labels={
            'co2_per_capita': 'CO₂ per Capita (tonnes/person, log scale)',
            'efficiency'    : 'Efficiency (Life Satisfaction / CO₂)',
        },
        title=(f'<b>CO₂ Efficiency vs. Life Satisfaction</b>  (2022)'
               f'<br><sup>Showing countries above the {min_pct:.0f}th efficiency percentile'
               f' — {len(filtered)} of {len(df_2022)} countries</sup>'),
        log_x=True,
        height=580,
    )

    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0.6, color='white')))

    # Expected-efficiency regression line
    fig.add_trace(go.Scatter(
        x=x_fit_eff, y=y_fit_eff, mode='lines',
        name='Expected efficiency',
        line=dict(color=SLATE, width=1.8, dash='dash'),
        hovertemplate='CO₂ = %{x:.2f}<br>Expected eff = %{y:.3f}<extra></extra>',
    ))

    # Global average reference line
    fig.add_hline(y=avg_eff_val, line_dash='dot', line_color='#888', line_width=1.5,
                  annotation_text=f'Global avg = {avg_eff_val:.2f}',
                  annotation_position='bottom right',
                  annotation_font=dict(size=10, color='#666'))

    fig.update_layout(
        **PLOT_DEFAULTS,
        margin=dict(l=70, r=160, t=90, b=60),
        legend=dict(title=dict(text='<b>Efficiency Status</b>', font=dict(size=11)),
                    x=1.01, y=1, bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#E0E0E0', borderwidth=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor='#F2F2F2', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#F2F2F2', zeroline=False)
    return fig

build_fig4().show()

df_decade = (merged[merged['Year'].between(2014, 2024)]
             .dropna(subset=['co2_per_capita', 'life_satisfaction'])
             .copy())

def process_trends(group):
    group = group.sort_values('year')
    start, end = group.iloc[0], group.iloc[-1]
    group['Carbon Index']   = (group['co2_per_capita']    / start['co2_per_capita'])    * 100
    group['Happiness Index']= (group['life_satisfaction'] / start['life_satisfaction']) * 100
    h_up   = end['life_satisfaction'] > start['life_satisfaction']
    c_down = end['co2_per_capita']    < start['co2_per_capita']
    # FIX: 'Declining' replaces the debug placeholder for the h_down + c_up case
    if   h_up and c_down:  status = 'Decoupled'
    elif h_up and not c_down: status = 'Coupled'
    elif not h_up and c_down: status = 'Both Down'
    else:                  status = 'Declining'
    group['Status']      = status
    group['Facet_Title'] = f"{start['country']} | {status}"
    return group

df_indexed = df_decade.groupby('country', group_keys=False).apply(process_trends)

# Global top-12 emitters anchor (2022 reference year)
top_emitters_global = (df_indexed[df_indexed['year'] == 2022]
                       .nlargest(12, 'co2_per_capita')['country'].unique())


# using dash region dropdown callback
def build_fig5(countries):
    df_sub    = df_indexed[df_indexed['country'].isin(countries)]
    df_melted = df_sub.melt(
        id_vars=['year', 'Facet_Title', 'Status', 'country'],
        value_vars=['Carbon Index', 'Happiness Index'],
        var_name='Metric', value_name='Score',
    )

    # Show only 2014 / 2024 endpoints by default (slope chart behaviour);
    # all years are still in the data for hover tooltips
    fig = px.line(
        df_melted,
        x='year', y='Score', color='Metric',
        facet_col='Facet_Title', facet_col_wrap=4,
        markers=True,
        title='<b>10-Year Decoupling Trends</b>  (2014 = 100)',
        color_discrete_map={'Carbon Index': CORAL, 'Happiness Index': TEAL},
        height=800,
        hover_data={'year': True, 'Score': ':.1f', 'Metric': False,
                    'Facet_Title': False, 'Status': False, 'country': False},
    )

    # Dropdown: filter which metric is visible
    show_both      = [True] * len(fig.data)
    show_carbon    = [t.name == 'Carbon Index'    for t in fig.data]
    show_happiness = [t.name == 'Happiness Index' for t in fig.data]

    fig.update_layout(
        **PLOT_DEFAULTS,
        title=dict(x=0.5, y=0.97, xanchor='center', font=dict(size=16)),
        margin=dict(t=160, b=50, r=180, l=60),
        legend=dict(title=dict(text='<b>Metric</b>'), x=1.02, y=1,
                    bgcolor='rgba(255,255,255,0.9)', bordercolor='#E0E0E0', borderwidth=1),
        updatemenus=[dict(
            type='dropdown', direction='down', x=0.0, y=1.10, showactive=True,
            bgcolor='white', bordercolor='#E0E0E0',
            font=dict(family='Arial', color=SLATE),
            buttons=[
                dict(label='Compare Both', method='update', args=[{'visible': show_both}]),
                dict(label='Carbon Only',  method='update', args=[{'visible': show_carbon}]),
                dict(label='Happiness Only', method='update', args=[{'visible': show_happiness}]),
            ],
        )],
    )

    fig.update_yaxes(matches=None, showgrid=True, gridcolor='#F2F2F2', zeroline=False,
                     title='Index (start year = 100)')
    fig.update_xaxes(showgrid=False, dtick=2,
                     tickvals=list(range(2014, 2025, 2)),
                     ticktext=[str(y) for y in range(2014, 2025, 2)])

    # Colour facet titles by status
    status_by_country = (df_sub.drop_duplicates('country')
                         .set_index('country')['Status'].to_dict())

    def colour_annotation(ann):
        raw = ann.text.split('=')[-1] 
        parts = [p.strip() for p in raw.split('|')]
        country_name = parts[0]
        status       = parts[1] if len(parts) > 1 else ''
        ann.update(
            text=f"<b>{country_name}</b>",
            font=dict(size=11, color=STATUS_COLORS.get(status, SLATE)),
        )

    fig.for_each_annotation(colour_annotation)

    # Add reference line at 100 (baseline) across all subplots
    fig.add_hline(y=100, line_dash='dot', line_color='#BBBBBB', line_width=1.2)

    return fig

build_fig5(top_emitters_global).show()

from dash import Dash, dcc, html, Input, Output, State, no_update
import json as _json

# ── Trace index constants for V3 ─────────────────────────────────────────────
MULTI_IDX  = 3    # Multiple Regression bar
FEAT_START = 4    # first feature coefficient bar

# ── Dash app ──────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Tab styling
_tab = dict(padding='8px 18px', fontFamily='Arial', fontSize='13px',
            color=SLATE, backgroundColor='white',
            borderBottom=f'2px solid #E8E8E8')
_tab_sel = dict(padding='8px 18px', fontFamily='Arial', fontSize='13px',
                fontWeight='bold', color=TEAL, backgroundColor='white',
                borderTop=f'3px solid {TEAL}', borderBottom='2px solid white')

# Pre-build static figures so they appear instantly on first load
_fig1 = copy.deepcopy(fig1)
_fig2 = copy.deepcopy(fig2)
_fig3 = copy.deepcopy(fig3)
_fig4 = build_fig4(0)
_fig5 = build_fig5(top_emitters_global)

app.layout = html.Div([

    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.H2('CO₂ Emissions & Life Satisfaction',
                style={'margin': '0', 'color': 'white', 'fontFamily': 'Arial',
                       'fontSize': '22px', 'fontWeight': 'bold'}),
        html.P('Cross-National Analysis  |  2014–2024  |  Group 12',
               style={'margin': '4px 0 0 0', 'color': 'rgba(255,255,255,0.75)',
                      'fontFamily': 'Arial', 'fontSize': '13px'}),
    ], style={'backgroundColor': TEAL, 'padding': '18px 32px'}),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dcc.Tabs(id='tabs', value='v1', style={'borderBottom': f'2px solid #E8E8E8'}, children=[

        dcc.Tab(label='1 · Development Paradox', value='v1',
                style=_tab, selected_style=_tab_sel),
        dcc.Tab(label='2 · Global Patterns',     value='v2',
                style=_tab, selected_style=_tab_sel),
        dcc.Tab(label='3 · Model Comparison',    value='v3',
                style=_tab, selected_style=_tab_sel),
        dcc.Tab(label='4 · CO₂ Efficiency',      value='v4',
                style=_tab, selected_style=_tab_sel),
        dcc.Tab(label='5 · 10-Year Trends',      value='v5',
                style=_tab, selected_style=_tab_sel),
    ]),

    html.Div(id='tab-content', style={'backgroundColor': 'white', 'padding': '20px 24px'}),

    # Persistent stores
    dcc.Store(id='v3-mask', data=_json.dumps([True] * len(FEATURES))),

], style={'backgroundColor': 'white', 'minHeight': '100vh'})


# ── Tab renderer ──────────────────────────────────────────────────────────────
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):

    desc_style = dict(fontFamily='Arial', fontSize='12px', color='#666',
                      marginBottom='8px', lineHeight='1.5')

    if tab == 'v1':
        return html.Div([
            html.P('Animated scatter showing the positive, but nonlinear relationship '
                   'between CO₂ per capita and life satisfaction. '
                   'The dotted cubic fit reveals diminishing returns at high emissions.',
                   style=desc_style),
            dcc.Graph(figure=_fig1, config={'displayModeBar': True},
                      style={'height': '600px'}),
        ])

    if tab == 'v2':
        return html.Div([
            html.P('Side-by-side choropleth maps linked to a shared year slider. '
                   'The diverging life satisfaction scale and log-compressed CO₂ scale '
                   'highlight geographic inequity. The highest emitters also score highest '
                   'on life satisfaction.',
                   style=desc_style),
            dcc.Graph(figure=_fig2, config={'displayModeBar': True},
                      style={'height': '520px'}),
        ])

    if tab == 'v3':
        return html.Div([
            html.P('Click any coefficient bar (right panel) to toggle that feature on/off. '
                   'The Multiple Regression R² bar and dashed reference line update live, '
                   'showing how much each factor independently contributes.',
                   style=desc_style),
            dcc.Store(id='v3-mask-local', data=_json.dumps([True] * len(FEATURES))),
            dcc.Graph(id='fig3-graph', figure=copy.deepcopy(fig3),
                      config={'displayModeBar': True}, style={'height': '580px'}),
        ])

    if tab == 'v4':
        return html.Div([
            html.P('Efficiency = Life Satisfaction ÷ CO₂ per capita. '
                   'Points above the dashed regression line are over-performing. '
                   'They achieve higher happiness per unit of emissions than expected. '
                   'Use the slider to filter out low-efficiency countries.',
                   style=desc_style),
            html.Div([
                html.Label('Minimum efficiency percentile:',
                           style={'fontFamily': 'Arial', 'fontSize': '12px',
                                  'color': SLATE, 'marginRight': '12px'}),
                dcc.Slider(id='eff-slider', min=0, max=90, step=5, value=0,
                           marks={i: f'{i}%' for i in range(0, 91, 10)},
                           tooltip={'placement': 'bottom', 'always_visible': False}),
            ], style={'maxWidth': '700px', 'marginBottom': '12px'}),
            dcc.Graph(id='fig4-graph', figure=_fig4,
                      config={'displayModeBar': True}, style={'height': '600px'}),
        ])

    if tab == 'v5':
        regions_all = ['All'] + sorted(merged['region'].unique().tolist())
        return html.Div([
            html.P('Base-year indexed trends (2014 = 100) for the top 12 emitters in the '
                   'selected region. Teal = Happiness Index, Coral = Carbon Index. '
                   'Title color indicates outcome: teal = Decoupled, coral = Coupled.',
                   style=desc_style),
            html.Div([
                html.Label('Filter by region:',
                           style={'fontFamily': 'Arial', 'fontSize': '12px',
                                  'color': SLATE, 'marginRight': '12px'}),
                dcc.Dropdown(id='region-filter', options=regions_all, value='All',
                             clearable=False,
                             style={'width': '220px', 'fontFamily': 'Arial',
                                    'fontSize': '12px'}),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '12px'}),
            dcc.Graph(id='fig5-graph', figure=_fig5,
                      config={'displayModeBar': True}, style={'height': '820px'}),
        ])

    return html.Div('Select a tab above.', style={'fontFamily': 'Arial', 'color': SLATE})


# ── V3 callback: feature toggle ───────────────────────────────────────────────
@app.callback(
    Output('fig3-graph', 'figure'),
    Output('v3-mask-local', 'data'),
    Input('fig3-graph', 'clickData'),
    State('fig3-graph', 'figure'),
    State('v3-mask-local', 'data'),
    prevent_initial_call=True,
)
def toggle_feature(click_data, current_fig, mask_json):
    if click_data is None:
        return no_update, no_update
    curve_num = click_data['points'][0]['curveNumber']
    if curve_num < FEAT_START:
        return no_update, no_update

    feat_idx    = curve_num - FEAT_START
    active_mask = _json.loads(mask_json)
    active_mask[feat_idx] = not active_mask[feat_idx]

    active_idx = [i for i, on in enumerate(active_mask) if on]
    if active_idx:
        X_sub  = X_scaled[:, active_idx]
        m      = LinearRegression().fit(X_sub, y_all)
        new_r2 = r2_score(y_all, m.predict(X_sub))
    else:
        new_r2 = 0.0

    fig = copy.deepcopy(current_fig)
    t   = fig['data'][curve_num]
    if active_mask[feat_idx]:
        t['marker']['color']   = bar_colors_right[feat_idx]
        t['marker']['opacity'] = 1.0
    else:
        t['marker']['color']   = INACTIVE
        t['marker']['opacity'] = 0.45

    fig['data'][MULTI_IDX]['y']    = [new_r2]
    fig['data'][MULTI_IDX]['text'] = [f'{new_r2:.2f}']
    fig['data'][MULTI_IDX]['hovertemplate'] = (
        f'<b>Multiple Regression</b><br>R² = {new_r2:.3f}<extra></extra>')

    fig['layout']['shapes'][0]['y0'] = new_r2
    fig['layout']['shapes'][0]['y1'] = new_r2
    for ann in fig['layout']['annotations']:
        if ann.get('text', '').startswith('Multiple R²'):
            ann['text'] = f'Multiple R² = {new_r2:.2f}'
            ann['y']    = new_r2 + 0.05
            break

    return fig, _json.dumps(active_mask)


# ── V4 callback: efficiency percentile slider ─────────────────────────────────
@app.callback(
    Output('fig4-graph', 'figure'),
    Input('eff-slider', 'value'),
    prevent_initial_call=True,
)
def update_v4(min_pct):
    return build_fig4(min_pct)


# ── V5 callback: region dropdown ──────────────────────────────────────────────
@app.callback(
    Output('fig5-graph', 'figure'),
    Input('region-filter', 'value'),
    prevent_initial_call=True,
)
def update_v5(selected_region):
    if selected_region == 'All':
        countries = top_emitters_global
    else:
        pool = (df_indexed[(df_indexed['year'] == 2022) &
                           (df_indexed['region'] == selected_region)]
                .nlargest(12, 'co2_per_capita')['country'].unique())
        countries = pool if len(pool) > 0 else top_emitters_global
    return build_fig5(countries)


# ── Run ───────────────────────────────────────────────────────────────────────
#app.run(debug=True, port=8050)
