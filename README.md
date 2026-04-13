# CO₂ Emissions & Life Satisfaction Dashboard

**A Cross-National Analysis of Environmental Impact and Human Well-Being, 2014–2024**

---

## Overview

This project investigates whether national per-capita CO₂ emissions predict average life satisfaction across countries from 2014 to 2024. Using data from the World Happiness Report and Our World in Data, we examine the tension between industrial progress and sustainability and find that the relationship is more nuanced than a simple correlation suggests.

**Central finding:** While CO₂ emissions and life satisfaction are moderately positively correlated (Pearson *r* = 0.51), this association is largely driven by the confounding effect of economic development. Once socioeconomic factors are controlled for, CO₂ emissions have a small independent effect on life satisfaction (multiple regression R² ≈ 0.73). A 2D Gaussian fit in log space reveals a clear inverted-U relationship, with diminishing returns beyond a peak emissions threshold. This is consistent with the hypothesis that environmental costs eventually outweigh the developmental benefits of industrialization.

---

## Live Dashboard

> Deploy to Vercel by following the instructions in the [Deployment](#deployment) section.

The dashboard is a five-tab Dash application covering each argumentative step of the analysis:

| Tab | Visual | Argument |
|-----|--------|----------|
| 1 · Development Paradox | Animated scatter + Gaussian fit | Positive correlation is real, but nonlinear |
| 2 · Regional Patterns | Region-filtered dual choropleth | Geographic inequity in emissions vs. well-being |
| 3 · Model Comparison | R² panel + coefficient chart | CO₂ is a proxy for development, not a cause |
| 4 · CO₂ Efficiency | Efficiency scatter + percentile slider | Which countries get the most happiness per unit of carbon? |
| 5 · 10-Year Trends | Faceted decoupling line charts | Rising emissions no longer reliably predict rising happiness |

---

## Repository Structure

```
.
├── app.py                          # Dash application + all figure builders (Vercel entry point)
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel routing config
│
├── data/
│   ├── co2-emissions-per-capita.csv        # Our World in Data — CO₂ per capita by country/year
│   └── WHR25_Data_Figure_2.1v3.xlsx        # World Happiness Report 2025
│
├── CO2_Dashboard.ipynb             # Full analysis notebook (data pipeline + all 5 visuals + Dash)
│
└── docs/
    ├── Design.pdf                  # Visual design specification
    └── Proposal.pdf                # Original project proposal
```

---

## Data Sources

**World Happiness Report 2025**
- 1,969 rows × 13 columns covering 186 countries (2011–2024)
- Key fields: `Life evaluation (3-year average)`, `Explained by: Log GDP per capita`, `Explained by: Social support`, `Explained by: Healthy life expectancy`, `Explained by: Freedom to make life choices`, `Explained by: Generosity`, `Explained by: Perceptions of corruption`
- Source: [worldhappiness.report](https://worldhappiness.report)

**Our World in Data — CO₂ Emissions per Capita**
- 50,000+ rows covering 250+ entities from 1750 to present
- Key field: `co2_per_capita` (tonnes of CO₂ per person per year)
- Filtered to 162 countries × 2014–2024 after merging and excluding aggregate rows (continents, income groups)
- Source: [ourworldindata.org/co2-emissions](https://ourworldindata.org/co2-emissions)

**Merged dataset:** 1,635 rows × 162 countries × 11 years (2014–2024), after country name harmonisation and ISO-3 filtering.

---

## Visuals

### Visual 1 — Animated Scatter: CO₂ vs. Life Satisfaction
An animated scatter plot with a year slider (2014–2024). Each point is a country, coloured by world region and sized by population. A **2D Gaussian fit in log(x) space** overlays the data, capturing the inverted-U / diminishing-returns shape that a linear or standard cubic fit misses on a log axis. The peak of the Gaussian marks the CO₂ level at which additional emissions stop improving life satisfaction.

*Interactions:* Year animation slider, zoom/pan, hover tooltip (country, CO₂, life satisfaction, year), click to isolate a country.

### Visual 2 — Dual Choropleth: Life Satisfaction & CO₂ per Capita
Two side-by-side world maps — life satisfaction (RdYlGn diverging scale, centred at the global mean) and CO₂ per capita (Blues sequential scale, log-compressed). A **region dropdown** zooms both maps to the selected area simultaneously. A **Play/Pause button** animates through 2014–2024, rebuilding both maps at each step via a Dash `dcc.Interval`.

*Interactions:* Region dropdown, year slider, Play/Pause animation, hover tooltip.

### Visual 3 — Model Comparison Panel
A two-panel figure: the left panel shows R² values for four regression models (linear, quadratic, cubic CO₂-only, and full multiple regression); the right panel shows bootstrapped standardised regression coefficients with 95% confidence intervals. **Clicking any coefficient bar in the right panel toggles that feature on/off**, and the Multiple Regression R² bar and reference line update live to show how much variance is explained without that predictor.

*Interactions:* Click-to-toggle features, live R² update, hover tooltips on both panels.

### Visual 4 — CO₂ Efficiency Scatter
Efficiency is defined as Life Satisfaction ÷ CO₂ per capita. Points are colour-coded by efficiency status (High / Global Average / Low) based on residuals from a log-linear regression — so points *above* the dashed expected-efficiency line are over-performing countries given their emissions level. A **percentile slider** filters the view to show only countries above a chosen efficiency threshold.

*Interactions:* Efficiency percentile slider, hover tooltip (country, CO₂, life satisfaction, efficiency rank), outlier country labels.

### Visual 5 — 10-Year Decoupling Trends
Faceted line charts (2014 = 100 index) for the top 12 emitters in the selected region. Each subplot shows both the Carbon Index (coral) and Happiness Index (teal) for one country. Facet title colour indicates the decoupling outcome: teal = Decoupled (happiness up, carbon down), coral = Coupled (both up), amber = Both Down, gray = Declining. A **region dropdown** re-runs the analysis for the top emitters in that region. A **metric dropdown** filters to Carbon Only, Happiness Only, or both.

*Interactions:* Region filter dropdown, metric visibility dropdown, hover tooltips.

---

## Argument Summary

The project builds a six-step argument:

1. **The development paradox** — raw data shows a positive correlation (r = 0.51), which initially contradicts the hypothesis, but is explained by the confounding effect of industrialisation.
2. **Nonlinearity** — a Gaussian fit in log space reveals an inverted-U: life satisfaction rises with early-stage emissions growth, peaks, then flattens or declines. A cubic polynomial (R² = 0.48) also captures this, but the Gaussian is more interpretable on a log axis.
3. **CO₂ is a proxy, not a driver** — multiple regression including GDP, social support, health expectancy, and freedom achieves R² ≈ 0.73, and the CO₂ coefficient shrinks dramatically, confirming it reflects development stage rather than independently driving well-being.
4. **Geographic inequity** — the highest emitters (Global North, Gulf states) score highest on life satisfaction, while the lowest emitters (Sub-Saharan Africa, South Asia) score lowest. The burden of environmental costs is borne disproportionately by those least responsible.
5. **Temporal decoupling** — over 2014–2024, mid-income countries that increased CO₂ did not see proportional life satisfaction gains, providing longitudinal evidence for diminishing returns.
6. **Machine learning confirmation** — Random Forest feature importance ranks CO₂ below GDP, social support, health, and freedom; a GAM identifies the specific emission level at which the CO₂–life satisfaction relationship transitions from positive to flat/negative.

---

## Key Technical Decisions

**Why Gaussian fit instead of cubic polynomial for Visual 1?**
The scatter uses a log x-axis. A standard Gaussian evaluated on raw CO₂ values has its bell peak at the arithmetic mean of CO₂ (~7–8 t/person), which sits far to the right on a log axis and appears nearly flat. The `gaussian_logx` function fits `exp(-((log(x) - b)² / 2c²))`, placing the peak at `exp(b)` in original units — wherever the data's arch actually is — and producing a symmetric bell on the log axis.

**Why Dash instead of Altair for the full dashboard?**
Visual 3's feature-toggle interaction requires a live server-side callback (refit regression, update bar height, move reference line, update annotation). Altair's Vega-Lite selections operate client-side and cannot refit a scikit-learn model. Dash callbacks run Python server-side, making the interaction straightforward. Visual 4's efficiency slider and Visual 2's Play/Pause animation follow the same pattern.

**Why `fig.update_geos(**geo_cfg)` instead of `fig.update_geos(patch=geo_cfg)`?**
Plotly's `patch=` keyword only updates the first geo subplot. Using `**kwargs` applies the geo style and scope to all geo subplots in the figure simultaneously, which is required for the dual choropleth layout.

**Country name harmonisation:**
Fifteen WHR country names do not match Our World in Data entity names directly (e.g. `"Türkiye"` vs `"Turkey"`, `"Viet Nam"` vs `"Vietnam"`). An explicit `NAME_MAP` dict handles these before the merge. Three entities (North Cyprus, Puerto Rico, Somaliland Region) have no CO₂ counterpart and are excluded. Aggregate rows in the CO₂ dataset (continents, income groups) are removed by requiring a valid ISO-3 code.

---

## References

Apergis, N. (2018). The impact of greenhouse gas emissions on personal well-being: Evidence from a panel of 58 countries. *Journal of Happiness Studies, 19*(1), 69–80. https://doi.org/10.1007/s10902-016-9809-y

Barrington-Leigh, C. P. (2021). Life satisfaction and sustainability: A policy framework. *SN Social Sciences, 1*(7), 176. https://doi.org/10.1007/s43545-021-00185-8

Barrington-Leigh, C., & Escande, A. (2018). Measuring progress and well-being: A comparative review of indicators. *Social Indicators Research, 135*, 893–925. https://doi.org/10.1007/s11205-016-1505-0

De Neve, J. E., & Sachs, J. D. (2020). Sustainable development and well-being. In J. Helliwell, R. Layard, J. Sachs, & J. E. De Neve (Eds.), *World Happiness Report 2020* (pp. 113–128). Sustainable Development Solutions Network.

Helliwell, J., Layard, R., Sachs, J., & De Neve, J. E. (Eds.). (2025). *World Happiness Report 2025*. Sustainable Development Solutions Network.

Ritchie, H., Roser, M., & Rosado, P. (2020). CO₂ and greenhouse gas emissions. *Our World in Data*. https://ourworldindata.org/co2-emissions
