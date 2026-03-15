# 🚧 Intelligent A/B Testing Framework

An end-to-end experimentation framework that automates hypothesis generation, power analysis, statistical testing (frequentist & Bayesian), and executive narrative creation using Claude. Includes SRM detection, novelty effect checks, and segmented drill-down analysis.

## Tech Stack

- **Data Generation**: PySpark, NumPy
- **Statistics Engine**: SciPy, StatsModels, NumPy
- **AI Integration**: Anthropic Claude API
- **Dashboard**: Gradio, Plotly, Matplotlib
- **Data Format**: Parquet, CSV (via PyArrow)

## Project Structure

```
ab-testing-framework/
├── src/
│   ├── data_generation/
│   │   └── generate_datasets.py      # PySpark synthetic data generator (1M rows x 3 datasets)
│   ├── stats/
│   │   ├── power_analysis.py          # Sample size calculator
│   │   ├── frequentist.py             # Z-test, Welch's t-test, log-rank test
│   │   ├── bayesian.py                # Beta-binomial posterior, P(B>A), expected loss
│   │   ├── srm_detection.py           # Sample ratio mismatch detection
│   │   ├── novelty_effect.py          # Early vs late lift comparison
│   │   └── segmentation.py            # Segment-level drill-down analysis
│   ├── claude_integration/
│   │   └── claude_integration.py      # Hypothesis generation, executive summaries, recommendations
│   └── ui/
│       └── app.py                     # Gradio dashboard (5-tab layout)
├── data/
│   ├── parquet/                       # Full datasets (1M rows each)
│   └── csv/                           # 10K row samples for quick inspection
├── tests/
│   ├── test_frequentist.py
│   ├── test_bayesian.py
│   └── test_srm.py
├── assets/                            # Screenshots, architecture diagram
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Sample Datasets

The framework ships with 3 synthetic experiments, each designed to demonstrate a different statistical outcome:

| Dataset | Scenario | Expected Result |
|---------|----------|-----------------|
| **Conversion Rate** | New checkout UI test | Clearly significant (p < 0.01) with novelty effect and segment heterogeneity |
| **Revenue / AOV** | Product recommendation widget | Borderline / ambiguous (p ~ 0.06–0.09), desktop-driven lift |
| **Subscription Renewal** | Redesigned reminder emails | Clean null result — no effect across any segment |

## Build Phases

| Phase | Name | Description | Status |
|-------|------|-------------|--------|
| 1 | Data Foundation | PySpark generation of 3 synthetic datasets (1M rows each) with seeded statistical properties | Not Started |
| 2 | Core Stats Engine | Power analysis, frequentist testing, Bayesian analysis, SRM detection, novelty checks, segmentation | Not Started |
| 3 | Claude Integration | AI-powered hypothesis generation, executive summaries, segment insights, ship/iterate/kill recommendations | Not Started |
| 4 | Gradio Dashboard | Interactive 5-tab UI with visualizations, drill-downs, and Claude-generated narratives | Not Started |
| 5 | Polish & Packaging | README, architecture diagram, tests, setup instructions | Not Started |

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/shreyeshi-somya/ab-testing-framework.git
   cd ab-testing-framework
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your Anthropic API key:

   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

4. Generate the datasets:

   ```bash
   python src/data_generation/generate_datasets.py
   ```

5. Launch the dashboard:

   ```bash
   python src/ui/app.py
   ```
