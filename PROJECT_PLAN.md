# Intelligent A/B Testing Framework

An end-to-end experimentation framework that automates hypothesis generation, power analysis, statistical testing (frequentist and Bayesian), and executive narrative creation using Claude. Includes SRM detection, novelty effect checks, and segmented drill-down analysis.

## Technologies & Topics

- Python
- Statistics
- AI/LLMs
- Bayesian Analysis
- Gradio

## Project Plan

1. Build core experimentation engine with power analysis, statistical testing (frequentist & Bayesian), SRM detection, and novelty effect checks using scipy and statsmodels
2. Integrate Anthropic API for automated hypothesis generation from uploaded data and executive summary creation with actionable recommendations
3. Create interactive Gradio dashboard with comprehensive visualizations including confidence intervals, lift distributions, cumulative metrics over time, and segmented analysis breakdowns
4. Develop sample dataset library with 3 pre-loaded experiments (conversion rate, AOV lift, subscription renewal) for immediate demo capability
5. Implement advanced segmentation analysis allowing drill-down by user characteristics (new vs returning, geography, demographics) with automated insight generation
6. Add Bayesian analysis option with credible intervals, posterior distributions, and probability of improvement calculations alongside traditional frequentist methods
