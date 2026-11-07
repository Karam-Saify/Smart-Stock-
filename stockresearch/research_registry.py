"""Central catalogue of implemented research methods and their limitations.

The registry describes the current executable T01-T18 engines. It does not
claim that every method is confirmatory or suitable for every dataset.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ResearchMethod:
    test_id: str
    domain: str
    question: str
    methods: tuple[str, ...]
    audience_note: str
    limitations: tuple[str, ...]
    status: str = "implemented"


_METHODS = (
    ResearchMethod("T01", "distribution", "How do return distributions compare with common parametric families?",
                   ("normal", "student_t", "empirical_quantiles"),
                   "Compare fitted distributions and empirical reference quantiles.",
                   ("In-sample fit criteria do not establish out-of-sample fit.", "Parameter uncertainty and goodness-of-fit need separate treatment.")),
    ResearchMethod("T02", "distribution_stability", "How does the sample mean evolve?",
                   ("rolling_mean", "expanding_mean", "cusum_screen"),
                   "Inspect mean evolution and a screening statistic.",
                   ("CUSUM screen is not a calibrated formal break-test p-value.",)),
    ResearchMethod("T03", "volatility", "How do variance estimators behave on a chronological holdout?",
                   ("rolling_variance", "ewma", "garch_training_fit"),
                   "Compare simple variance estimators and inspect a fitted GARCH diagnostic.",
                   ("GARCH output is a training-fit diagnostic, not a like-for-like full-holdout forecast.",)),
    ResearchMethod("T04", "distribution", "How does return asymmetry evolve?",
                   ("rolling_skewness",),
                   "Inspect rolling and full-sample skewness.",
                   ("Rolling estimates are noisy and overlapping.",)),
    ResearchMethod("T05", "distribution", "How does return tail-weight evolve?",
                   ("rolling_excess_kurtosis",),
                   "Inspect rolling and full-sample excess kurtosis.",
                   ("Rolling estimates are noisy and overlapping.",)),
    ResearchMethod("T06", "distribution_change", "Do first- and second-half return distributions differ?",
                   ("kolmogorov_smirnov", "anderson_darling", "wasserstein"),
                   "Compare two historical subperiods using distribution diagnostics.",
                   ("Exploratory period comparison.", "P-values are unadjusted until a hypothesis family is defined and corrected.")),
    ResearchMethod("T07", "data_diagnostics", "Which return observations are flagged as unusual?",
                   ("zscore", "robust_zscore", "empirical_percentiles"),
                   "Flag unusual observations for inspection, not automatic removal.",
                   ("Flags are diagnostics and must not trigger automatic deletion.",)),
    ResearchMethod("T08", "stationarity", "Are returns and squared returns stationary?",
                   ("adf", "kpss", "phillips_perron"),
                   "Use complementary tests with different null hypotheses.",
                   ("Tests have different nulls and assumptions; interpret jointly.",)),
    ResearchMethod("T09", "dependence", "Is there serial dependence in returns?",
                   ("acf", "pacf", "ljung_box"),
                   "Inspect autocorrelation patterns and joint lag diagnostics.",
                   ("Repeated lag p-values are unadjusted in the current engine.",)),
    ResearchMethod("T10", "dependence", "Is there AR(1)-style persistence?",
                   ("ar1_regression", "conditional_half_life_diagnostic"),
                   "Inspect a descriptive AR(1) coefficient and conditional half-life.",
                   ("Half-life interpretation requires stronger assumptions.",)),
    ResearchMethod("T11", "trend_dependence", "How do rolling trends and cumulative-return dependence behave?",
                   ("rolling_linear_trend", "cumulative_return_autocorrelation"),
                   "Explore rolling slope summaries and cumulative-return persistence.",
                   ("Overlapping windows and cumulative returns induce dependence.",)),
    ResearchMethod("T12", "seasonality", "Are there weekday or month-related return differences?",
                   ("weekday_anova", "month_anova", "period_5_decomposition"),
                   "Explore calendar-group differences and a descriptive decomposition.",
                   ("Exploratory ANOVA p-values are unadjusted.", "Calendar effects can be unstable.")),
    ResearchMethod("T13", "structural_change", "Is there evidence of parameter instability or mean shifts?",
                   ("ols_cusum", "pelt_mean_segmentation"),
                   "Inspect regression stability and exploratory segmentation.",
                   ("PELT locations have no calibrated p-values.", "Penalty sensitivity and temporal stability are required.")),
    ResearchMethod("T14", "volatility", "How do volatility forecasts compare on a chronological holdout?",
                   ("rolling_historical_variance", "ewma", "garch_bounded_origin"),
                   "Compare historical and EWMA forecasts, with sampled GARCH forecast origins.",
                   ("GARCH is evaluated at sampled holdout origins; method coverage differs.",)),
    ResearchMethod("T15", "volatility", "Are ARCH effects present and how persistent is fitted variance?",
                   ("arch_lm", "garch_persistence"),
                   "Diagnose conditional heteroskedasticity and fitted persistence.",
                   ("Persistence is specification-dependent and not proof of future persistence.",)),
    ResearchMethod("T16", "volatility", "Do asymmetric volatility specifications fit differently?",
                   ("egarch", "gjr_garch"),
                   "Inspect asymmetric conditional-volatility model diagnostics.",
                   ("Information criteria are in-sample; out-of-sample comparison is needed.",)),
    ResearchMethod("T17", "risk", "How do VaR and CVaR estimates behave on a chronological holdout?",
                   ("historical_var_cvar", "normal_var_cvar", "student_t_var_cvar"),
                   "Compare training-estimated tail thresholds against a holdout.",
                   ("Holdout violation counts have substantial sampling uncertainty.", "CVaR is estimated from the training tail.")),
    ResearchMethod("T18", "regimes_structural_change", "Are there latent states or volatility change points?",
                   ("two_state_gaussian_hmm", "pelt_volatility_segmentation"),
                   "Explore in-sample latent-state and volatility segmentation.",
                   ("In-sample segmentation does not establish online state stability.", "Penalty sensitivity and chronological refitting are needed.")),
)

METHODS: dict[str, ResearchMethod] = {method.test_id: method for method in _METHODS}


def list_methods(domain: str | None = None) -> list[ResearchMethod]:
    """Return methods in stable ID order, optionally filtered by domain."""
    values = list(_METHODS)
    if domain:
        values = [item for item in values if item.domain == domain]
    return values


def get_method(test_id: str) -> ResearchMethod:
    """Return one method definition or raise a descriptive KeyError."""
    try:
        return METHODS[test_id]
    except KeyError as exc:
        raise KeyError(f"Unknown research method {test_id!r}") from exc


def methods_as_dicts(domain: str | None = None) -> list[dict]:
    """Return JSON-friendly method metadata."""
    return [asdict(item) for item in list_methods(domain)]
