import psutil
from mlflow.system_metrics.metrics.base_metrics_monitor import BaseMetricsMonitor


class SwapMonitor(BaseMetricsMonitor):
    """Monitor swap memory usage for MLflow system metrics logging."""

    def collect_metrics(self) -> None:
        swap = psutil.swap_memory()
        self._metrics["swap_usage_megabytes"].append(swap.used / 1e6)
        self._metrics["swap_usage_percentage"].append(swap.percent)

    def aggregate_metrics(self) -> dict[str, float]:  # pyright: ignore[reportIncompatibleMethodOverride]  # base class has no return type annotation; our return is semantically correct
        return {k: round(sum(v) / len(v), 1) for k, v in self._metrics.items()}
