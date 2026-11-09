"""OpenTelemetry setup for the example, in one of two modes.

collector  Every signal goes to one OTLP endpoint (an OpenTelemetry Collector, default
           http://localhost:4318). The collector filters and fans out.
incode     No collector. One TracerProvider with two span processors:
             Arize      only AI spans; the top AI span becomes the root
             Dynatrace  all spans, prompt and completion content removed
           Logs and metrics (delta temporality) go to Dynatrace.

Either way there is one TracerProvider, so every span, log, and metric exemplar from a
request carries the same trace id.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from opentelemetry import metrics, trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider, ObservableCounter
from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from openinference_processors import ContentRedactingExporter, OpenInferenceSpanProcessor

SERVICE_NAME = "ax-dt-example"
ARIZE_TRACES_URL = "https://otlp.arize.com/v1/traces"

# Dynatrace stores counters and histograms as deltas.
DELTA = {
    Counter: AggregationTemporality.DELTA,
    ObservableCounter: AggregationTemporality.DELTA,
    Histogram: AggregationTemporality.DELTA,
}


@dataclass
class Telemetry:
    mode: str
    shutdown_fns: list[Callable[[], None]] = field(default_factory=list)

    def shutdown(self) -> None:
        for fn in self.shutdown_fns:
            fn()


def setup(mode: str) -> Telemetry:
    if mode not in ("collector", "incode"):
        raise ValueError(f"unknown TELEMETRY_MODE {mode!r}")

    attributes = {
        "service.name": SERVICE_NAME,
        "service.version": "1.0.0",
        "deployment.environment.name": os.getenv("DEPLOYMENT_ENVIRONMENT", "dev"),
    }
    # In collector mode the collector adds the project name if the app does not.
    project = os.getenv("ARIZE_PROJECT_NAME")
    if mode == "incode" or project:
        attributes["openinference.project.name"] = project or "ax-dt-incode-example"
    resource = Resource.create(attributes)

    if mode == "collector":
        exporters = _collector_exporters()
    else:
        exporters = _incode_exporters()

    tracer_provider = TracerProvider(resource=resource)
    for processor in exporters.span_processors:
        tracer_provider.add_span_processor(processor)
    trace.set_tracer_provider(tracer_provider)

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporters.logs))
    set_logger_provider(logger_provider)
    handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
    app_logger = logging.getLogger("ax_dt_example")
    app_logger.setLevel(logging.INFO)
    app_logger.addHandler(handler)

    reader = PeriodicExportingMetricReader(exporters.metrics, export_interval_millis=10_000)
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meter_provider)

    telemetry = Telemetry(mode=mode)
    # Flush order: spans first (they are what the verify script looks up), then logs, metrics.
    telemetry.shutdown_fns += [tracer_provider.shutdown, logger_provider.shutdown, meter_provider.shutdown]
    return telemetry


@dataclass
class _Exporters:
    span_processors: list
    logs: OTLPLogExporter
    metrics: OTLPMetricExporter


def _collector_exporters() -> _Exporters:
    # Endpoint and headers come from the standard OTEL_EXPORTER_OTLP_* variables.
    base = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318").rstrip("/")
    return _Exporters(
        span_processors=[BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{base}/v1/traces"))],
        logs=OTLPLogExporter(endpoint=f"{base}/v1/logs"),
        # Cumulative on purpose: the collector's cumulative_to_delta processor converts it.
        metrics=OTLPMetricExporter(endpoint=f"{base}/v1/metrics"),
    )


def _incode_exporters() -> _Exporters:
    arize_headers = {
        "space_id": _required("ARIZE_SPACE_ID"),
        "api_key": _required("ARIZE_API_KEY"),
    }
    arize = OpenInferenceSpanProcessor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=ARIZE_TRACES_URL, headers=arize_headers)),
        reparent_orphans=_flag("ARIZE_REPARENT_ORPHANS", True),
    )

    dt_base = dynatrace_otlp_base(_required("DYNATRACE_ENV_URL"))
    dt_headers = {"Authorization": dynatrace_authorization(_required("DYNATRACE_API_TOKEN"))}
    dt_spans = OTLPSpanExporter(endpoint=f"{dt_base}/v1/traces", headers=dt_headers)
    if _flag("DYNATRACE_REDACT_AI_CONTENT", True):
        dt_spans = ContentRedactingExporter(dt_spans)

    return _Exporters(
        span_processors=[arize, BatchSpanProcessor(dt_spans)],
        logs=OTLPLogExporter(endpoint=f"{dt_base}/v1/logs", headers=dt_headers),
        metrics=OTLPMetricExporter(
            endpoint=f"{dt_base}/v1/metrics", headers=dt_headers, preferred_temporality=DELTA
        ),
    )


def dynatrace_otlp_base(env_url: str) -> str:
    """https://{env}.live.dynatrace.com -> https://{env}.live.dynatrace.com/api/v2/otlp"""
    url = env_url.rstrip("/")
    return url if url.endswith("/api/v2/otlp") else f"{url}/api/v2/otlp"


def dynatrace_authorization(token: str) -> str:
    """Access tokens (dt0c01.*) use Api-Token; platform tokens (dt0s16.*) use Bearer."""
    scheme = os.getenv("DYNATRACE_AUTH_SCHEME") or ("Api-Token" if token.startswith("dt0c") else "Bearer")
    return f"{scheme} {token}"


def _required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


def _flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")
