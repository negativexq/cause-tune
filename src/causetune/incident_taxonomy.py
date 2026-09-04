"""Frozen taxonomies for Experiment 02A incident diagnosis."""

from __future__ import annotations

from dataclasses import dataclass


FAILURE_FAMILIES: tuple[str, ...] = (
    "db_connection_pool_exhaustion",
    "db_query_regression",
    "memory_leak",
    "downstream_dependency_timeout",
    "cache_stampede",
    "kafka_consumer_lag",
    "thread_pool_exhaustion",
    "disk_io_saturation",
    "dns_resolution_failure",
    "tls_certificate_expiration",
    "rate_limit_misconfiguration",
    "configuration_regression",
)
FAILURE_FAMILY_SET = frozenset(FAILURE_FAMILIES)

ACTIONS: tuple[str, ...] = (
    "rollback_recent_change",
    "restart_or_replace_instance",
    "scale_capacity",
    "revert_configuration",
    "restore_dependency",
    "renew_certificate",
    "mitigate_query",
    "drain_or_rebalance_workload",
    "clear_or_throttle_cache_pressure",
    "investigate_before_action",
)
ACTION_SET = frozenset(ACTIONS)

SLICES: tuple[str, ...] = ("standard", "hard", "transfer")
SLICE_COUNTS = {"standard": 72, "hard": 48, "transfer": 24}
DIFFICULTIES = frozenset(SLICES)


@dataclass(frozen=True)
class FailureSpec:
    action: str
    culprit_role: str
    metric_label: str
    metric_change: str
    primary_log: str
    secondary_signal: str
    alert_signal: str


FAILURE_SPECS: dict[str, FailureSpec] = {
    "db_connection_pool_exhaustion": FailureSpec(
        action="scale_capacity",
        culprit_role="db",
        metric_label="database connection wait p95",
        metric_change="35 ms -> 2.8 s",
        primary_log="timed out after 2000 ms waiting for an available database connection",
        secondary_signal="requests queue while database CPU remains below saturation",
        alert_signal="connection acquisition time crossed the service threshold",
    ),
    "db_query_regression": FailureSpec(
        action="mitigate_query",
        culprit_role="db",
        metric_label="orders query p95 duration",
        metric_change="82 ms -> 4.6 s",
        primary_log="the same read path now spends most of its time inside a database statement",
        secondary_signal="database CPU and buffer reads rise with the slow query volume",
        alert_signal="query latency breached while connection wait stayed normal",
    ),
    "memory_leak": FailureSpec(
        action="restart_or_replace_instance",
        culprit_role="app",
        metric_label="application resident memory",
        metric_change="2.1 GiB -> 7.7 GiB",
        primary_log="heap usage grows between requests and the process is eventually killed by the runtime",
        secondary_signal="latency increases on one long-lived process while peer replicas stay steady",
        alert_signal="memory working-set threshold crossed without a matching traffic spike",
    ),
    "downstream_dependency_timeout": FailureSpec(
        action="restore_dependency",
        culprit_role="external",
        metric_label="outbound dependency p99 latency",
        metric_change="180 ms -> 7.1 s",
        primary_log="the client waited for a response from the external endpoint until its deadline expired",
        secondary_signal="the calling service shows deadline errors but its local queue is healthy",
        alert_signal="dependency health checks report slow responses from the remote endpoint",
    ),
    "cache_stampede": FailureSpec(
        action="clear_or_throttle_cache_pressure",
        culprit_role="cache",
        metric_label="cache miss ratio",
        metric_change="4% -> 92%",
        primary_log="many workers request the same expired key at once instead of sharing one fill",
        secondary_signal="database read traffic spikes while application request volume is flat",
        alert_signal="hot-key misses increase immediately after a shared expiry boundary",
    ),
    "kafka_consumer_lag": FailureSpec(
        action="drain_or_rebalance_workload",
        culprit_role="consumer",
        metric_label="event consumer backlog",
        metric_change="240 -> 184,000 records",
        primary_log="the consumer group polls slowly and committed offsets stop advancing",
        secondary_signal="producer ingress remains normal while event age grows in the queue",
        alert_signal="consumer delay exceeds the delivery objective for the affected partition",
    ),
    "thread_pool_exhaustion": FailureSpec(
        action="restart_or_replace_instance",
        culprit_role="app",
        metric_label="busy request-worker threads",
        metric_change="63% -> 100%",
        primary_log="new requests wait because every request worker is occupied by a blocked operation",
        secondary_signal="the process is alive and CPU is moderate, but queue depth keeps increasing",
        alert_signal="request-worker availability reaches zero on several replicas",
    ),
    "disk_io_saturation": FailureSpec(
        action="scale_capacity",
        culprit_role="db",
        metric_label="storage device await time",
        metric_change="8 ms -> 190 ms",
        primary_log="writes wait behind a saturated volume and filesystem sync calls take seconds",
        secondary_signal="database CPU is moderate while the device queue remains full",
        alert_signal="volume utilization and I/O wait exceed the storage budget",
    ),
    "dns_resolution_failure": FailureSpec(
        action="restore_dependency",
        culprit_role="dns",
        metric_label="service-name lookup p99",
        metric_change="3 ms -> 2.4 s",
        primary_log="temporary host lookup errors appear before clients can open a connection",
        secondary_signal="existing connections remain healthy while new destinations fail to resolve",
        alert_signal="the resolver reports elevated negative answers for the dependency name",
    ),
    "tls_certificate_expiration": FailureSpec(
        action="renew_certificate",
        culprit_role="cert",
        metric_label="secure-handshake failure rate",
        metric_change="0.1% -> 96%",
        primary_log="the remote endpoint presents a certificate outside its validity window",
        secondary_signal="network latency is normal but new secure sessions fail verification",
        alert_signal="certificate-validity checks report an expired credential on the endpoint",
    ),
    "rate_limit_misconfiguration": FailureSpec(
        action="revert_configuration",
        culprit_role="gateway",
        metric_label="gateway responses with status 429",
        metric_change="0.2% -> 41%",
        primary_log="legitimate callers are rejected by a quota that is lower than the intended service limit",
        secondary_signal="backend latency and capacity remain normal while rejection counts rise",
        alert_signal="quota rejection rate crosses the client-error threshold after a policy edit",
    ),
    "configuration_regression": FailureSpec(
        action="rollback_recent_change",
        culprit_role="app",
        metric_label="request error rate",
        metric_change="0.4% -> 18%",
        primary_log="the process accepts traffic but a newly enabled behavior returns invalid responses",
        secondary_signal="the error signature begins immediately after a feature setting changes",
        alert_signal="the application error budget burns without a dependency health change",
    ),
}


TOPOLOGIES: dict[str, dict[str, object]] = {
    "commerce_core": {
        "slice_group": "core",
        "components": {
            "gateway": "commerce-gateway",
            "app": "checkout-api",
            "auth": "fraud-service",
            "worker": "order-worker",
            "consumer": "order-consumer",
            "db": "orders-postgres",
            "cache": "orders-redis",
            "external": "bank-gateway",
            "queue": "order-events",
            "dns": "platform-dns",
            "cert": "bank-gateway",
        },
        "edges": (
            "commerce-gateway -> checkout-api",
            "checkout-api -> fraud-service",
            "checkout-api -> orders-postgres",
            "checkout-api -> orders-redis",
            "checkout-api -> bank-gateway",
            "order-worker -> order-events",
            "order-consumer -> orders-postgres",
        ),
    },
    "saas_api": {
        "slice_group": "core",
        "components": {
            "gateway": "public-api",
            "app": "billing-service",
            "auth": "auth-service",
            "worker": "billing-worker",
            "consumer": "billing-consumer",
            "db": "billing-postgres",
            "cache": "billing-redis",
            "external": "external-idp",
            "queue": "billing-events",
            "dns": "corp-dns",
            "cert": "external-idp",
        },
        "edges": (
            "public-api -> billing-service",
            "billing-service -> auth-service",
            "billing-service -> billing-postgres",
            "billing-service -> billing-redis",
            "billing-service -> external-idp",
            "billing-worker -> billing-events",
            "billing-consumer -> billing-postgres",
        ),
    },
    "event_processing": {
        "slice_group": "core",
        "components": {
            "gateway": "ingest-api",
            "app": "event-processor",
            "auth": "schema-service",
            "worker": "stream-worker",
            "consumer": "stream-consumer",
            "db": "events-postgres",
            "cache": "events-cache",
            "external": "object-store",
            "queue": "events-kafka",
            "dns": "cluster-dns",
            "cert": "object-store",
        },
        "edges": (
            "ingest-api -> event-processor",
            "event-processor -> schema-service",
            "event-processor -> events-kafka",
            "stream-consumer -> events-kafka",
            "stream-consumer -> events-postgres",
            "stream-worker -> object-store",
        ),
    },
    "worker_system": {
        "slice_group": "core",
        "components": {
            "gateway": "jobs-api",
            "app": "job-coordinator",
            "auth": "scheduler",
            "worker": "report-worker",
            "consumer": "job-consumer",
            "db": "jobs-db",
            "cache": "jobs-cache",
            "external": "smtp-relay",
            "queue": "jobs-queue",
            "dns": "service-dns",
            "cert": "smtp-relay",
        },
        "edges": (
            "jobs-api -> job-coordinator",
            "job-coordinator -> scheduler",
            "job-coordinator -> jobs-db",
            "report-worker -> jobs-queue",
            "job-consumer -> jobs-db",
            "job-coordinator -> smtp-relay",
        ),
    },
    "notification_platform": {
        "slice_group": "transfer",
        "components": {
            "gateway": "notification-edge",
            "app": "template-service",
            "auth": "recipient-service",
            "worker": "push-worker",
            "consumer": "delivery-consumer",
            "db": "delivery-database",
            "cache": "notification-cache",
            "external": "push-provider",
            "queue": "notification-stream",
            "dns": "notification-resolver",
            "cert": "push-provider",
        },
        "edges": (
            "notification-edge -> template-service",
            "template-service -> recipient-service",
            "template-service -> notification-cache",
            "push-worker -> notification-stream",
            "delivery-consumer -> delivery-database",
            "push-worker -> push-provider",
        ),
    },
    "identity_platform": {
        "slice_group": "transfer",
        "components": {
            "gateway": "identity-edge",
            "app": "session-service",
            "auth": "policy-service",
            "worker": "token-worker",
            "consumer": "audit-consumer",
            "db": "identity-database",
            "cache": "session-cache",
            "external": "partner-idp",
            "queue": "identity-events",
            "dns": "identity-resolver",
            "cert": "partner-idp",
        },
        "edges": (
            "identity-edge -> session-service",
            "session-service -> policy-service",
            "session-service -> session-cache",
            "session-service -> partner-idp",
            "audit-consumer -> identity-events",
            "audit-consumer -> identity-database",
        ),
    },
    "media_pipeline": {
        "slice_group": "transfer",
        "components": {
            "gateway": "media-ingress",
            "app": "transcode-service",
            "auth": "media-catalog",
            "worker": "transcode-worker",
            "consumer": "media-consumer",
            "db": "media-database",
            "cache": "media-cache",
            "external": "blob-store",
            "queue": "transcode-stream",
            "dns": "media-resolver",
            "cert": "blob-store",
        },
        "edges": (
            "media-ingress -> transcode-service",
            "transcode-service -> media-catalog",
            "transcode-service -> media-cache",
            "transcode-worker -> transcode-stream",
            "media-consumer -> media-database",
            "transcode-worker -> blob-store",
        ),
    },
    "analytics_pipeline": {
        "slice_group": "transfer",
        "components": {
            "gateway": "analytics-gateway",
            "app": "query-service",
            "auth": "metadata-service",
            "worker": "rollup-worker",
            "consumer": "rollup-consumer",
            "db": "metrics-warehouse",
            "cache": "query-cache",
            "external": "catalog-store",
            "queue": "analytics-stream",
            "dns": "analytics-resolver",
            "cert": "catalog-store",
        },
        "edges": (
            "analytics-gateway -> query-service",
            "query-service -> metadata-service",
            "query-service -> metrics-warehouse",
            "query-service -> query-cache",
            "rollup-consumer -> analytics-stream",
            "rollup-worker -> catalog-store",
        ),
    },
}


def validate_taxonomy() -> None:
    """Fail loudly if the frozen taxonomy is internally inconsistent."""

    if len(FAILURE_FAMILIES) != 12 or len(FAILURE_FAMILY_SET) != 12:
        raise ValueError("Experiment 02A requires exactly 12 failure families")
    if len(ACTIONS) != len(ACTION_SET):
        raise ValueError("actions must be unique")
    if set(FAILURE_SPECS) != FAILURE_FAMILY_SET:
        raise ValueError("every failure family needs one frozen spec")
    if any(spec.action not in ACTION_SET for spec in FAILURE_SPECS.values()):
        raise ValueError("failure spec references an unknown action")
    required_roles = {"gateway", "app", "consumer", "db", "cache", "external", "dns", "cert"}
    for name, topology in TOPOLOGIES.items():
        components = topology["components"]
        if not required_roles.issubset(components):
            raise ValueError(f"topology {name} is missing required roles")
