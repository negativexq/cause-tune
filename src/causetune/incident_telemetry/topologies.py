"""Bounded deterministic topology catalog for Experiment 03B."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ServiceComponent, ServiceTopology, TopologyEdge


@dataclass(frozen=True)
class TopologyFamilySpec:
    family_id: str
    roles: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not self.family_id.strip() or not self.roles:
            raise ValueError("topology family requires an ID and roles")
        if len(self.roles) != len(set(self.roles)):
            raise ValueError(f"duplicate roles in topology family: {self.family_id}")
        role_set = set(self.roles)
        for source, target in self.edges:
            if source not in role_set or target not in role_set or source == target:
                raise ValueError(f"invalid topology edge in {self.family_id}: {source}->{target}")


TOPOLOGY_FAMILIES: tuple[TopologyFamilySpec, ...] = (
    TopologyFamilySpec("WEB_DB", ("frontend", "api", "postgres"), (("frontend", "api"), ("api", "postgres"))),
    TopologyFamilySpec(
        "WEB_CACHE_DB",
        ("frontend", "api", "redis", "postgres"),
        (("frontend", "api"), ("api", "redis"), ("api", "postgres")),
    ),
    TopologyFamilySpec(
        "CHECKOUT",
        ("frontend", "checkout_api", "payment_api", "redis", "postgres"),
        (
            ("frontend", "checkout_api"),
            ("checkout_api", "payment_api"),
            ("checkout_api", "redis"),
            ("checkout_api", "postgres"),
        ),
    ),
    TopologyFamilySpec(
        "ASYNC_WORKER",
        ("api", "broker", "worker", "postgres"),
        (("api", "broker"), ("broker", "worker"), ("worker", "postgres")),
    ),
    TopologyFamilySpec(
        "EVENT_PIPELINE",
        ("producer", "kafka", "consumer", "database"),
        (("producer", "kafka"), ("kafka", "consumer"), ("consumer", "database")),
    ),
    TopologyFamilySpec(
        "MULTI_SERVICE",
        ("gateway", "orders_api", "inventory_api", "payment_api"),
        (("gateway", "orders_api"), ("gateway", "inventory_api"), ("gateway", "payment_api")),
    ),
)
TOPOLOGY_FAMILY_BY_ID = {item.family_id: item for item in TOPOLOGY_FAMILIES}


def validate_topology_catalog() -> None:
    if len(TOPOLOGY_FAMILIES) != len(TOPOLOGY_FAMILY_BY_ID):
        raise ValueError("topology catalog contains duplicate family IDs")
    for family in TOPOLOGY_FAMILIES:
        if len(family.edges) != len(set(family.edges)):
            raise ValueError(f"topology family contains duplicate edges: {family.family_id}")


def build_topology(family_id: str) -> ServiceTopology:
    """Build one stable graph; no random graph generation is permitted."""

    try:
        spec = TOPOLOGY_FAMILY_BY_ID[family_id]
    except KeyError as exc:
        raise ValueError(f"unknown topology family: {family_id!r}") from exc
    components = tuple(
        ServiceComponent(
            component_id=f"{family_id.lower()}_{role}",
            role=role,
            service_name=f"{role.replace('_', '-')}-{family_id.lower()}",
        )
        for role in spec.roles
    )
    role_to_id = {component.role: component.component_id for component in components}
    edges = tuple(TopologyEdge(role_to_id[source], role_to_id[target], "depends_on") for source, target in spec.edges)
    return ServiceTopology(
        topology_id=f"topology_{family_id.lower()}",
        topology_family=family_id,
        components=components,
        edges=edges,
    )


validate_topology_catalog()
