"""CertiQ index architecture — model, certificate, interaction, geometry."""

from certiqnet.dispatcher.certiq.certificate import (
    DifferentiableKLProjection,
    arrival_coordinate,
    kl_project_linear,
    normalize_policy,
    policy_entropy,
)
from certiqnet.dispatcher.certiq.interaction import (
    DispatchInteractionEncoder,
    index_token_features,
)
from certiqnet.dispatcher.certiq.index_model import CertiQIndexModel, MarginalIndexHead
from certiqnet.dispatcher.certiq.geometry import CertifiedGeometry

__all__ = [
    "CertiQIndexModel",
    "DifferentiableKLProjection",
    "DispatchInteractionEncoder",
    "MarginalIndexHead",
    "CertifiedGeometry",
    "index_token_features",
    "arrival_coordinate",
    "kl_project_linear",
    "normalize_policy",
    "policy_entropy",
]
