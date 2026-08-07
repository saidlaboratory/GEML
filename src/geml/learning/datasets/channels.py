"""Honest representation-channel identities for Goal 6 materialization."""

from __future__ import annotations

from enum import StrEnum


class RepresentationChannel(StrEnum):
    """The approved four-arm representation contract."""

    AST_DAG = "ast_dag"
    PURE_EML_DAG = "pure_eml_dag"
    FREQUENT_MACRO_MOTIF_DAG = "frequent_macro_motif_dag"
    MOTIF_AST_FAIR_CONTROL = "motif_ast_fair_control"


class ChannelState(StrEnum):
    """Whether a channel can be materialized under the present approved scope."""

    AVAILABLE = "available"
    BLOCKED = "blocked"


CHANNEL_SOURCE_CONTRACT: dict[RepresentationChannel, tuple[ChannelState, str]] = {
    RepresentationChannel.AST_DAG: (ChannelState.AVAILABLE, "immutable Goal 5 AST-DAG export"),
    RepresentationChannel.PURE_EML_DAG: (
        ChannelState.AVAILABLE,
        "immutable Goal 5 strict pure-EML-DAG export",
    ),
    RepresentationChannel.FREQUENT_MACRO_MOTIF_DAG: (
        ChannelState.AVAILABLE,
        (
            "immutable Goal 5 frequent motif graph over official-v4 macro representation; "
            "not strict EML"
        ),
    ),
    RepresentationChannel.MOTIF_AST_FAIR_CONTROL: (
        ChannelState.AVAILABLE,
        (
            "train-only exact motif compression over AST-DAG with the frequent macro-motif "
            "dictionary count and MDL-bit budget"
        ),
    ),
}


def channel_state(channel: RepresentationChannel) -> ChannelState:
    """Return the declared availability state without inferring an alternative source."""

    return CHANNEL_SOURCE_CONTRACT[channel][0]


def channel_source_note(channel: RepresentationChannel) -> str:
    """Return the persisted source/provenance disclosure for one channel."""

    return CHANNEL_SOURCE_CONTRACT[channel][1]


def require_channel_available(channel: RepresentationChannel) -> None:
    """Fail loudly instead of replacing an unavailable channel with a convenient proxy."""

    if channel_state(channel) is not ChannelState.AVAILABLE:
        raise ValueError(f"channel {channel.value!r} is blocked: {channel_source_note(channel)}")
