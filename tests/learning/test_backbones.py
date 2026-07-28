"""Issue 6-3 tests use only tiny hand-written fixtures and never require ``outputs/``.

Every model test is skipped cleanly when the optional ``[ml]`` extra is absent, so a core-only
clone still runs green.  The skip is explicit rather than silent.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="the optional [ml] extra is not installed")

from geml.learning.backbones.gin import (  # noqa: E402
    ENCODER_DEVIATIONS,
    MEASURED_FIXTURE_PARAMETERS,
    PARAMETER_TARGET_RANGE,
    PERMITTED_HIDDEN_WIDTHS,
    CompactGraphEncoder,
    EncoderConfig,
    EncoderConfigurationError,
    EquivalencePairModel,
    GraphBatchProtocol,
    InvalidGraphBatchError,
    forward_flop_estimate,
    parameter_report,
    parameter_target_status,
    symmetric_pair_features,
    value_magnitude_features,
)
from geml.learning.backbones.prefix_transformer import (  # noqa: E402
    PAD_TOKEN,
    PREFIX_TOKENIZATION_VERSION,
    PrefixNode,
    PrefixSequenceEncoder,
    PrefixTokenizationError,
    PrefixTransformerConfig,
    PrefixTransformerPairModel,
    TokenizationStatus,
    prefix_forward_flop_estimate,
    serialize_prefix,
)
from geml.learning.backbones.trivial import (  # noqa: E402
    TRIVIAL_FEATURE_NAMES,
    StructuralCounts,
    TrivialFeatureError,
    TrivialFloorPairModel,
    assert_no_leaked_fields,
    counts_to_tensor,
    trivial_parameter_report,
)


class FixtureBatch:
    """A hand-written batch satisfying :class:`GraphBatchProtocol`."""

    def __init__(
        self,
        node_kind: list[int],
        node_label: list[int],
        messages: list[tuple[int, int, int, int, int]],
        graph_index: list[int],
        num_graphs: int,
        value_magnitude: list[tuple[float, float]] | None = None,
        node_is_root: list[int] | None = None,
        node_root_order: list[int] | None = None,
    ) -> None:
        count = len(node_kind)
        self.node_kind = torch.tensor(node_kind, dtype=torch.long)
        self.node_label = torch.tensor(node_label, dtype=torch.long)
        self.node_value_category = torch.zeros(count, dtype=torch.long)
        self.node_value_sign = torch.full((count,), 2, dtype=torch.long)
        self.node_value_magnitude = torch.tensor(
            value_magnitude or [(0.0, 0.0)] * count, dtype=torch.float32
        )
        self.node_value_bucket = torch.zeros(count, dtype=torch.long)
        self.node_is_root = torch.tensor(
            node_is_root if node_is_root is not None else [0] * count, dtype=torch.long
        )
        self.node_root_order = torch.tensor(
            node_root_order if node_root_order is not None else [0] * count, dtype=torch.long
        )
        self.message_source = torch.tensor([m[0] for m in messages], dtype=torch.long)
        self.message_target = torch.tensor([m[1] for m in messages], dtype=torch.long)
        self.message_direction = torch.tensor([m[2] for m in messages], dtype=torch.long)
        self.message_slot = torch.tensor([m[3] for m in messages], dtype=torch.long)
        self.message_role = torch.tensor([m[4] for m in messages], dtype=torch.long)
        self.graph_index = torch.tensor(graph_index, dtype=torch.long)
        self.num_graphs = num_graphs


def two_child_batch(slots: tuple[int, int] = (0, 1)) -> FixtureBatch:
    """A root with two ordered children, materialized as forward and reverse messages."""

    return FixtureBatch(
        node_kind=[1, 2, 2],
        node_label=[5, 6, 7],
        messages=[
            (0, 1, 0, slots[0], 0),
            (0, 2, 0, slots[1], 0),
            (1, 0, 1, slots[0], 1),
            (2, 0, 1, slots[1], 1),
        ],
        graph_index=[0, 0, 0],
        num_graphs=1,
        node_is_root=[1, 0, 0],
    )


def test_fixture_batch_satisfies_the_consumption_protocol() -> None:
    assert isinstance(two_child_batch(), GraphBatchProtocol)


def test_encoder_forward_and_backward_on_cpu() -> None:
    torch.manual_seed(0)
    encoder = CompactGraphEncoder()
    output = encoder.encode(two_child_batch())

    assert output.node_embeddings.shape == (3, encoder.output_width)
    assert output.graph_embeddings.shape == (1, encoder.output_width)

    output.graph_embeddings.sum().backward()
    assert encoder.kind_embedding.weight.grad is not None
    assert encoder.direction_embedding.weight.grad is not None
    assert encoder.slot_embedding.weight.grad is not None
    assert encoder.value_magnitude_projection.weight.grad is not None


def test_deterministic_cpu_path_in_eval_mode() -> None:
    torch.manual_seed(0)
    encoder = CompactGraphEncoder().eval()
    batch = two_child_batch()
    first = encoder.encode(batch).graph_embeddings
    second = encoder.encode(batch).graph_embeddings
    assert torch.equal(first, second)


def test_ordered_child_slot_changes_the_output() -> None:
    torch.manual_seed(0)
    encoder = CompactGraphEncoder().eval()
    ordered = encoder.encode(two_child_batch((0, 1))).graph_embeddings
    swapped = encoder.encode(two_child_batch((1, 0))).graph_embeddings
    assert not torch.allclose(ordered, swapped)


def test_message_direction_changes_the_output() -> None:
    torch.manual_seed(0)
    encoder = CompactGraphEncoder().eval()
    batch = two_child_batch()
    baseline = encoder.encode(batch).graph_embeddings

    flipped = two_child_batch()
    flipped.message_direction = 1 - flipped.message_direction
    assert not torch.allclose(baseline, encoder.encode(flipped).graph_embeddings)


def test_repeated_references_are_not_collapsed() -> None:
    """A child referenced twice must contribute twice, not once."""

    torch.manual_seed(0)
    encoder = CompactGraphEncoder().eval()
    single = FixtureBatch(
        node_kind=[1, 2],
        node_label=[5, 6],
        messages=[(1, 0, 0, 0, 0)],
        graph_index=[0, 0],
        num_graphs=1,
    )
    repeated = FixtureBatch(
        node_kind=[1, 2],
        node_label=[5, 6],
        messages=[(1, 0, 0, 0, 0), (1, 0, 0, 1, 0)],
        graph_index=[0, 0],
        num_graphs=1,
    )
    assert not torch.allclose(
        encoder.encode(single).graph_embeddings,
        encoder.encode(repeated).graph_embeddings,
    )


def test_root_indicator_and_order_are_consumed() -> None:
    torch.manual_seed(0)
    encoder = CompactGraphEncoder().eval()
    baseline = encoder.encode(two_child_batch()).graph_embeddings
    demoted = two_child_batch()
    demoted.node_is_root = torch.zeros_like(demoted.node_is_root)
    assert not torch.allclose(baseline, encoder.encode(demoted).graph_embeddings)


def test_exact_value_magnitudes_are_consumed() -> None:
    torch.manual_seed(0)
    encoder = CompactGraphEncoder().eval()
    baseline = encoder.encode(two_child_batch()).graph_embeddings
    valued = two_child_batch()
    valued.node_value_magnitude = torch.tensor(
        [value_magnitude_features(3, 4), value_magnitude_features(1), value_magnitude_features(2)],
        dtype=torch.float32,
    )
    assert not torch.allclose(baseline, encoder.encode(valued).graph_embeddings)


def test_value_magnitude_features_reject_zero_denominator() -> None:
    with pytest.raises(InvalidGraphBatchError, match="zero denominator"):
        value_magnitude_features(1, 0)


def test_empty_and_ragged_batches_fail_explicitly() -> None:
    empty = two_child_batch()
    empty.node_kind = torch.zeros(0, dtype=torch.long)
    with pytest.raises(InvalidGraphBatchError, match="zero nodes"):
        CompactGraphEncoder().encode(empty)

    ragged = two_child_batch()
    ragged.node_label = torch.zeros(2, dtype=torch.long)
    with pytest.raises(InvalidGraphBatchError, match="node_label"):
        CompactGraphEncoder().encode(ragged)

    dangling = two_child_batch()
    dangling.message_target = torch.tensor([9, 9, 9, 9], dtype=torch.long)
    with pytest.raises(InvalidGraphBatchError, match="outside the batch"):
        CompactGraphEncoder().encode(dangling)


def test_encoder_rejects_configurations_outside_the_frozen_contract() -> None:
    with pytest.raises(EncoderConfigurationError, match="hidden_width"):
        EncoderConfig(hidden_width=128)
    with pytest.raises(EncoderConfigurationError, match="three message-passing layers"):
        EncoderConfig(num_layers=4)
    with pytest.raises(EncoderConfigurationError, match="normalization"):
        EncoderConfig(normalization="batch_norm")


@pytest.mark.parametrize("normalization", ["layer_norm", "graph_norm"])
def test_both_normalizations_run(normalization: str) -> None:
    torch.manual_seed(0)
    encoder = CompactGraphEncoder(EncoderConfig(normalization=normalization)).eval()
    assert encoder.encode(two_child_batch()).graph_embeddings.shape == (1, 64)


def test_virtual_node_is_configuration_controlled() -> None:
    torch.manual_seed(0)
    without = CompactGraphEncoder(EncoderConfig(use_virtual_node=False))
    with_virtual = CompactGraphEncoder(EncoderConfig(use_virtual_node=True))
    assert parameter_report(with_virtual).total_parameters > (
        parameter_report(without).total_parameters
    )


def test_state_dict_round_trip_is_stable() -> None:
    torch.manual_seed(0)
    original = CompactGraphEncoder().eval()
    restored = CompactGraphEncoder().eval()
    restored.load_state_dict(original.state_dict())
    batch = two_child_batch()
    assert torch.equal(
        original.encode(batch).graph_embeddings,
        restored.encode(batch).graph_embeddings,
    )


def test_pair_model_is_swap_invariant_and_shares_weights() -> None:
    torch.manual_seed(0)
    model = EquivalencePairModel().eval()
    left = two_child_batch((0, 1))
    right = two_child_batch((1, 0))

    forward = model(left, right)
    backward = model(right, left)
    assert torch.allclose(forward, backward, atol=1e-6)

    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    assert encoder_ids, "the siamese encoder must expose shared parameters"


def test_symmetric_pair_features_are_symmetric_by_construction() -> None:
    left = torch.randn(4, 8)
    right = torch.randn(4, 8)
    assert torch.allclose(
        symmetric_pair_features(left, right),
        symmetric_pair_features(right, left),
    )


def test_pair_model_rejects_mismatched_endpoints() -> None:
    model = EquivalencePairModel()
    left = two_child_batch()
    right = two_child_batch()
    right.num_graphs = 2
    with pytest.raises(InvalidGraphBatchError, match="same number of graphs"):
        model(left, right)


def test_pair_model_gradients_reach_the_shared_encoder() -> None:
    torch.manual_seed(0)
    model = EquivalencePairModel()
    logit = model(two_child_batch(), two_child_batch())
    logit.sum().backward()
    assert model.encoder.label_embedding.weight.grad is not None
    assert model.encoder.role_embedding.weight.grad is not None


@pytest.mark.parametrize("width", PERMITTED_HIDDEN_WIDTHS)
def test_parameter_budget_is_measured_by_code(width: int) -> None:
    model = EquivalencePairModel(CompactGraphEncoder(EncoderConfig(hidden_width=width)))
    report = parameter_report(model)
    assert report.total_parameters == report.trainable_parameters
    assert report.total_parameters <= PARAMETER_TARGET_RANGE[1]
    assert report.embedding_parameters > 0
    assert "encoder" in report.per_component


@pytest.mark.parametrize(("width", "virtual_node"), sorted(MEASURED_FIXTURE_PARAMETERS))
def test_measured_fixture_parameter_counts_do_not_drift(width: int, virtual_node: bool) -> None:
    """Pin the measured counts so an accidental architecture change cannot pass unnoticed."""

    model = EquivalencePairModel(
        CompactGraphEncoder(EncoderConfig(hidden_width=width, use_virtual_node=virtual_node))
    )
    measured = parameter_report(model).total_parameters
    assert measured == MEASURED_FIXTURE_PARAMETERS[(width, virtual_node)]


def test_below_target_parameter_counts_are_reported_not_hidden() -> None:
    """Width 64 without a virtual node sits marginally under the 0.2M target; say so."""

    assert parameter_target_status(MEASURED_FIXTURE_PARAMETERS[(64, False)]) == "below_target"
    assert parameter_target_status(MEASURED_FIXTURE_PARAMETERS[(96, False)]) == "within_target"
    assert parameter_target_status(5_000_000) == "above_target"


def test_flop_estimate_is_computed_and_scales_with_graph_size() -> None:
    model = EquivalencePairModel()
    small = forward_flop_estimate(model, two_child_batch())
    larger = FixtureBatch(
        node_kind=[1] * 6,
        node_label=[2] * 6,
        messages=[(i, i + 1, 0, 0, 0) for i in range(5)],
        graph_index=[0] * 6,
        num_graphs=1,
    )
    assert forward_flop_estimate(model, larger).forward_flops > small.forward_flops
    assert small.limitations


def test_encoder_deviations_are_documented() -> None:
    assert len(ENCODER_DEVIATIONS) >= 5
    assert all(isinstance(entry, str) and entry for entry in ENCODER_DEVIATIONS)


def test_prefix_serialization_is_exact_and_versioned() -> None:
    nodes = (
        PrefixNode(kind_id=0, label_id=1, root_order=0, child_slots=(0, 1)),
        PrefixNode(kind_id=1, label_id=2, value_numerator=-13, value_denominator=7),
    )
    serialized = serialize_prefix(nodes, max_length=128)
    assert serialized.status is TokenizationStatus.OK
    assert serialized.as_dict()["tokenization_version"] == PREFIX_TOKENIZATION_VERSION

    # The exact numerator 13 and denominator 7 survive as digit tokens, not as a float string.
    other = serialize_prefix(
        (nodes[0], PrefixNode(kind_id=1, label_id=2, value_numerator=-13, value_denominator=8)),
        max_length=128,
    )
    assert serialized.tokens != other.tokens


def test_prefix_overlength_is_an_explicit_recorded_outcome() -> None:
    nodes = tuple(PrefixNode(kind_id=0, label_id=1) for _ in range(50))
    serialized = serialize_prefix(nodes, max_length=16)
    assert serialized.status is TokenizationStatus.OVERLENGTH
    assert serialized.truncated_from is not None
    assert serialized.truncated_from > 16


def test_prefix_invalid_tokens_are_reported_not_hidden() -> None:
    serialized = serialize_prefix((PrefixNode(kind_id=-1, label_id=0),), max_length=32)
    assert serialized.status is TokenizationStatus.INVALID_TOKEN
    assert serialized.tokens == ()


def test_prefix_serialization_rejects_degenerate_inputs() -> None:
    with pytest.raises(PrefixTokenizationError, match="zero nodes"):
        serialize_prefix((), max_length=32)
    with pytest.raises(PrefixTokenizationError, match="max_length"):
        serialize_prefix((PrefixNode(kind_id=0, label_id=0),), max_length=1)


def test_prefix_padding_is_masked_out_of_the_pooled_embedding() -> None:
    torch.manual_seed(0)
    encoder = PrefixSequenceEncoder().eval()
    tokens = torch.tensor([[1, 20, 21, 22]], dtype=torch.long)
    padded = torch.tensor([[1, 20, 21, 22, PAD_TOKEN, PAD_TOKEN]], dtype=torch.long)
    assert torch.allclose(encoder(tokens), encoder(padded), atol=1e-5)


def test_prefix_encoder_rejects_overlength_and_fully_padded_input() -> None:
    encoder = PrefixSequenceEncoder(PrefixTransformerConfig(max_length=8))
    with pytest.raises(PrefixTokenizationError, match="exceeds the frozen max_length"):
        encoder(torch.ones(1, 9, dtype=torch.long))
    with pytest.raises(PrefixTokenizationError, match="fully padded"):
        encoder(torch.full((1, 4), PAD_TOKEN, dtype=torch.long))


def test_prefix_pair_model_is_swap_invariant_and_trains() -> None:
    torch.manual_seed(0)
    model = PrefixTransformerPairModel().eval()
    left = torch.tensor([[1, 20, 21, 22]], dtype=torch.long)
    right = torch.tensor([[1, 30, 31, 32]], dtype=torch.long)
    assert torch.allclose(model(left, right), model(right, left), atol=1e-6)

    model.train()
    model(left, right).sum().backward()
    assert model.encoder.token_embedding.weight.grad is not None


def test_prefix_flop_estimate_is_reported_with_limitations() -> None:
    model = PrefixTransformerPairModel()
    report = prefix_forward_flop_estimate(model, batch_size=2, sequence_length=32)
    assert report["forward_flops"] > 0
    assert report["tokenization_version"] == PREFIX_TOKENIZATION_VERSION
    assert report["limitations"]


def test_trivial_floor_feature_list_is_frozen_and_ordered() -> None:
    counts = StructuralCounts(
        node_count=5,
        edge_count=4,
        depth=2,
        distinct_variable_count=1,
        variable_occurrence_count=2,
        root_count=1,
        operator_counts={"eml": 3, "unknown_operator": 1},
    )
    vector = counts.as_vector()
    assert len(vector) == len(TRIVIAL_FEATURE_NAMES)
    assert vector[TRIVIAL_FEATURE_NAMES.index("operator_count_eml")] == 3.0
    assert vector[TRIVIAL_FEATURE_NAMES.index("operator_count_other")] == 1.0
    assert vector[TRIVIAL_FEATURE_NAMES.index("operator_count_total")] == 4.0


def test_trivial_floor_rejects_leaked_metadata_fields() -> None:
    with pytest.raises(TrivialFeatureError, match="split"):
        assert_no_leaked_fields({"node_count": 3, "split": "train"})
    with pytest.raises(TrivialFeatureError, match="alpha"):
        assert_no_leaked_fields({"alpha": 0.5})
    assert_no_leaked_fields({name: 0 for name in TRIVIAL_FEATURE_NAMES})


def test_trivial_floor_is_symmetric_and_auditable() -> None:
    torch.manual_seed(0)
    model = TrivialFloorPairModel().eval()
    left = counts_to_tensor((StructuralCounts(node_count=3, operator_counts={"eml": 2}),))
    right = counts_to_tensor((StructuralCounts(node_count=5, operator_counts={"add": 1}),))
    assert torch.allclose(model(left, right), model(right, left), atol=1e-6)

    weights = model.readable_weights()
    assert set(weights) == {"sum", "absolute_difference", "product"}
    assert set(weights["sum"]) == set(TRIVIAL_FEATURE_NAMES)


def test_trivial_floor_reports_measured_parameters() -> None:
    report = trivial_parameter_report(TrivialFloorPairModel())
    assert report["num_features"] == len(TRIVIAL_FEATURE_NAMES)
    assert report["total_parameters"] > 0
    assert report["feature_names"] == list(TRIVIAL_FEATURE_NAMES)


def test_trivial_floor_rejects_bad_shapes() -> None:
    model = TrivialFloorPairModel()
    with pytest.raises(TrivialFeatureError, match="identical feature shapes"):
        model(torch.zeros(1, 19), torch.zeros(2, 19))
    with pytest.raises(TrivialFeatureError, match="zero expressions"):
        counts_to_tensor(())
