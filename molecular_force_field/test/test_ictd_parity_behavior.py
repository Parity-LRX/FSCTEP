from __future__ import annotations

import torch

from molecular_force_field.models.ictd_irreps import (
    parity_letter_to_sign,
    direction_harmonics_irreps,
    parse_irreps_string,
)


def test_parse_irreps_string_preserves_parity_tag() -> None:
    assert parse_irreps_string("1e") == [(1, 1, parity_letter_to_sign("e"))]
    assert parse_irreps_string("1o") == [(1, 1, parity_letter_to_sign("o"))]
    assert parse_irreps_string("1e") != parse_irreps_string("1o")


def test_direction_harmonics_irreps_enforces_geometric_parity() -> None:
    n = torch.tensor([[0.2, -0.3, 0.5]], dtype=torch.float64)
    y_o = direction_harmonics_irreps(n, "1o")
    assert y_o.shape == (1, 3)
    try:
        direction_harmonics_irreps(n, "1e")
    except ValueError as exc:
        assert "only supports geometric parity" in str(exc)
    else:
        raise AssertionError("direction_harmonics_irreps should reject non-geometric parity for l=1")
