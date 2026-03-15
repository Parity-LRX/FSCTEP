from __future__ import annotations

from molecular_force_field.cli import main as main_cli


def test_main_cli_dispatch_merge_multifidelity(monkeypatch) -> None:
    called: dict[str, list[str]] = {}

    def fake_submain() -> None:
        import sys

        called["argv"] = list(sys.argv)

    monkeypatch.setattr(
        "molecular_force_field.cli.merge_multifidelity_h5.main",
        fake_submain,
    )

    main_cli.main(
        [
            "--merge-multifidelity",
            "--inputs",
            "a.h5",
            "b.h5",
            "--fidelity-ids",
            "0",
            "1",
            "--output-h5",
            "merged.h5",
            "--output-fidelity-npy",
            "fid.npy",
        ]
    )

    assert called["argv"][1:] == [
        "--inputs",
        "a.h5",
        "b.h5",
        "--fidelity-ids",
        "0",
        "1",
        "--output-h5",
        "merged.h5",
        "--output-fidelity-npy",
        "fid.npy",
    ]
