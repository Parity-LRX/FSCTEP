"""
O(2)-named wrapper for the local-frame SO(2) accelerated ICTD layer.

At the message-passing level this currently shares the same local-frame kernel as
the SO(2) variant: the acceleration comes from edge-aligned local SO(2) mixing,
while the public model interface remains aligned with pure_cartesian_ictd_layers.
"""

from __future__ import annotations

from molecular_force_field.models.pure_cartesian_ictd_layers_so2 import (
    LocalMultipleContractionO2,
    PureCartesianICTDSO2TransformerLayer,
)


class PureCartesianICTDO2TransformerLayer(PureCartesianICTDSO2TransformerLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_group = "o2"
        if self.save_readout_mode == "multiple-contraction":
            self.multiple_contraction_last = LocalMultipleContractionO2(
                in_channels=self.channels,
                hidden_channels=self.channels,
                lmax=self.lmax,
                correlation=self.save_contraction_order,
                ictd_tp_path_policy=self._local_ictd_tp_path_policy,
                ictd_tp_max_rank_other=self._local_ictd_tp_max_rank_other,
                internal_compute_dtype=self.tp2_layers[0].internal_compute_dtype if len(self.tp2_layers) > 0 else None,
            )
            self.multiple_contraction_mix = LocalMultipleContractionO2(
                in_channels=self.channels * self.num_interaction,
                hidden_channels=self.save_multiple_mix_channels,
                lmax=self.lmax,
                correlation=self.save_contraction_order,
                ictd_tp_path_policy=self._local_ictd_tp_path_policy,
                ictd_tp_max_rank_other=self._local_ictd_tp_max_rank_other,
                internal_compute_dtype=self.tp2_layers[0].internal_compute_dtype if len(self.tp2_layers) > 0 else None,
            )
