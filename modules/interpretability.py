import torch
import torch.nn as nn
import numpy as np


class InterpretabilityModule(nn.Module):
    """
    Clinical interpretability module for ACTSNet v2.

    Aggregates attribution signals from:
    1. FreqLens attention -> which frequency components matter
    2. SubBandFusion -> which EEG bands matter
    3. ChannelAttention -> which electrodes & connectivity patterns matter
    4. SpatialSpectralGraph -> learned graph structure

    This module does NOT affect the forward pass (gradient-free).
    It is called separately for generating clinical reports.
    """

    def __init__(self, electrode_names=None, band_names=None):
        super().__init__()
        self.electrode_names = electrode_names or ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8']
        self.band_names = band_names or ['delta', 'theta', 'alpha', 'beta', 'gamma']

    @torch.no_grad()
    def generate_attribution(self, model, x, label=None):
        """
        Generate full attribution map for a single sample.

        Args:
            model: ACTSNetV2 model instance
            x: (1, 7, 5, T) — single EEG sample
            label: optional true label

        Returns:
            dict with attribution information
        """
        model.eval()
        result = {}

        # Forward pass
        logits, embeddings = model(x)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1).item()
        confidence = probs.max(dim=-1).values.item()

        result['prediction'] = pred
        result['confidence'] = confidence

        # 1. Band importance from SubBandFusion
        if hasattr(model, 'subband_fusion'):
            band_imp = model.subband_fusion.get_band_importance()
            if band_imp:
                result['band_importance'] = band_imp

        # 2. Channel connectivity from ChannelAttention
        if hasattr(model, 'channel_attention'):
            conn, names = model.channel_attention.get_channel_connectivity()
            if conn is not None:
                result['channel_connectivity'] = conn
                result['electrode_importance'] = conn.mean(axis=1)
                result['electrode_names'] = names

        # 3. Frequency attribution from FreqLens
        if hasattr(model, 'freqlens_layers') and hasattr(model, '_last_patch_features'):
            if model._last_patch_features is not None:
                freq_attr = model.freqlens_layers[0].get_freq_attribution(
                    model._last_patch_features
                )
                result['freq_attribution'] = freq_attr.mean(dim=[0, 1]).cpu().numpy()

        return result

    def format_clinical_report(self, attribution: dict) -> str:
        """Format attribution into a readable clinical report string."""
        lines = []
        lines.append("=" * 60)
        lines.append("ACTSNet v2 -- Clinical Interpretability Report")
        lines.append("=" * 60)

        pred_label = "Responder" if attribution['prediction'] == 1 else "Non-Responder"
        lines.append(f"Prediction: {pred_label}")
        lines.append(f"Confidence: {attribution['confidence']:.2%}")
        lines.append("")

        if 'band_importance' in attribution:
            lines.append("--- Frequency Band Importance ---")
            sorted_bands = sorted(
                attribution['band_importance'].items(),
                key=lambda x: x[1], reverse=True,
            )
            for band, score in sorted_bands:
                bar = "#" * int(score * 20)
                lines.append(f"  {band:>6s}: {score:.3f} {bar}")
            lines.append("")

        if 'electrode_importance' in attribution:
            lines.append("--- Electrode Importance ---")
            names = attribution.get('electrode_names', self.electrode_names)
            importances = attribution['electrode_importance']
            sorted_elec = sorted(
                zip(names, importances),
                key=lambda x: x[1], reverse=True,
            )
            for name, score in sorted_elec:
                bar = "#" * int(score * 20)
                lines.append(f"  {name:>4s}: {score:.3f} {bar}")

        lines.append("=" * 60)
        return "\n".join(lines)
