import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
import argparse
import json
from pathlib import Path

from model import ACTSNetV2
from modules.interpretability import InterpretabilityModule


@torch.no_grad()
def evaluate_model(model_path: str, test_data: np.ndarray, test_labels: np.ndarray,
                   device='cpu', output_path=None):
    """Load model and run full evaluation with interpretability."""

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    model = ACTSNetV2(
        seq_len=test_data.shape[-1],
        patch_len=config.get('patch_len', 32),
        d_model=config.get('d_model', 128),
        n_freqlens_layers=config.get('n_freqlens_layers', 2),
        dropout=config.get('dropout', 0.1),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run predictions
    test_tensor = torch.FloatTensor(test_data).to(device)
    logits, embeddings = model(test_tensor)
    probs = torch.softmax(logits, dim=-1)
    preds = probs.argmax(dim=-1).cpu().numpy()
    probs_np = probs.cpu().numpy()

    # Classification metrics
    print("=" * 60)
    print("ACTSNet v2 Evaluation Results")
    print("=" * 60)
    print(classification_report(
        test_labels, preds, target_names=['Non-Responder', 'Responder']
    ))
    try:
        auc = roc_auc_score(test_labels, probs_np[:, 1])
        print(f"AUC-ROC: {auc:.4f}")
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(test_labels, preds)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0, 0]}  FP={cm[0, 1]}")
    print(f"  FN={cm[1, 0]}  TP={cm[1, 1]}")

    results = {
        'accuracy': accuracy_score(test_labels, preds),
        'f1': f1_score(test_labels, preds, zero_division=0),
        'auc_roc': auc,
        'confusion_matrix': cm.tolist(),
        'per_sample': [
            {
                'index': i,
                'true_label': int(test_labels[i]),
                'predicted_label': int(preds[i]),
                'prob_responder': float(probs_np[i, 1]),
            }
            for i in range(len(test_labels))
        ],
    }

    # Generate interpretability reports
    interp = InterpretabilityModule()
    print("\n--- Sample Interpretability Reports ---")
    for i in range(min(5, len(test_data))):
        sample = test_tensor[i:i + 1]
        attribution = interp.generate_attribution(model, sample, test_labels[i])
        report = interp.format_clinical_report(attribution)
        print(f"\n--- Sample {i} (true={test_labels[i]}) ---")
        print(report)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove non-serializable items for JSON
        json_results = {k: v for k, v in results.items()}
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACTSNet v2")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to test data .npy (N, 7, 5, T)")
    parser.add_argument("--labels_path", type=str, required=True,
                        help="Path to test labels .npy (N,)")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    test_data = np.load(args.data_path)
    test_labels = np.load(args.labels_path)
    evaluate_model(args.checkpoint, test_data, test_labels, args.device, args.output)


if __name__ == '__main__':
    main()
