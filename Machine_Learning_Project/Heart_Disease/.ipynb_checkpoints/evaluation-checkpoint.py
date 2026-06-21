# evaluation.py

import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score
)


def evaluate_model(model, train_loader, test_loader):

    model.eval()

    with torch.no_grad():

        # ------------------------
        # Predictions
        # ------------------------
        train_pred = model(train_loader.dataset.tensors[0])
        train_pred = torch.argmax(train_pred, dim=1)

        test_pred = model(test_loader.dataset.tensors[0])
        test_pred = torch.argmax(test_pred, dim=1)

        train_true = train_loader.dataset.tensors[1]
        test_true = test_loader.dataset.tensors[1]

    # ------------------------
    # Confusion Matrix
    # ------------------------
    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        train_true,
        train_pred
    )
    plt.title("Train Confusion Matrix")
    plt.show()

    plt.figure(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        test_true,
        test_pred
    )
    plt.title("Test Confusion Matrix")
    plt.show()

    # ------------------------
    # Classification Report
    # ------------------------
    print("\nClassification Report:\n")

    report = classification_report(
        test_true,
        test_pred
    )

    print(report)

    # ------------------------
    # F1 Score
    # ------------------------
    f1 = f1_score(
        test_true,
        test_pred
    )

    print(f"\nF1 Score: {f1:.4f}")

    # ------------------------
    # Precision / Recall / F1 Plot
    # ------------------------
    report_dict = classification_report(
        test_true,
        test_pred,
        output_dict=True
    )

    df = pd.DataFrame(report_dict).T.loc[
        ['0', '1'],
        ['precision', 'recall', 'f1-score']
    ]

    plt.figure(figsize=(6, 4))
    df.plot(kind='bar')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.title("Classification Metrics")
    plt.show()

    return report_dict