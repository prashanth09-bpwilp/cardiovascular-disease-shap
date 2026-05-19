import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Convert scaled test data back to DataFrame

X_test_df = pd.DataFrame(
    X_test,
    columns=feature_names
)

# SHAP Explainer

explainer = shap.TreeExplainer(rf_model)

shap_values = explainer.shap_values(X_test_df)

print("SHAP values generated")

# HANDLE MULTICLASS SHAP OUTPUT

shap_array = np.array(shap_values)

print("SHAP ARRAY SHAPE:", shap_array.shape)

# Normalize shape
# Expected final shape:(n_classes, n_samples, n_features)

if shap_array.ndim == 3:

    # Already correct
    if shap_array.shape[0] == len(np.unique(y)):

        shap_3d = shap_array

    # Convert from:
    # (samples, features, classes)
    elif shap_array.shape[2] == len(np.unique(y)):

        shap_3d = np.transpose(
            shap_array,
            (2, 0, 1)
        )

    else:
        raise ValueError(
            f"Unexpected SHAP shape: {shap_array.shape}"
        )

else:
    raise ValueError(
        f"Expected 3D SHAP array, got: {shap_array.shape}"
    )


target_classes = [
    "HIGH",
    "INTERMEDIARY",
    "LOW"
]

for i, cls in enumerate(target_classes):

    sv = shap_3d[i]

    # Create large figure
    plt.figure(figsize=(10, 8))

    # SHAP summary plot
    shap.summary_plot(
        sv,
        X_test_df,
        feature_names=feature_names,
        show=False,
        plot_size=None
    )

    # Customize title
    plt.title(
        f"SHAP Feature Importance — CVD Risk: {cls}",
        fontsize=16,
        pad=20
    )

    # Improve layout
    plt.tight_layout()

    # Show plot
    plt.show()
