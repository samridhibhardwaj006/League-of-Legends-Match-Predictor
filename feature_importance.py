import pandas as pd
import matplotlib.pyplot as plt


def show_feature_importance(
    model,
    feature_names
):

    weights = (
        model.linear.weight
        .detach()
        .numpy()
        .flatten()
    )

    importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": weights
        }
    )

    importance = importance.sort_values(
        by="Importance",
        key=lambda x: abs(x),
        ascending=False
    )

    print(importance)

    plt.figure(
        figsize=(8, 6)
    )

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()
    plt.show()
