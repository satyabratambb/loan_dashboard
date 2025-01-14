# utils.py
def plot_feature_importance(feature_names, importances):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names, importances, align="center")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance in Random Forest")
    plt.gca().invert_yaxis()
    return fig
