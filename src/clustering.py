import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from .config import RANDOM_STATE, MAX_CLUSTERS


def perform_clustering(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    silhouette_scores = {}
    for k in range(2, MAX_CLUSTERS + 1):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )
        labels = model.fit_predict(X_scaled)
        silhouette_scores[k] = silhouette_score(X_scaled, labels)

    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Best k based on silhouette score: {best_k}")

    final_model = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10
    )
    labels = final_model.fit_predict(X_scaled)

    df_out = df.copy()
    df_out["Cluster"] = labels

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis")
    plt.title("KMeans Clusters (PCA)")
    plt.show()

    linked = linkage(X_scaled, method="ward")
    dendrogram(linked, truncate_mode="lastp", p=12)
    plt.title("Hierarchical Dendrogram")
    plt.show()

    return df_out
