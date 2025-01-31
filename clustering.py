from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from util import load_json, load_csv


def perform_clustering(df: pd.DataFrame):
    """
    Perform clustering on the provided DataFrame and visualize the results using
    KMeans, PCA, and hierarchical clustering (dendrogram).
    """
    X = df

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Elbow Method: Calculate inertia for different values of k
    inertia = [
        KMeans(n_clusters=k, random_state=42).fit(X_scaled).inertia_
        for k in range(1, 11)
    ]

    # Plot the Elbow Method (Inertia vs Number of Clusters)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, "bo-", markersize=8, label="Inertia")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K")
    plt.grid()
    plt.legend()
    plt.show()

    # Calculate silhouette score for each k
    silhouette_scores = [
        silhouette_score(
            X_scaled, KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
        )
        for k in range(2, 11)
    ]

    # Print silhouette scores as percentage values
    print("Silhouette Scores:")
    for k, score in zip(range(2, 11), silhouette_scores):
        print(f"Silhouette Score for k={k}: {score*100:.2f}%")

    # Fit KMeans with the optimal number of clusters (e.g., k=3 based on previous analysis)
    best_n_clusters = range(2, 11)[silhouette_scores.index(max(silhouette_scores))]  # type: ignore
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df["Cluster"] = labels

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Scatter plot of PCA reduced data with KMeans cluster centers
    plt.figure(figsize=(8, 5))
    plt.scatter(
        X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=50, label="Data Points"
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("KMeans Cluster Visualization (PCA Reduced)")
    plt.legend()
    plt.show()

    # Hierarchical clustering and dendrogram
    linked = linkage(X_scaled, method="ward")

    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(
        linked, truncate_mode="lastp", p=12, leaf_rotation=90.0, leaf_font_size=10.0
    )
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

    return df


if __name__ == "__main__":

    file = load_json("C:/Users/pujar/Desktop/Project/DS/json/file.json")
    df = load_csv(file["output"])
    perform_clustering(df)
