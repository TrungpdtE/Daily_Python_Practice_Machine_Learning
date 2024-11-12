

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Tạo dữ liệu mẫu với make_blobs
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Hiển thị dữ liệu ban đầu
plt.scatter(X[:, 0], X[:, 1], s=50, c='b')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Dữ liệu ban đầu")
plt.show()

# Khởi tạo các biến để lưu trữ các chỉ số đánh giá
sil_score = []
davies_score = []
elbow_score = []

# Thử với các giá trị K từ 2 đến 10
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    
    # Dự đoán cụm
    labels = kmeans.labels_
    
    # Tính toán các chỉ số silhouette, davies_bouldin và inertia (cho elbow method)
    sil_score.append(silhouette_score(X, labels))
    davies_score.append(davies_bouldin_score(X, labels))
    elbow_score.append(kmeans.inertia_)

# Vẽ biểu đồ silhouette score
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(range(2, 11), sil_score, marker='o', color='g')
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Analysis")

# Vẽ biểu đồ davies_bouldin score
plt.subplot(1, 3, 2)
plt.plot(range(2, 11), davies_score, marker='o', color='r')
plt.xlabel("K")
plt.ylabel("Davies-Bouldin Score")
plt.title("Davies-Bouldin Analysis")

# Vẽ biểu đồ Elbow Method
plt.subplot(1, 3, 3)
plt.plot(range(2, 11), elbow_score, marker='o', color='b')
plt.xlabel("K")
plt.ylabel("Inertia (SSE)")
plt.title("Elbow Method")

plt.tight_layout()
plt.show()

# Sử dụng KMeans với giá trị K tối ưu (giả sử từ elbow method)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans.fit(X)
centers = kmeans.cluster_centers_

# Vẽ các cụm và tâm cụm
plt.scatter(X[:, 0], X[:, 1], s=50, c=kmeans.labels_, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel("X1")
plt.ylabel("X2")
plt.title(f"Clustering với K = {optimal_k}")
plt.show()