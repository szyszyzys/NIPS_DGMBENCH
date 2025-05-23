import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn import mixture

def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances

class Kmeans():
    def __init__(self, k=2, max_iterations=2000, varepsilon=1e-5, resetTimes=10):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        self.resetTimes = resetTimes

    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))

        idx = random.sample(range(n_samples),self.k)
        while all(X[idx[0]] == X[idx[1]]):
            idx = random.sample(range(n_samples),self.k)

        for i in range(self.k):
            centroid = X[idx[i]]
            centroids[i] = centroid
        return centroids

    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    def get_cluster_labels_new(self, centroids, clusters, X):
        dist_list = []
        for centroid in centroids:
            dist = np.sqrt(np.sum(np.square(centroid)))
            dist_list.append(dist)
        copy = dist_list.copy()
        copy.sort()
        idx = np.zeros(len(centroids))
        for j in range(len(centroids)):
            for i in range(len(centroids)):
                if dist_list[j] == copy[i]:
                    idx[j] = i

        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = idx[cluster_i]
        return y_pred

    def predict(self, X):
        times = 0
        resultGood = False
        while(times<self.resetTimes) and not resultGood:
            times = times + 1
            centroids = self.init_random_centroids(X)

            for i in range(self.max_iterations):
                clusters = self.create_clusters(centroids, X)
                former_centroids = centroids

                centroids = self.update_centroids(clusters, X)

                diff = centroids - former_centroids
                if diff.any() < self.varepsilon:
                    break
            if(abs(len(clusters[0])-len(clusters[1])) < 4):
                resultGood = True

        return self.get_cluster_labels_new(centroids, clusters, X)

def Classify_kmeans(X):
    pca = PCA(n_components=2)
    newX = pca.fit_transform(X)

    Clf = Kmeans(k=2)
    y_pred = Clf.predict(newX)
    
    benign = 0
    mali = 0
    for f in y_pred:
        if f==1:
            benign+=1
        else:
            mali+=1

    if benign < mali:
        benign = mali
        for idx,_ in enumerate(y_pred):
            y_pred[idx] = (y_pred[idx]+1)%2

    return y_pred


def GMM(mask_list):
    pca = PCA(n_components=2)
    newX = pca.fit_transform(mask_list)
    gmm = mixture.GaussianMixture(n_components=2).fit(newX)
    y_pred = gmm.predict(newX)

    benign = 0
    mali = 0
    for f in y_pred:
        if f==1:
            benign+=1
        else:
            mali+=1

    if benign < mali:
        benign = mali
        for idx,_ in enumerate(y_pred):
            y_pred[idx] = (y_pred[idx]+1)%2
    return y_pred


def GMM2(mask_list):
    if not mask_list or len(mask_list) < 2:
        print("Warning: GMM requires at least 2 samples. Returning default labels.")
        return [0] * len(mask_list) # Mark all as outliers if not enough data

    # Ensure input is a numpy array for sklearn
    try:
        mask_array = np.array(mask_list)
        if mask_array.ndim == 1: # If PCA gets 1D data (e.g., only 1 sample after filtering)
             print("Warning: PCA/GMM input is 1D. Adjusting.")
             # Handle 1D case: maybe skip PCA or return default
             # For now, let's assume GMM can handle 1D directly if n_components=1 or handle error
             # Let's return default for simplicity in this edge case
             return [0] * len(mask_list)

        # Proceed with PCA only if input is suitable
        if mask_array.shape[1] < 2: # Check if feature dimension is less than n_components
            print("Warning: Feature dimension < 2. Skipping PCA.")
            newX = mask_array
            n_components_gmm = min(2, mask_array.shape[0]) # Adjust GMM components based on samples
        else:
            pca = PCA(n_components=2)
            newX = pca.fit_transform(mask_array)
            n_components_gmm = 2

        # Ensure n_components <= n_samples for GMM
        n_components_gmm = min(n_components_gmm, newX.shape[0])
        if n_components_gmm < 1:
             print("Warning: Not enough samples for GMM after PCA. Returning defaults.")
             return [0] * len(mask_list)

        gmm = mixture.GaussianMixture(n_components=n_components_gmm, random_state=0).fit(newX) # Added random_state
        y_pred = gmm.predict(newX)

        # --- Post-processing logic ---
        # If GMM used only 1 component, all predictions will be 0. Interpret appropriately.
        if n_components_gmm == 1:
            print("GMM ran with 1 component. Interpretation might need adjustment.")
            # Decide if 1 component means all inliers or requires different handling. Assume inliers for now.
            return [1] * len(mask_list)


        # Recalculate benign/mali based on actual predictions
        benign = np.sum(y_pred == 1)
        mali = np.sum(y_pred == 0)

        # Check if the majority class needs flipping - THIS LOGIC IS SUSPICIOUS
        # The original code flips if the *last* element is 0. This seems arbitrary.
        # A more common approach is to assume the larger cluster is benign (1).
        if mali > benign: # If cluster 0 is larger, flip labels so 1 represents the larger cluster
             print("Flipping GMM labels: Assuming larger cluster is benign (1).")
             y_pred = 1 - y_pred # Flip 0s to 1s and 1s to 0s

        return y_pred.tolist() # Return as list

    except Exception as e:
        print(f"Error during GMM/PCA: {e}")
        print(f"Input mask_list shapes: {[m.shape for m in mask_list]}")
        # Handle error: maybe return all outliers
        return [0] * len(mask_list)