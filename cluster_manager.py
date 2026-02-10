import numpy as np
import cv2

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


class ClusterManager:
    """
    People cluster detection using DBSCAN on detection centers.

    If sklearn not available -> fallback naive grouping.
    """

    def __init__(self, eps=60, min_samples=2):
        self.eps = eps
        self.min_samples = min_samples

    def _fallback_clusters(self, points):
        """
        Simple fallback: treat every point as its own cluster.
        """
        clusters = []
        for p in points:
            clusters.append([p])
        return clusters

    def compute_clusters(self, centers):
        """
        centers: list of tuples [(cx, cy), ...]
        returns clusters: list[list[(cx,cy)]]
        """
        if len(centers) == 0:
            return []

        if not SKLEARN_OK:
            return self._fallback_clusters(centers)

        X = np.array(centers, dtype=np.float32)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)

        labels = clustering.labels_
        clusters = {}

        for idx, label in enumerate(labels):
            if label == -1:
                # noise point -> cluster alone
                clusters[f"noise_{idx}"] = [centers[idx]]
            else:
                clusters.setdefault(label, []).append(centers[idx])

        return list(clusters.values())

    def draw_cluster_overlay(self, frame, clusters):
        """
        Draw circles + labels for clusters.
        """
        if frame is None:
            return frame

        overlay = frame.copy()

        for i, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue

            pts = np.array(cluster, dtype=np.int32)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))

            # radius proportional to spread
            if len(cluster) == 1:
                radius = 20
            else:
                dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
                radius = int(max(30, min(160, np.max(dists) + 25)))

            # draw cluster circle
            cv2.circle(overlay, (cx, cy), radius, (0, 255, 180), 3)

            # cluster label
            label = f"C{i+1}: {len(cluster)}"
            cv2.putText(
                overlay,
                label,
                (cx - 40, cy - radius - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 180),
                2,
                cv2.LINE_AA,
            )

        return overlay
