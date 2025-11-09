import numpy as np
PARAM_GRIDS = {
    "KMeans": {
        "n_clusters": [2, 3, 4],
        "init": ["k-means++", "random"],
        "n_init": [10, 20],
        "max_iter": [300, 500],
        "tol": [1e-4, 1e-3],
        "random_state": [42],
        "algorithm": ["lloyd", "elkan"]
    },
    "AgglomerativeClustering": {
        "n_clusters": [2, 3, 4],
        "linkage": ["ward", "complete", "average", "single"],
        "metric": ["euclidean", "l1", "l2", "manhattan", "cosine"],
        "compute_full_tree": ["auto", True, False],
        "distance_threshold": [None]
    },
    "DBSCAN": {
        "eps": [0.5, 1.0, 1.5, 2.0],
        "min_samples": [3, 5, 8, 10],
        "metric": ["euclidean", "manhattan", "cosine"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [20, 30, 40],
        "p": [1, 2]
    },
    "MeanShift": {
        "bandwidth": [None, 0.5, 1.0, 2.0],
        "seeds": [None],
        "bin_seeding": [True, False],
        "cluster_all": [True, False],
        "max_iter": [300, 500]
    },
    "GaussianMixture": {
        "n_components": [2, 3, 4],
        "covariance_type": ["full", "tied", "diag", "spherical"],
        "tol": [1e-3, 1e-4],
        "max_iter": [100, 200],
        "n_init": [1, 5, 10],
        "init_params": ["kmeans", "random"],
        "random_state": [42],
        "reg_covar": [1e-6, 1e-5]
    },
    "Birch": {
        "n_clusters": [None, 2, 3, 4],
        "threshold": [0.3, 0.5, 1.0, 1.5],
        "branching_factor": [25, 50, 100],
        "copy": [True, False],
        "compute_labels": [True, False]
    },
    "AffinityPropagation": {
        "damping": [0.5, 0.7, 0.9],
        "max_iter": [200, 500],
        "convergence_iter": [15, 30, 50],
        "preference": [-50, -10, None],
        "affinity": ["euclidean", "precomputed"],
        "verbose": [False]
    },
    "OPTICS": {
        "min_samples": [5, 10],
        "max_eps": [np.inf, 1.5, 2.0],
        "metric": ["euclidean", "manhattan", "cosine"],
        "p": [2],
        "xi": [0.05, 0.1, 0.2],
        "min_cluster_size": [0.05, 0.1],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [20, 30, 40]
    },
    "MiniBatchKMeans": {
        "n_clusters": [2, 3, 4],
        "init": ["k-means++", "random"],
        "n_init": [10, 20],
        "batch_size": [50, 100, 200],
        "max_iter": [100, 300],
        "tol": [1e-4, 1e-3],
        "max_no_improvement": [5, 10],
        "init_size": [None, 100, 300],
        "random_state": [42],
        "reassignment_ratio": [0.01, 0.1]
    }
}