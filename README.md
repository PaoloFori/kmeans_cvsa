## kmeans_cvsa

This directory contains the K-Means clustering node. This node is responsible for classifying real-time EEG power-band features using a pre-trained K-Means model.

---

### 1. Input

* **Topic:** `/cvsa/eeg_power`
* **Data:** The node subscribes to this topic, expecting the `msg.data` field to contain a flattened matrix of EEG signal power, structured as `[channels x bands]`.

---

### 2. Configuration

This node **requires a YAML configuration file** that defines the K-Means model and all necessary processing parameters.

The YAML file must contain the following fields:

* `band`: A list of strings specifying the frequency bands used to train the model (e.g., `['delta', 'theta', 'alpha']`). This is used to validate against the bands provided in the input message, ensuring the correct features are extracted.
* `indices`: A list of integers. These are the specific indices from the `[channels x bands]` input matrix that will be used to construct the feature vector.
* `mu`: A list of floats representing the mean values (one for each feature) used for standardization.
* `sigma`: A list of floats representing the standard deviation values (one for each feature) used for standardization.
* `n_clusters`: The number of clusters (K).
* `n_features`: The total number of features (must match the length of `indices`, `mu`, and `sigma`).
* `centroids`: A nested list `[n_clusters x n_features]` containing the coordinates of the pre-trained cluster centroids.

#### Example `kmeans_model.yaml`

```yaml
KmeansModelCfg:
  name: "kmeans_model"
  filenames: "c7.20250217.160026.calibration.cvsa_lbrb.gdf;
c7.20250217.161215.calibration.cvsa_lbrb.gdf;
c7.20250217.164432.calibration.cvsa_lbrb.gdf"
  params:
    K: 2 # number of clusters
    nfeatures: 3 # number of features
    occipital_left_idx: [13, 17, 29, 30, 33, 34, 37]
    occipital_left: ['P3', 'O1', 'P5', 'P1', 'PO5', 'PO3', 'PO7']
    occipital_right_idx: [15, 18, 31, 32, 35, 36, 38]
    occipital_right: ['P4', 'O2', 'P2', 'P6', 'PO4', 'PO6', 'PO8']
    frontal_idx: [3, 4, 5, 20, 21]
    frontal: ['F3', 'FZ', 'F4', 'F1', 'F2']
    central_left_idx: [6, 8, 11, 22, 25, 27]
    central_left: ['FC1', 'C3', 'CP1', 'FC3', 'C1', 'CP3']
    central_right_idx: [7, 10, 12, 24, 26, 28]
    central_right: ['FC2', 'C4', 'CP2', 'FC4', 'C2', 'CP4']
    excluded_idx: [1, 2, 19]
    excluded: ['FP1', 'FP2', 'EOG']
    mu: [0.19273941, 0.12819353, 22.92394539]
    sigma: [0.14682772, 0.06617118, 9.71295373]
    centroids: 
    - [1.01705337, 1.05682130, 0.41802147]
    - [-0.48712113, -0.50616811, -0.20021279]
    band: 
    - [8, 14]
```

---

### 3. Model Generation

The K-Means model (the `.yaml` file containing centroids, mu, sigma, etc.) is generated using a **MATLAB script** located in the `/create_kmeans` directory.

This script is responsible for:
1.  Training the K-Means model based on a training dataset.
2.  Exporting all required parameters (`centroids`, `mu`, `sigma`, `indices`, etc.) into the YAML file format required by this node.
3.  Generating and saving the datasets that are subsequently used to train the **QDA (Quadratic Discriminant Analysis)** classifier for the IC (Impaired Consciousness) state.

---

### 4. Workflow

1.  **Load Model:** The node loads the K-Means parameters (centroids, normalization values, indices) from the specified YAML file.
2.  **Receive Data:** It listens for incoming messages on `/cvsa/eeg_power`.
3.  **Extract Features:** Using the `indices` from the YAML, the node builds the feature vector from the `msg.data` array. (This step is performed by the `compute_sparsity` function or a similar utility).
4.  **Standardize:** The raw feature vector is normalized using the Z-score formula: $feature_{norm} = (feature_{raw} - \mu) / \sigma$.
5.  **Classify:** The node calculates the Euclidean distance from the standardized feature vector to each of the `centroids` and assigns the data to the nearest cluster.
6.  **Publish:** The node publishes the resulting classification probability (or cluster assignment) to the output topic.

---

### 5. Output

* **Topic:** `/cvsa/neuroprediction/kmeans`
* **Data:** Publishes the classification probability based on the K-Means model.

---

### 6. Testing

The validation tests for this node are located in the `test/node` directory. A **launch file** is provided in this same directory to run the test.

The testing process validates the node's functionality across both ROS and MATLAB environments to ensure identical outputs:

1.  **ROS Execution:** The test is initiated using the launch file. The node processes a standard input file, **`processed_data.csv`**. The resulting classification probabilities are captured and saved to **`classified.csv`**.
2.  **MATLAB Verification:** The ROS output file, **`classified.csv`**, is loaded into MATLAB.
3.  **Comparison:** The probabilities from **`classified.csv`** are compared against the results from the equivalent MATLAB implementation, which processes the *same* **`processed_data.csv`** input file.

The test passes if the output probabilities from both ROS and MATLAB are identical, confirming that the processing logic is correct. You need Yamlmatlab just for the test to load the yaml file.