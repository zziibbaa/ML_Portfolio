# Few-Shot Image Recognition with Siamese Network & ResNet50

A **metric-learning based image recognition project** built with TensorFlow/Keras, using a **ResNet50 ImageNet-pretrained backbone**, **Siamese Network**, **Contrastive Loss**, and **prototype-based Few-Shot classification** on the **Omniglot dataset**.

The project focuses on learning a transferable embedding space rather than training a conventional fixed-class classifier.

---

## 🎯 Objective

The goal is to learn an embedding function:

```text
Image → Encoder → Embedding
```

such that:

```text
Same Class       → Small Distance
Different Class  → Large Distance
```

The learned embedding space is then used for **N-way K-shot classification** on previously unseen classes.

---

## 🧠 Approach

The project consists of two stages:

### Stage 1 — Metric Learning

A Siamese Network learns meaningful image embeddings using Contrastive Loss.

```text
Image A ─────────┐
                 ├──► Shared ResNet50 Encoder ──► Embedding A
Image B ─────────┘                              └► Embedding B
                                                        │
                                                        ▼
                                                  Euclidean Distance
                                                        │
                                                        ▼
                                                 Contrastive Loss
```

The ResNet50 backbone is initialized with **ImageNet pretrained weights** and initially frozen.

The encoder architecture is:

```text
Input
  ↓
ResNet50
  ↓
Global Average Pooling
  ↓
2048-dim Feature Vector
  ↓
Dense(256)
  ↓
Batch Normalization
  ↓
ReLU
  ↓
L2 Normalization
  ↓
256-dim Embedding
```

---

### Stage 2 — Few-Shot Classification

After learning the embedding space, classification is performed using support examples from previously unseen classes.

Example:

```text
3-Way 3-Shot

Class A → 3 Support Images
Class B → 3 Support Images
Class C → 3 Support Images
```

Support images are embedded and averaged to create class prototypes:

```text
Support Embeddings
       ↓
Mean Embedding
       ↓
Class Prototype
```

A query image is then classified according to its distance from each prototype:

```text
Query Image
     ↓
Encoder
     ↓
Query Embedding
     ↓
 ┌───────────────┐
 │ Distance to A │
 │ Distance to B │
 │ Distance to C │
 └───────────────┘
     ↓
Minimum Distance
     ↓
Predicted Class
```

---

## 📊 Dataset

The project uses the **Omniglot handwritten character dataset**, which contains many classes of handwritten characters across different alphabets.

To evaluate generalization to unseen classes, the split is performed **at the class level**, rather than randomly splitting individual images.

```text
Omniglot
│
├── Training Classes
│      ↓
│   Siamese Training
│
└── Unseen Test Classes
       ↓
   Few-Shot Episodes
       ├── Support Set
       └── Query Set
```

This setup is important because the objective is to evaluate whether the learned embedding can generalize to classes that were not observed during encoder training.

---

## 🔬 Pair Generation

Training pairs are generated dynamically.

For each anchor image:

```text
Anchor
 ├── Same-Class Image      → Positive Pair → Label 1
 └── Different-Class Image → Negative Pair → Label 0
```

The pair pipeline is implemented using:

```python
tf.data.Dataset.from_generator()
```

and image loading/preprocessing is performed through the TensorFlow data pipeline.

---

## 📐 Contrastive Loss

The model uses Contrastive Loss with a margin:

```text
Positive Pair:
L = d²

Negative Pair:
L = max(margin - d, 0)²
```

where `d` represents the Euclidean distance between the two embeddings.

Implementation:

```python
def contrastive_loss(y_true, y_pred, margin=1.0):

    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.reshape(y_true, (-1, 1))

    positive_loss = y_true * tf.square(y_pred)

    negative_loss = (
        (1 - y_true)
        * tf.square(tf.maximum(margin - y_pred, 0))
    )

    return tf.reduce_mean(
        positive_loss + negative_loss
    )
```

---

## 📈 Current Results

The current implementation produced:

```text
Mean Positive Distance ≈ 0.46
Mean Negative Distance ≈ 0.66
```

This indicates that the learned embedding space is separating same-class and different-class pairs to some degree.

A pair-level evaluation on the current implementation achieved approximately:

```text
Accuracy ≈ 67%
```

The Few-Shot evaluation is performed using episodic N-way K-shot classification.

> Results are based on the current implementation and are not intended as a benchmark against state-of-the-art methods.

---

## ⚙️ Engineering Challenges

One of the main implementation challenges was GPU memory usage.

The project was developed with limited GPU VRAM, while ResNet50 is a relatively large backbone and the Siamese architecture processes two image branches.

To reduce memory pressure, the project uses:

```python
dataset.batch(batch_size)
dataset.prefetch(tf.data.AUTOTUNE)
```

and embeddings are generated in small batches rather than loading the complete dataset into GPU memory.

This also led to practical experience with TensorFlow's `tf.data` pipeline and debugging `ResourceExhaustedError` issues.

---

## 🛠️ Technologies

| Category                  | Technologies                   |
| ------------------------- | ------------------------------ |
| Language                  | Python                         |
| Deep Learning             | TensorFlow / Keras             |
| Backbone                  | ResNet50                       |
| Pretraining               | ImageNet                       |
| Learning Paradigm         | Metric Learning                |
| Architecture              | Siamese Network                |
| Loss                      | Contrastive Loss               |
| Few-Shot Method           | Prototype-based classification |
| Dataset                   | Omniglot                       |
| Data Pipeline             | `tf.data`                      |
| Numerical Computing       | NumPy                          |
| Preprocessing / Splitting | scikit-learn                   |

---

## 📚 Key Concepts

This project provided hands-on experience with:

* Transfer Learning
* ResNet50
* CNN feature extraction
* Siamese Networks
* Shared-weight architectures
* Representation Learning
* Metric Learning
* Image Embeddings
* Euclidean Distance
* Contrastive Loss
* Support / Query Sets
* N-way K-shot Learning
* Prototypes
* Episodic Evaluation
* Class-level train/test splitting
* TensorFlow `tf.data`
* GPU memory management

---

## 🚀 Future Improvements

Potential next steps include:

* Hard-negative mining
* Better pair sampling strategies
* Data augmentation
* Fine-tuning selected ResNet50 layers
* Cosine similarity comparison
* Triplet Loss
* Prototypical Networks
* Comparison between different metric-learning approaches
* Evaluation across a larger number of episodes
* Confidence intervals for Few-Shot performance
* Approximate Nearest Neighbor search for large-scale embedding retrieval

---

## 📁 Project Structure

```text
few-shot-siamese-resnet/
│
├── FSL.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 💡 Takeaway

This project moves beyond conventional image classification toward **representation-based recognition**.

Instead of learning:

```text
Image → Fixed Class
```

the model learns:

```text
Image → Embedding
```

and classification is performed through relationships between embeddings.

The combination of **Siamese Networks + Contrastive Loss + ResNet50 embeddings + Few-Shot prototype classification** provides a practical introduction to modern metric-learning based computer vision systems.
