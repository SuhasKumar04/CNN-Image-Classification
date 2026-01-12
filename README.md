# CNN Image Processing Demo (Convolution → Activation → Pooling)

This project is a simple, visual walkthrough of the **core building blocks of a Convolutional Neural Network (CNN)**.  
Instead of training a full CNN, the code applies a **fixed convolution filter** to an image, then runs a standard CNN-style pipeline:

1. **Convolution** (feature extraction)
2. **Activation** (non-linearity)
3. **Pooling** (downsampling)

The output at each stage is displayed so you can see what a CNN is doing internally.

---

## What is a CNN (High Level)?

A **Convolutional Neural Network (CNN)** is a deep learning model designed for images (and other grid-like data).  
CNNs work well because they:
- learn **local patterns** (like edges, textures, shapes),
- reuse the same filter across an image (parameter sharing),
- build **hierarchical features**: edges → corners → parts → objects.

A typical CNN looks like:
**Input Image → [Conv → ReLU → Pool] × N → Flatten → Dense → Prediction**

This repo focuses on the **Conv → Activation → Pool** portion.

---

## What this code does

### 1) Load + preprocess image
- Decode image into a tensor
- Convert to grayscale (1 channel)
- Resize to a fixed size (300×300)
- Convert to `float32` and normalize pixel values to `[0, 1]`
- Add a batch dimension so TensorFlow can run convolution

Key ideas:
- CNNs expect inputs shaped like **(batch, height, width, channels)**.

---

### 2) Define a convolution kernel (filter)

A **kernel** is a small matrix ("filter") that slides over the image and computes weighted sums.

This project uses a classic **3×3 edge detection kernel**:

```python
[[-1, -1, -1],
 [-1,  8, -1],
 [-1, -1, -1]]
