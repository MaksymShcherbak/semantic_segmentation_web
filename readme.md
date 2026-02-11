# Semantic Segmentation 🧩

![example1](example1.png)

This application allows exploring various approaches to the design of semantic segmentation neural networks.

This was my **Master's Degree** project at university.

![example1](example2.png)

![example1](example3.png)

## What is Semantic Segmentation?

Semantic Segmentation is a **computer vision task** in which a model assigns a class label to **every individual pixel** of an image (for example: sky, road, person, or vehicle).  
Unlike traditional image classification, where a single label describes the whole image, semantic segmentation produces a **pixel-level understanding** of the scene.

This detailed spatial information makes the method especially important in areas such as **autonomous driving**, **robotics**, **medical imaging**, and other systems that must interpret complex environments with high precision.

## Tech Stack

- 🐍 [Python](https://www.python.org/) as the most popular Programming language in the ML field.
- ⚡ [Keras/TensorFlow](https://keras.io/) Framework for building and training deep learning models.
- 🥤 [Flask](https://flask.palletsprojects.com/en/stable/) Framework for creating a full-stack web application.
- Classic **HTML/CSS/JavaScript** combo for front-end development.

## Setting Up the Environment

If you want to build the project locally, you can use **Anaconda** to create the environment.

1. Install [Anaconda](https://www.anaconda.com/products/distribution).

2. Create the Conda environment using the provided `environment.yml` file:

```bat
conda env create -f environment.yml
```

3. Activate the environment:

```bat
conda activate tf-ss-env
```

_If you encounter any issues while running the project on Windows, your system Python user packages may be leaking into the Conda environment._

To prevent this, disable Python user site-packages:

```bat
setx PYTHONNOUSERSITE 1
```

## Training the Models

The models come pre-trained, so this step is completely optional.

If you want to train the models yourself, execute notebooks in the following order (using [Google Colab](https://colab.research.google.com/) is recommended):

1. `generate-weights.ipynb`
2. `semantic_segmentation.ipynb`

## Running the Application

```bat
cd web
python index.py
```

The project will be accessible at `http://localhost:3000`.
