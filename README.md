# Vision AI TensorFlow + React App

This project now includes:

- a TensorFlow training backend
- a FastAPI inference server
- a React camera frontend that draws the live stream onto a canvas and sends captured frames to the backend for classification
- an in-app sample collection flow that saves labeled camera frames into the training dataset

Training images should be placed in class-specific subfolders under `training/images`:

```text
training/
  images/
    cats/
      cat-001.jpg
      cat-002.jpg
    dogs/
      dog-001.jpg
      dog-002.jpg
```

Each subfolder name becomes a class label.

## Environment

TensorFlow is installed in the local virtual environment at `.venv`.

Install frontend packages with:

```powershell
npm install
```

## 1. Train a Model

Run training with:

```powershell
.\.venv\Scripts\python.exe -m backend --epochs 15
```

Optional flags:

```powershell
.\.venv\Scripts\python.exe -m backend --epochs 20 --batch-size 16 --image-size 224 --export-name pets-v1
```

The training script:

- reads images from `training/images`
- creates a training/validation split automatically
- trains a CNN classifier with data augmentation and dropout
- saves model artifacts into `models/<run-name>/`

Saved outputs include:

- `classifier.keras`
- `best.keras`
- `labels.json`
- `history.json`
- `metadata.json`

## 2. Start the Inference API

The React app uses the Python backend to classify captured frames.

Start the API with:

```powershell
.\.venv\Scripts\python.exe -m backend.api
```

By default the API runs at `http://127.0.0.1:8000`.

Available routes:

- `GET /api/health`
- `GET /api/models`
- `POST /api/classify`

The API automatically uses the newest model in `models/` unless a specific model name is requested.

## 3. Start the React Frontend

Run the frontend with:

```powershell
npm run dev
```

By default the frontend runs at `http://localhost:5173` and proxies `/api/*` requests to the backend on port `8000`.

## Frontend Behavior

The React app:

- requests camera access
- draws the live video feed onto a canvas
- has a collection mode for saving labeled images into `training/images/<label>/`
- lets you classify the current canvas frame
- can auto-classify frames on an interval
- shows dataset image and label counts while collecting
- replaces the left panel in collection mode with a gallery of saved training images
- lets you relabel or delete saved training images one by one
- enables training only when there are trainable images and the current dataset is newer than the latest trained model
- keeps a training log that records which images were used for each model run
- lets you delete a saved model without losing the historical training log
- shows the top prediction and confidence breakdown

If no trained model exists yet, the frontend will show a helpful message instead of trying to classify.

## Collection Workflow

1. Open the frontend.
2. Switch the camera panel into collection mode.
3. Enter a label such as `mug` or `apple`.
4. Click `Add labeled image` to save the current canvas frame into `training/images/<label>/`.
5. Review the image gallery on the left, fix labels, and delete bad captures as needed.
6. Repeat until you have enough examples for at least two labels.
7. Click `Train model` when the dataset is ready.

After training completes, the newest model in `models/` becomes available for classification and the training run is recorded in the UI log and in `models/training-log.json`.

Deleting a model removes its saved files from `models/<run-name>/`, but the training log remains as a record of what was trained. Once a matching model is deleted, the dataset becomes trainable again.
