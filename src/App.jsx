function PlusIcon(props) {
  return (
    <IconBase {...props}>
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </IconBase>
  );
}
function MinusIcon(props) {
  return (
    <IconBase {...props}>
      <line x1="5" y1="12" x2="19" y2="12" />
    </IconBase>
  );
}
function LabelIcon(props) {
  return (
    <IconBase {...props}>
      <rect x="4" y="8" width="16" height="8" rx="2" />
      <circle cx="8" cy="12" r="1.5" />
      <path d="M16 12h2" />
    </IconBase>
  );
}
function CameraIcon(props) {
  return (
    <IconBase {...props}>
      <path d="M4 8h3l1.8-2h6.4L17 8h3a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2Z" />
      <circle cx="12" cy="13" r="4" />
    </IconBase>
  );
}

function ExpandIcon(props) {
  return (
    <IconBase {...props}>
      <path d="M15 4h5v5" />
      <path d="M14 10 20 4" />
      <path d="M9 20H4v-5" />
      <path d="M10 14 4 20" />
    </IconBase>
  );
}

import {
  startTransition,
  useEffect,
  useEffectEvent,
  useRef,
  useState,
} from "react";

import * as tf from "@tensorflow/tfjs";

// StatusDot component for inference mode indicator
function StatusDot({ color = "green" }) {
  return (
    <svg width="14" height="14" style={{ marginLeft: 4, verticalAlign: "middle" }}>
      <circle cx="7" cy="7" r="5" fill={color} stroke={color} />
    </svg>
  );
}

function ImagesIcon(props) {
  return (
    <IconBase {...props}>
      <rect x="3" y="7" width="18" height="13" rx="2" />
      <circle cx="8.5" cy="13" r="2.5" />
      <path d="M21 15l-5-5-4 4-2-2-4 4" />
    </IconBase>
  );
}

const CAPTURE_WIDTH = 960;
const CAPTURE_HEIGHT = 540;
const MIN_CAPTURE_COUNT = 1;
const MAX_CAPTURE_COUNT = 99;
const CAPTURE_INTERVAL_OPTIONS = Array.from({ length: 10 }, (_, index) => index + 1);

function IconBase({ children, className = "icon", ...props }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className={className}
      {...props}
    >
      {children}
    </svg>
  );
};

function CollapseIcon(props) {
  return (
    <IconBase {...props}>
      <path d="M9 4H4v5" />
      <path d="M10 10 4 4" />
      <path d="M15 20h5v-5" />
      <path d="m14 14 6 6" />
      <path d="M20 9V4h-5" />
      <path d="M14 10 20 4" />
      <path d="M4 15v5h5" />
      <path d="m4 20 6-6" />
    </IconBase>
  );
}

function RefreshIcon(props) {
  return (
    <IconBase {...props}>
      <path d="M20 11a8 8 0 0 0-14.9-4" />
      <path d="M4 4v4h4" />
      <path d="M4 13a8 8 0 0 0 14.9 4" />
      <path d="M20 20v-4h-4" />
    </IconBase>
  );
}

function EditIcon(props) {
  return (
    <IconBase {...props}>
      <path d="M12 20h9" />
      <path d="m16.5 3.5 4 4L8 20l-5 1 1-5 12.5-12.5Z" />
    </IconBase>
  );
}

function TrashIcon(props) {
  return (
    <IconBase {...props}>
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="m19 6-1 14H6L5 6" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </IconBase>
  );
}

function LockIcon(props) {
  return (
    <IconBase {...props}>
      <rect x="4" y="11" width="16" height="10" rx="2" />
      <path d="M8 11V8a4 4 0 1 1 8 0v3" />
    </IconBase>
  );
}

function ClockIcon(props) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </IconBase>
  );
}

function CalendarIcon(props) {
  return (
    <IconBase {...props}>
      <rect x="3" y="5" width="18" height="16" rx="2" />
      <path d="M16 3v4" />
      <path d="M8 3v4" />
      <path d="M3 10h18" />
    </IconBase>
  );
}

function ModelIcon(props) {
  return (
    <IconBase {...props}>
      <path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" />
      <path d="m12 12 8-4.5" />
      <path d="m12 12-8-4.5" />
      <path d="M12 12v9" />
    </IconBase>
  );
}

function TrainIcon(props) {
  return (
    <IconBase {...props}>
      <path d="M7 17h10" />
      <path d="M8 17V7a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v10" />
      <path d="M9 10h6" />
      <path d="m8 17-2 3" />
      <path d="m16 17 2 3" />
    </IconBase>
  );
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) {
    return "--";
  }

  const totalSeconds = Math.max(0, Number(seconds));
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(totalSeconds < 10 ? 1 : 0)}s`;
  }

  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = Math.round(totalSeconds % 60);
  if (minutes < 60) {
    return `${minutes}m ${remainingSeconds}s`;
  }

  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
}

function formatConfidence(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatTimestamp(value) {
  if (!value) {
    return "--";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleString();
}

function formatDateOnly(value) {
  if (!value) {
    return "--";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleDateString();
}

function formatTimeOnly(value) {
  if (!value) {
    return "--";
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return date.toLocaleTimeString();
}

function formatFileSize(bytes) {
  const numericBytes = Number(bytes);
  if (!Number.isFinite(numericBytes) || numericBytes < 0) {
    return "--";
  }

  if (numericBytes < 1024) {
    return `${numericBytes} B`;
  }

  const kilobytes = numericBytes / 1024;
  if (kilobytes < 1024) {
    return `${kilobytes.toFixed(kilobytes < 10 ? 1 : 0)} KB`;
  }

  const megabytes = kilobytes / 1024;
  return `${megabytes.toFixed(megabytes < 10 ? 1 : 0)} MB`;
}

function formatResolution(width, height, resolution) {
  if (resolution) {
    return resolution;
  }

  if (Number.isFinite(Number(width)) && Number.isFinite(Number(height))) {
    return `${width}x${height}`;
  }

  return "--";
}

function pluralize(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function clampCaptureCount(value) {
  const numericValue = Number.parseInt(value, 10);
  if (!Number.isFinite(numericValue)) {
    return MIN_CAPTURE_COUNT;
  }

  return Math.min(MAX_CAPTURE_COUNT, Math.max(MIN_CAPTURE_COUNT, numericValue));
}

function confidenceToneClass(confidence) {
  const numericConfidence = Number(confidence);
  if (!Number.isFinite(numericConfidence)) {
    return "";
  }

  if (numericConfidence >= 0.8) {
    return "overlay-confidence high";
  }

  if (numericConfidence >= 0.5) {
    return "overlay-confidence medium";
  }

  return "overlay-confidence low";
}

async function readErrorMessage(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    const payload = await response.json();
    return payload.detail || payload.message || "Request failed.";
  }

  const text = await response.text();
  return text || `Request failed with status ${response.status}.`;
}

async function captureCanvasBlob(canvas) {
  return new Promise((resolve) => {
    canvas.toBlob(resolve, "image/jpeg", 0.92);
  });
}

async function fetchModels() {
  const response = await fetch("/api/models");
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  return response.json();
}

async function fetchDatasetStatus() {
  const response = await fetch("/api/dataset");
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  return response.json();
}

async function classifyFrame(blob, modelName) {
  const query = modelName ? `?model=${encodeURIComponent(modelName)}` : "";
  const response = await fetch(`/api/classify${query}`, {
    method: "POST",
    headers: {
      "Content-Type": blob.type || "image/jpeg",
    },
    body: blob,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function saveTrainingFrame(blob, label) {
  const response = await fetch(`/api/capture?label=${encodeURIComponent(label)}`, {
    method: "POST",
    headers: {
      "Content-Type": blob.type || "image/jpeg",
    },
    body: blob,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function triggerTraining() {
  const response = await fetch("/api/train", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({}),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function deleteModelByName(modelName) {
  const response = await fetch(`/api/models/${encodeURIComponent(modelName)}`, {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function relabelDatasetImage(relativePath, label) {
  const response = await fetch("/api/dataset/images/label", {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      relative_path: relativePath,
      label,
    }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

async function deleteDatasetImage(relativePath) {
  const response = await fetch("/api/dataset/images", {
    method: "DELETE",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      relative_path: relativePath,
    }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}


export default function App() {
  // All hooks at the top
  const [tfjsStatus, setTfjsStatus] = useState("idle"); // idle | converting | ready | error
  const [tfjsModel, setTfjsModel] = useState(null);
  const [tfjsLabels, setTfjsLabels] = useState([]);
  const [inferenceMode, setInferenceMode] = useState("api"); // "api" or "tfjs"
  const [liveClassificationEnabled, setliveClassificationEnabled] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(0);
  const streamRef = useRef(null);
  const latestClassifyRef = useRef(0);
  const previousTrainingRef = useRef(false);
  const captureSequenceTimerRef = useRef(0);
  const captureSequenceResolverRef = useRef(null);
  const captureSequenceAbortRef = useRef(false);

  const [cameraState, setCameraState] = useState("idle");
  const [cameraError, setCameraError] = useState("");
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [modelsStatus, setModelsStatus] = useState("loading");
  const [modelsError, setModelsError] = useState("");
  const [datasetStatus, setDatasetStatus] = useState(null);
  const [datasetFetchState, setDatasetFetchState] = useState("loading");
  const [datasetError, setDatasetError] = useState("");
  const [classification, setClassification] = useState(null);
  const [predictionError, setPredictionError] = useState("");
  const [isClassifying, setIsClassifying] = useState(false);
  const [collectionMode, setCollectionMode] = useState(false);
  const [isGalleryExpanded, setIsGalleryExpanded] = useState(false);
  const [labelInput, setLabelInput] = useState("");
  const [captureCount, setCaptureCount] = useState(1);
  const [captureIntervalSeconds, setCaptureIntervalSeconds] = useState(2);
  const [captureSequence, setCaptureSequence] = useState({
    isRunning: false,
    total: 0,
    taken: 0,
    countdown: 0,
  });
  const [isSavingSample, setIsSavingSample] = useState(false);
  const [captureMessage, setCaptureMessage] = useState("");
  const [trainingMessage, setTrainingMessage] = useState("");
  const [trainingError, setTrainingError] = useState("");
  const [modelMessage, setModelMessage] = useState("");
  const [isDeletingModel, setIsDeletingModel] = useState(false);
  const [pendingModelDelete, setPendingModelDelete] = useState("");
  const [imageActionPath, setImageActionPath] = useState("");
  const [imageActionMessage, setImageActionMessage] = useState("");
  const [activeImagePath, setActiveImagePath] = useState("");
  const [modalLabelDraft, setModalLabelDraft] = useState("");

  // --- All functions and effects below this line ---

  // Trigger TF.js conversion for selected model
  async function triggerTfjsConversion(modelName) {
    setTfjsStatus("converting");
    setInferenceMode("api");
    setliveClassificationEnabled(false);
    setTfjsModel(null);
    setTfjsLabels([]);

    const response = await fetch(
      `/api/convert-tfjs?model_name=${encodeURIComponent(modelName)}`,
      { method: "POST" }
    );
    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const payload = await response.json();
    if (!payload?.job_id) {
      throw new Error("TFJS conversion job id was missing from backend response.");
    }

    await pollTfjsStatus(modelName, payload.job_id);
  }

  // Poll TF.js conversion status
  async function pollTfjsStatus(modelName, expectedJobId) {
    let done = false;
    setTfjsStatus("converting");
    while (!done) {
      const res = await fetch("/api/convert-tfjs/status");
      if (!res.ok) {
        throw new Error(await readErrorMessage(res));
      }
      const status = await res.json();

      if (status.job_id !== expectedJobId) {
        await new Promise((resolve) => setTimeout(resolve, 600));
        continue;
      }

      if (status.in_progress) {
        await new Promise((resolve) => setTimeout(resolve, 1200));
        continue;
      }

      if (status.last_error) {
        setTfjsStatus("error");
        setInferenceMode("api");
        setliveClassificationEnabled(false);
        done = true;
      } else if (status.last_result && status.last_result.returncode === 0) {
        try {
          const model = await tf.loadLayersModel(
            `/models/${encodeURIComponent(modelName)}/model.json`
          );
          let labels = [];
          const labelsResponse = await fetch(
            `/models/${encodeURIComponent(modelName)}/labels.json`
          );
          if (labelsResponse.ok) {
            const labelsPayload = await labelsResponse.json();
            if (Array.isArray(labelsPayload)) {
              labels = labelsPayload.map((label) => String(label));
            }
          }

          setTfjsModel(model);
          setTfjsLabels(labels);
          setTfjsStatus("ready");
          setInferenceMode("tfjs");
        } catch (e) {
          setPredictionError(e?.message || "Could not load converted TFJS model.");
          setTfjsStatus("error");
          setInferenceMode("api");
          setliveClassificationEnabled(false);
        }
        done = true;
      } else {
        setTfjsStatus("idle");
        setInferenceMode("api");
        setliveClassificationEnabled(false);
        done = true;
      }
    }
  }

  // When selectedModel changes, trigger TF.js conversion
  useEffect(() => {
    if (selectedModel) {
      void triggerTfjsConversion(selectedModel).catch((error) => {
        setTfjsStatus("error");
        setInferenceMode("api");
        setliveClassificationEnabled(false);
        setPredictionError(error.message || "TFJS conversion failed.");
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);

  const drawVideoFrame = useEffectEvent(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(drawVideoFrame);
      return;
    }

    const context = canvas.getContext("2d");
    const sourceWidth = video.videoWidth || CAPTURE_WIDTH;
    const sourceHeight = video.videoHeight || CAPTURE_HEIGHT;
    const scale = Math.max(canvas.width / sourceWidth, canvas.height / sourceHeight);
    const drawWidth = sourceWidth * scale;
    const drawHeight = sourceHeight * scale;
    const offsetX = (canvas.width - drawWidth) / 2;
    const offsetY = (canvas.height - drawHeight) / 2;

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);

    animationFrameRef.current = requestAnimationFrame(drawVideoFrame);
  });

  const refreshModels = useEffectEvent(async () => {
    setModelsStatus("loading");
    setModelsError("");

    try {
      const payload = await fetchModels();
      startTransition(() => {
        setModels(payload.models);
        setSelectedModel((current) => {
          if (current && payload.models.some((item) => item.name === current)) {
            return current;
          }
          return payload.models[0]?.name ?? "";
        });
        setModelsStatus("ready");
      });
    } catch (error) {
      setModelsError(error.message);
      setModelsStatus("error");
    }
  });

  const refreshDataset = useEffectEvent(async () => {
    setDatasetFetchState("loading");
    setDatasetError("");

    try {
      const payload = await fetchDatasetStatus();
      startTransition(() => {
        setDatasetStatus(payload);
        setDatasetFetchState("ready");
      });
    } catch (error) {
      setDatasetError(error.message);
      setDatasetFetchState("error");
    }
  });

  const captureAndClassify = useEffectEvent(async () => {
    const canvas = canvasRef.current;
    if (!canvas || isClassifying || models.length === 0) {
      return;
    }

    const operationId = Date.now();
    latestClassifyRef.current = operationId;
    setIsClassifying(true);
    setPredictionError("");

    try {
      let payload = null;
      if (inferenceMode === "tfjs" && tfjsModel) {
        const imageSize = Number(selectedModelMeta?.image_size) || 180;
        const pixels = tf.browser.fromPixels(canvas);
        const resized = tf.image.resizeBilinear(pixels, [imageSize, imageSize]);
        const batch = resized.expandDims(0).toFloat();
        const output = tfjsModel.predict(batch);
        const outputTensor = Array.isArray(output) ? output[0] : output;
        const values = await outputTensor.data();
        const probabilities = Array.from(values);

        pixels.dispose();
        resized.dispose();
        batch.dispose();
        if (Array.isArray(output)) {
          output.forEach((tensor) => tensor.dispose());
        } else {
          outputTensor.dispose();
        }

        const rankedPredictions = probabilities
          .map((confidence, index) => ({
            label: tfjsLabels[index] || `class_${index}`,
            confidence,
          }))
          .sort((left, right) => right.confidence - left.confidence);

        payload = {
          top_prediction: rankedPredictions[0] || null,
          predictions: rankedPredictions.slice(0, 3),
          model: { name: selectedModel },
          image_size: imageSize,
        };
      } else {
        // API inference
        const blob = await captureCanvasBlob(canvas);
        if (!blob) {
          setIsClassifying(false);
          setPredictionError("Could not capture the current frame.");
          return;
        }
        payload = await classifyFrame(blob, selectedModel || undefined);
      }
      if (latestClassifyRef.current !== operationId) {
        return;
      }
      startTransition(() => {
        setClassification(payload);
      });
    } catch (error) {
      setPredictionError(error.message);
    } finally {
      if (latestClassifyRef.current === operationId) {
        setIsClassifying(false);
      }
    }
  });

  const captureAndSaveSample = useEffectEvent(async () => {
    const canvas = canvasRef.current;
    const captureLabel = labelInput.trim();
    const totalShots = clampCaptureCount(captureCount);
    const delaySeconds = Math.min(
      CAPTURE_INTERVAL_OPTIONS[CAPTURE_INTERVAL_OPTIONS.length - 1],
      Math.max(CAPTURE_INTERVAL_OPTIONS[0], captureIntervalSeconds)
    );

    if (!canvas || isSavingSample) {
      return;
    }

    if (!captureLabel) {
      setCaptureMessage("");
      setTrainingError("");
      setDatasetError("Enter a label before saving a training image.");
      return;
    }

    captureSequenceAbortRef.current = false;
    setIsSavingSample(true);
    setCaptureSequence({
      isRunning: true,
      total: totalShots,
      taken: 0,
      countdown: 0,
    });
    setCaptureMessage("");
    setTrainingError("");
    setDatasetError("");

    let savedCount = 0;
    let lastSaved = null;

    try {
      for (let shotIndex = 0; shotIndex < totalShots; shotIndex += 1) {
        if (captureSequenceAbortRef.current) {
          break;
        }

        const blob = await captureCanvasBlob(canvas);
        if (!blob) {
          throw new Error("Could not capture the current frame.");
        }

        const payload = await saveTrainingFrame(blob, captureLabel);
        lastSaved = payload.saved;
        savedCount += 1;

        startTransition(() => {
          setDatasetStatus(payload.status);
        });
        setLabelInput(payload.saved.label);
        setCaptureSequence({
          isRunning: true,
          total: totalShots,
          taken: savedCount,
          countdown: 0,
        });

        if (savedCount < totalShots) {
          for (let remaining = delaySeconds; remaining > 0; remaining -= 1) {
            if (captureSequenceAbortRef.current) {
              break;
            }

            setCaptureSequence({
              isRunning: true,
              total: totalShots,
              taken: savedCount,
              countdown: remaining,
            });

            await new Promise((resolve) => {
              captureSequenceResolverRef.current = resolve;
              captureSequenceTimerRef.current = window.setTimeout(() => {
                captureSequenceTimerRef.current = 0;
                captureSequenceResolverRef.current = null;
                resolve();
              }, 1000);
            });
          }
        }
      }

      if (captureSequenceAbortRef.current) {
        if (savedCount > 0) {
          setCaptureMessage(
            `Saved ${savedCount} of ${totalShots} labeled ${savedCount === 1 ? "image" : "images"} before capture stopped.`
          );
        }
        return;
      }

      if (savedCount === 1 && lastSaved) {
        setCaptureMessage(
          `Saved ${lastSaved.relative_path} (${formatResolution(lastSaved.width, lastSaved.height, lastSaved.resolution)}, ${formatFileSize(lastSaved.file_size_bytes)})`
        );
      } else if (savedCount > 1 && lastSaved) {
        setCaptureMessage(
          `Saved ${savedCount} labeled images to ${lastSaved.label} at ${delaySeconds}s intervals.`
        );
      }
    } catch (error) {
      if (savedCount > 0) {
        setCaptureMessage(
          `Saved ${savedCount} of ${totalShots} labeled ${savedCount === 1 ? "image" : "images"} before an error interrupted capture.`
        );
      }
      setDatasetError(error.message);
    } finally {
      if (captureSequenceTimerRef.current) {
        window.clearTimeout(captureSequenceTimerRef.current);
        captureSequenceTimerRef.current = 0;
      }
      if (captureSequenceResolverRef.current) {
        const resolvePendingCountdown = captureSequenceResolverRef.current;
        captureSequenceResolverRef.current = null;
        resolvePendingCountdown();
      }
      captureSequenceAbortRef.current = false;
      setCaptureSequence({
        isRunning: false,
        total: 0,
        taken: 0,
        countdown: 0,
      });
      setIsSavingSample(false);
    }
  });

  const startTrainingRun = useEffectEvent(async () => {
    setTrainingError("");
    setTrainingMessage("");
    setCaptureMessage("");
    setModelMessage("");
    setImageActionMessage("");

    try {
      const payload = await triggerTraining();
      setTrainingMessage(payload.message);
      await refreshDataset();
    } catch (error) {
      setTrainingError(error.message);
    }
  });

  const handleDeleteModel = useEffectEvent(async (modelToDelete) => {
    if (!modelToDelete || isDeletingModel) {
      return;
    }

    setIsDeletingModel(true);
    setPendingModelDelete("");
    setModelsError("");
    setModelMessage("");
    setTrainingError("");
    setImageActionMessage("");

    try {
      const payload = await deleteModelByName(modelToDelete);
      startTransition(() => {
        setModels(payload.models);
        setDatasetStatus(payload.status);
        setSelectedModel((current) => {
          if (current && current !== modelToDelete && payload.models.some((item) => item.name === current)) {
            return current;
          }
          return payload.models[0]?.name ?? "";
        });
        if (classification?.model?.name === modelToDelete) {
          setClassification(null);
        }
      });
      setModelMessage(`Deleted model ${payload.deleted.deleted_model}`);
    } catch (error) {
      setModelsError(error.message);
    } finally {
      setIsDeletingModel(false);
    }
  });

  const openDeleteModelModal = useEffectEvent((modelName) => {
    const targetModel = modelName || selectedModel;
    if (!targetModel || isDeletingModel) {
      return;
    }
    setPendingModelDelete(targetModel);
  });

  const closeDeleteModelModal = useEffectEvent(() => {
    if (isDeletingModel) {
      return;
    }
    setPendingModelDelete("");
  });

  const handleOpenImageModal = useEffectEvent((item) => {
    setActiveImagePath(item.relativePath);
    setModalLabelDraft(item.label);
    setDatasetError("");
    setImageActionMessage("");
  });

  const handleCloseImageModal = useEffectEvent(() => {
    if (imageActionPath) {
      return;
    }
    setActiveImagePath("");
    setModalLabelDraft("");
  });

  const handleRelabelImage = useEffectEvent(async (relativePath, nextLabel) => {
    const trimmedLabel = nextLabel.trim();
    if (!trimmedLabel) {
      setDatasetError("Enter a label before saving the change.");
      setImageActionMessage("");
      return;
    }

    setImageActionPath(relativePath);
    setDatasetError("");
    setImageActionMessage("");

    try {
      const payload = await relabelDatasetImage(relativePath, trimmedLabel);
      startTransition(() => {
        setDatasetStatus(payload.status);
      });
      setActiveImagePath(payload.updated.new_relative_path);
      setModalLabelDraft(payload.updated.new_label);
      setImageActionMessage(
        `Moved ${payload.updated.old_relative_path} to ${payload.updated.new_relative_path}`
      );
    } catch (error) {
      setDatasetError(error.message);
    } finally {
      setImageActionPath("");
    }
  });

  const handleDeleteImage = useEffectEvent(async (relativePath) => {
    const confirmed = window.confirm(`Delete ${relativePath}?`);
    if (!confirmed) {
      return;
    }

    setImageActionPath(relativePath);
    setDatasetError("");
    setImageActionMessage("");

    try {
      const payload = await deleteDatasetImage(relativePath);
      startTransition(() => {
        setDatasetStatus(payload.status);
      });
      if (activeImagePath === relativePath) {
        setActiveImagePath("");
        setModalLabelDraft("");
      }
      setImageActionMessage(`Deleted ${payload.deleted.relative_path}`);
    } catch (error) {
      setDatasetError(error.message);
    } finally {
      setImageActionPath("");
    }
  });

  useEffect(() => {
    void refreshModels();
    void refreshDataset();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = CAPTURE_WIDTH;
      canvas.height = CAPTURE_HEIGHT;
    }
  }, []);

  useEffect(() => {
    let isMounted = true;

    async function startCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraState("error");
        setCameraError("This browser does not support camera access.");
        return;
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            facingMode: "environment",
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });

        if (!isMounted) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        streamRef.current = stream;
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          await video.play();
        }

        setCameraState("ready");
        setCameraError("");
        animationFrameRef.current = requestAnimationFrame(drawVideoFrame);
      } catch (error) {
        setCameraState("error");
        setCameraError(error.message || "Camera permission was denied.");
      }
    }

    setCameraState("loading");
    startCamera();

    return () => {
      isMounted = false;
      cancelAnimationFrame(animationFrameRef.current);
      captureSequenceAbortRef.current = true;
      if (captureSequenceTimerRef.current) {
        window.clearTimeout(captureSequenceTimerRef.current);
        captureSequenceTimerRef.current = 0;
      }
      if (captureSequenceResolverRef.current) {
        const resolvePendingCountdown = captureSequenceResolverRef.current;
        captureSequenceResolverRef.current = null;
        resolvePendingCountdown();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const trainingRuntime = datasetStatus?.training?.runtime ?? {};
  const isTraining = Boolean(trainingRuntime.in_progress);

  useEffect(() => {
    if (collectionMode) {
      return;
    }

    setIsGalleryExpanded(false);
    captureSequenceAbortRef.current = true;
    if (captureSequenceTimerRef.current) {
      window.clearTimeout(captureSequenceTimerRef.current);
      captureSequenceTimerRef.current = 0;
    }
    if (captureSequenceResolverRef.current) {
      const resolvePendingCountdown = captureSequenceResolverRef.current;
      captureSequenceResolverRef.current = null;
      resolvePendingCountdown();
    }
    setCaptureSequence({
      isRunning: false,
      total: 0,
      taken: 0,
      countdown: 0,
    });
    setIsSavingSample(false);
  }, [collectionMode]);

  useEffect(() => {
    if (
      !liveClassificationEnabled
      || inferenceMode !== "tfjs"
      || !tfjsModel
      || cameraState !== "ready"
      || models.length === 0
      || collectionMode
    ) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void captureAndClassify();
    }, 1800);

    return () => window.clearInterval(intervalId);
  }, [
    liveClassificationEnabled,
    inferenceMode,
    tfjsModel,
    cameraState,
    models.length,
    collectionMode,
  ]);

  useEffect(() => {
    if (!isTraining) {
      return;
    }

    const intervalId = window.setInterval(() => {
      void refreshDataset();
    }, 2500);

    return () => window.clearInterval(intervalId);
  }, [isTraining]);

  useEffect(() => {
    const wasTraining = previousTrainingRef.current;
    if (wasTraining && !isTraining) {
      if (trainingRuntime.last_error) {
        setTrainingError(trainingRuntime.last_error);
      } else if (trainingRuntime.last_result?.model_name) {
        setTrainingMessage(`Training complete: ${trainingRuntime.last_result.model_name}`);
        void refreshModels();
      }
    }
    previousTrainingRef.current = isTraining;
  }, [isTraining, trainingRuntime.last_error, trainingRuntime.last_result?.model_name]);

  const activePrediction = classification?.top_prediction ?? null;
  const selectedModelMeta =
    models.find((model) => model.name === selectedModel) ?? classification?.model ?? null;
  const backendStatus =
    modelsStatus === "error" || datasetFetchState === "error"
      ? "offline"
      : modelsStatus === "loading" || datasetFetchState === "loading"
        ? "loading"
        : "connected";

  const datasetSummary = datasetStatus?.dataset;
  const trainingSummary = datasetStatus?.training;
  const trainingLogs = datasetStatus?.logs ?? [];
  const availableModelNames = new Set(models.map((model) => model.name));
  const canTrain = Boolean(trainingSummary?.can_train);
  const trainingReason = trainingSummary?.reason || "";
  const trimmedLabelInput = labelInput.trim();
  const labelValidationMessage = trimmedLabelInput
    ? ""
    : "Enter a label to enable camera capture.";
  const captureSequenceRemaining = Math.max(captureSequence.total - captureSequence.taken, 0);
  const captureButtonLabel = isSavingSample
    ? `Capturing ${captureSequence.taken}/${captureSequence.total}`
    : captureCount === 1
      ? "Capture labeled image"
      : `Capture ${captureCount} labeled images`;
  const datasetCountsLabel = datasetSummary
    ? `${pluralize(datasetSummary.total_images, "image")} across ${pluralize(datasetSummary.label_count, "label")}`
    : "No dataset loaded yet";
  const datasetGalleryItems =
    datasetSummary?.labels?.flatMap((label) =>
      (label.image_details?.length
        ? label.image_details
        : label.images.map((relativePath) => ({
            label: label.name,
            relative_path: relativePath,
          }))
      ).map((item) => {
        const relativePath = item.relative_path ?? item.relativePath;
        return {
          relativePath,
          label: item.label ?? label.name,
          capturedAt: item.captured_at ?? item.capturedAt ?? "",
          width: item.width ?? null,
          height: item.height ?? null,
          resolution: item.resolution ?? "",
          fileSizeBytes: item.file_size_bytes ?? item.fileSizeBytes ?? null,
          previewUrl: `/api/training-image?path=${encodeURIComponent(relativePath)}&v=${encodeURIComponent(datasetSummary.fingerprint || "")}`,
        };
      })
    ) ?? [];
  const [galleryLabelFilter, setGalleryLabelFilter] = useState([]);
  const activeGalleryItem =
    datasetGalleryItems.find((item) => item.relativePath === activeImagePath) ?? null;
  const trainButtonIcon = isTraining ? <ClockIcon /> : canTrain ? <TrainIcon /> : <LockIcon />;
  const isGalleryFullscreen = collectionMode && isGalleryExpanded;
  const overlayRightLabel = collectionMode
    ? isSavingSample
      ? `${captureSequence.taken}/${captureSequence.total} saved${captureSequence.countdown > 0 ? ` | next in ${captureSequence.countdown}s` : ""}`
      : trimmedLabelInput
        ? `Label: ${trimmedLabelInput}`
        : "Enter a label"
    : selectedModel || "No model loaded";
  const overlayConfidenceClass = confidenceToneClass(activePrediction?.confidence);

  useEffect(() => {
    if (!activeImagePath) {
      return;
    }

    const selectedItem = datasetGalleryItems.find((item) => item.relativePath === activeImagePath);
    if (!selectedItem) {
      setActiveImagePath("");
      setModalLabelDraft("");
      return;
    }

    setModalLabelDraft(selectedItem.label);
  }, [activeImagePath, datasetSummary?.fingerprint]);

  return (
    <main className="app-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <section className={isGalleryFullscreen ? "hero gallery-expanded" : "hero"}>
        <div
          className={
            collectionMode
              ? isGalleryFullscreen
                ? "hero-copy collection-browser expanded"
                : "hero-copy collection-browser"
              : "hero-copy"
          }
        >
       
          {collectionMode ? (
            <>
              <div className="collection-browser-header">
                <div>
                  <p className="eyebrow">Training Images</p>
                  <h2>Dataset gallery</h2>
                </div>
                <div className="collection-browser-actions">
                  <span className="gallery-count"><ImagesIcon /> {datasetCountsLabel}</span>
                  <button
                    className="ghost-button icon-button"
                    onClick={() => setIsGalleryExpanded((value) => !value)}
                    type="button"
                    aria-label={isGalleryFullscreen ? "Collapse dataset gallery" : "Expand dataset gallery"}
                    title={isGalleryFullscreen ? "Collapse dataset gallery" : "Expand dataset gallery"}
                  >
                    {isGalleryFullscreen ? <CollapseIcon /> : <ExpandIcon />}
                  </button>
                </div>
              </div>

              <p className="lede">
                Review the saved samples, correct labels, and remove bad captures one by one.
              </p>

              <div className="status-row status-row-compact">
                <div className="status-pill">
                  <span className="status-label pill-heading"><ImagesIcon /> Images</span>
                  <strong>{datasetSummary?.total_images ?? 0}</strong>
                </div>
                <div className="status-pill">
                  <span className="status-label pill-heading"><LabelIcon /> Labels</span>
                  <strong>{datasetSummary?.label_count ?? 0}</strong>
                </div>
                <div className="status-pill">
                  <span className="status-label pill-heading">
                    {canTrain ? <TrainIcon /> : <LockIcon />} Train
                  </span>
                  <strong>{canTrain ? "ready" : "blocked"}</strong>
                </div>
              </div>

              {datasetGalleryItems.length ? (
                <div className="gallery-grid">
                  {datasetGalleryItems.map((item) => {
                    const isWorking = imageActionPath === item.relativePath;
                    return (
                      <article className={isGalleryFullscreen ? "sample-card expanded" : "sample-card"} key={item.relativePath}>
                        <div className={isGalleryFullscreen ? "sample-preview-tile expanded" : "sample-preview-tile"}>
                          <div
                            className="sample-preview-button"
                            onClick={() => handleOpenImageModal(item)}
                          >
                            <img
                              className="sample-preview"
                              src={item.previewUrl}
                              alt={item.label}
                              loading="lazy"
                            />
                            {isGalleryFullscreen && (
                              <div className="sample-preview-overlay">
                                <p className="sample-label sample-label-row">
                                  <LabelIcon />
                                  <span>{item.label}</span>
                                </p>
                                <p className="sample-path">{item.relativePath}</p>
                                <div className="sample-actions">
                                  <button
                                    className="ghost-button compact-button"
                                    onClick={(e) => { e.stopPropagation(); handleOpenImageModal(item); }}
                                    type="button"
                                  >
                                    <span className="button-content"><EditIcon /> Open editor</span>
                                  </button>
                                  <button
                                    className="danger-button compact-button"
                                    onClick={(e) => { e.stopPropagation(); handleDeleteImage(item.relativePath); }}
                                    type="button"
                                    disabled={isWorking}
                                  >
                                    <span className="button-content"><TrashIcon /> Delete</span>
                                  </button>
                                </div>
                              </div>
                            )}
                          </div>
                          {!isGalleryFullscreen && (
                            <div className="sample-card-body">
                              <p className="sample-label sample-label-row">
                                <LabelIcon />
                                <span>{item.label}</span>
                              </p>
                            </div>
                          )}
                        </div>
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="empty-gallery">
                  <h2>No saved samples yet</h2>
                  <p className="muted">
                    Use the camera panel on the right to capture labeled images into the training
                    dataset.
                  </p>
                </div>
              )}
            </>
          ) : (
            <>
              <p className="eyebrow">Vision AI Studio</p>
              <h1>Collect labeled images, train locally, then classify live camera frames.</h1>
              <p className="lede">
                Switch the camera panel into collection mode to save labeled samples directly into
                the training dataset. When the image set changes, the app will let you train a new
                model and keep a record of exactly what was used.
              </p>

              <div className="status-row">
                <div className="status-pill">
                  <span className="status-label pill-heading"><CameraIcon /> Camera</span>
                  <strong>{cameraState}</strong>
                </div>
                <div className="status-pill">
                  <span className="status-label pill-heading"><ModelIcon /> Models</span>
                  <strong>{models.length}</strong>
                </div>
                <div className="status-pill">
                  <span className="status-label pill-heading"><ImagesIcon /> Dataset</span>
                  <strong>{datasetSummary?.total_images ?? 0}</strong>
                </div>
                <div className="status-pill">
                  <span className="status-label pill-heading"><RefreshIcon /> Backend</span>
                  <strong>{backendStatus}</strong>
                </div>
              </div>
            </>
          )}
        </div>

        {!isGalleryFullscreen ? (
          <div className="hero-panel">
            <div className="camera-card">
              <div className="camera-header">
                <div>
                  <p className="panel-label">Live Canvas</p>
                  <h2>Camera feed</h2>
                </div>
                <div className="header-actions">
                  <button className="ghost-button" onClick={refreshDataset} type="button">
                    <span className="button-content"><RefreshIcon /> Refresh dataset</span>
                  </button>
                  <button className="ghost-button" onClick={refreshModels} type="button">
                    <span className="button-content"><RefreshIcon /> Refresh models</span>
                  </button>
                </div>
              </div>

              <div className="canvas-frame">
                <video ref={videoRef} className="hidden-video" playsInline muted />
                <canvas ref={canvasRef} className="camera-canvas" />
                <div className="camera-overlay">
                  {collectionMode ? (
                    <>
                      <span>Collection mode</span>
                      <span>{overlayRightLabel}</span>
                    </>
                  ) : liveClassificationEnabled && activePrediction ? (
                    <>
                      <span className="overlay-prediction">
                        <strong>{activePrediction.label}</strong>
                        <span className={overlayConfidenceClass}>
                          {formatConfidence(activePrediction.confidence)}
                        </span>
                      </span>
                      <span>{overlayRightLabel}</span>
                    </>
                  ) : (
                    <>
                      <span>Classification mode</span>
                      <span>{overlayRightLabel}</span>
                    </>
                  )}
                </div>
              </div>

              <div className="mode-toggle">
                <span>Mode</span>
                <label className="switch">
                  <input
                    type="checkbox"
                    checked={collectionMode}
                    onChange={(event) => {
                      const enabled = event.target.checked;
                      setCollectionMode(enabled);
                      if (enabled) {
                        setliveClassificationEnabled(false);
                        setPredictionError("");
                      }
                    }}
                  />
                  <span className="switch-track">
                    <span className="switch-thumb" />
                  </span>
                </label>
                <strong>{collectionMode ? "Collect samples" : "Classify frame"}</strong>
              </div>

              <div className="controls">
                {collectionMode ? (
                  <div className="collection-panel">
                    <label className="field">
                      <span className="field-title"><LabelIcon /> Label</span>
                      <input
                        className={labelValidationMessage ? "text-input invalid" : "text-input"}
                        type="text"
                        value={labelInput}
                        onChange={(event) => setLabelInput(event.target.value)}
                        placeholder="ex: mug, screwdriver, apple"
                      />
                    </label>
                    {labelValidationMessage ? (
                      <p className="message error compact-line validation-row">
                        <LabelIcon />
                        <span>{labelValidationMessage}</span>
                      </p>
                    ) : null}

                  <div className="capture-config-grid">
                    <label className="field sample-field">
                      <span className="field-title"><CameraIcon /> Photos to capture</span>
                      <div className="stepper">
                        <button
                          className="ghost-button stepper-button"
                          onClick={() => setCaptureCount((current) => clampCaptureCount(current - 1))}
                          type="button"
                          disabled={isSavingSample || captureCount <= MIN_CAPTURE_COUNT}
                        >
                          <span className="button-content"><MinusIcon /></span>
                        </button>
                        <input
                          className="stepper-input"
                          type="number"
                          min={MIN_CAPTURE_COUNT}
                          max={MAX_CAPTURE_COUNT}
                          value={captureCount}
                          onChange={(event) => setCaptureCount(clampCaptureCount(event.target.value))}
                          disabled={isSavingSample}
                        />
                        <button
                          className="ghost-button stepper-button"
                          onClick={() => setCaptureCount((current) => clampCaptureCount(current + 1))}
                          type="button"
                          disabled={isSavingSample || captureCount >= MAX_CAPTURE_COUNT}
                        >
                          <span className="button-content"><PlusIcon /></span>
                        </button>
                      </div>
                    </label>

                    <div className="field sample-field">
                      <span className="field-title"><ClockIcon /> Time between photos</span>
                      <div className="interval-options">
                        {CAPTURE_INTERVAL_OPTIONS.map((seconds) => (
                          <button
                            key={seconds}
                            className={
                              seconds === captureIntervalSeconds
                                ? "secondary-button compact-button active interval-button"
                                : "ghost-button compact-button interval-button"
                            }
                            onClick={() => setCaptureIntervalSeconds(seconds)}
                            type="button"
                            disabled={isSavingSample}
                          >
                            <span className="button-content">{seconds}s</span>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>

                  <div className="dataset-metrics">
                    <div className="metric-card">
                      <span className="status-label metric-label"><ImagesIcon /> Images</span>
                      <strong>{datasetSummary?.total_images ?? 0}</strong>
                    </div>
                    <div className="metric-card">
                      <span className="status-label metric-label"><LabelIcon /> Labels</span>
                      <strong>{datasetSummary?.label_count ?? 0}</strong>
                    </div>
                  </div>

                  <div className="label-chip-row">
                    {datasetSummary?.labels?.length ? (
                      datasetSummary.labels.map((label) => (
                        <span className="label-chip" key={label.name}>
                          <LabelIcon className="icon chip-icon" />
                          <span>{label.name} | {label.image_count}</span>
                        </span>
                      ))
                    ) : (
                      <span className="muted">No labels captured yet.</span>
                    )}
                  </div>

                  {captureSequence.isRunning ? (
                    <div className="capture-sequence-card">
                      <div className="capture-sequence-stats">
                        <div className="metric-card sequence-metric">
                          <span className="status-label metric-label"><CameraIcon /> Taken</span>
                          <strong>{captureSequence.taken}</strong>
                        </div>
                        <div className="metric-card sequence-metric">
                          <span className="status-label metric-label"><ImagesIcon /> Left</span>
                          <strong>{captureSequenceRemaining}</strong>
                        </div>
                        <div className="metric-card sequence-metric">
                          <span className="status-label metric-label"><ClockIcon /> Countdown</span>
                          <strong>{captureSequence.countdown > 0 ? `${captureSequence.countdown}s` : "--"}</strong>
                        </div>
                      </div>
                      <p className="muted compact-line">
                        {captureSequence.countdown > 0
                          ? `Next photo in ${captureSequence.countdown}s.`
                          : captureSequenceRemaining > 0
                            ? "Saving the next photo..."
                            : "Finishing the capture sequence..."}
                      </p>
                    </div>
                  ) : null}

                  <div className="button-row">
                    <button
                      className="primary-button"
                      onClick={captureAndSaveSample}
                      type="button"
                      disabled={cameraState !== "ready" || isSavingSample || !trimmedLabelInput}
                    >
                      <span className="button-content">
                        {isSavingSample ? <ClockIcon /> : <CameraIcon />}
                        {captureButtonLabel}
                      </span>
                    </button>

                    <button
                      className="secondary-button"
                      onClick={startTrainingRun}
                      type="button"
                      disabled={!canTrain || isSavingSample}
                    >
                      <span className="button-content">
                        {trainButtonIcon}
                        {isTraining ? "Training..." : "Train model"}
                      </span>
                    </button>
                  </div>

                  <p className="muted compact-line">{datasetCountsLabel}</p>
                  {trainingReason ? (
                    <p className="muted compact-line reason-row">
                      <LockIcon />
                      <span>{trainingReason}</span>
                    </p>
                  ) : null}
                  </div>
                ) : (
                  <>
                    <label className="field">
                      <span className="field-title"><ModelIcon /> Model</span>
                      <select
                        value={selectedModel}
                        onChange={(event) => setSelectedModel(event.target.value)}
                        disabled={models.length === 0}
                      >
                        {models.length === 0 ? <option>No trained models yet</option> : null}
                        {models.map((model) => (
                          <option key={model.name} value={model.name}>
                            {model.name}
                          </option>
                        ))}
                      </select>
                    </label>

                    <div className="button-row">
                      <button
                        className="primary-button"
                        onClick={captureAndClassify}
                        type="button"
                        disabled={
                          cameraState !== "ready"
                          || models.length === 0
                          || liveClassificationEnabled
                          || isClassifying
                        }
                      >
                        <span className="button-content">
                          {liveClassificationEnabled
                            ? <CameraIcon />
                            : isClassifying
                              ? <ClockIcon />
                              : <CameraIcon />}
                          {liveClassificationEnabled
                            ? "Classify current frame"
                            : isClassifying
                              ? "Classifying..."
                              : "Classify current frame"}
                        </span>
                      </button>
                      <button
                        className={
                          liveClassificationEnabled
                            ? "live-stream-button on"
                            : tfjsStatus === "converting"
                              ? "live-stream-button pending"
                              : "live-stream-button off"
                        }
                        onClick={() => {
                          if (!liveClassificationEnabled && (inferenceMode !== "tfjs" || !tfjsModel)) {
                            return;
                          }
                          setliveClassificationEnabled((value) => {
                            const nextValue = !value;
                            if (!nextValue) {
                              setClassification(null);
                            }
                            return nextValue;
                          });
                        }}
                        type="button"
                        disabled={cameraState !== "ready" || models.length === 0 || tfjsStatus === "converting"}
                      >
                        <span className="button-content">
                          <StatusDot
                            color={
                              liveClassificationEnabled
                                ? "#7ef0db"
                                : tfjsStatus === "error"
                                  ? "#ff7a7a"
                                  : "#f5b860"
                            }
                          />
                          {liveClassificationEnabled
                            ? "Live Classification ON"
                            : tfjsStatus === "converting"
                              ? "Preparing live classification"
                              : "Live Classification OFF"}
                        </span>
                      </button>

                      <span className="live-stream-hint">
                        {tfjsStatus === "converting"
                          ? "Converting model for live classification, this may take a moment..."
                          : inferenceMode === "tfjs"
                            ? "On-device model ready"
                            : tfjsStatus === "error"
                              ? "On-device model unavailable"
                              : "Select a model to prepare live classification"}
                      </span>
            
                     
                    </div>
                  </>
                )}
              </div>

              {cameraError ? <p className="message error">{cameraError}</p> : null}
              {modelsError ? <p className="message error">{modelsError}</p> : null}
              {datasetError ? <p className="message error">{datasetError}</p> : null}
              {predictionError ? <p className="message error">{predictionError}</p> : null}
              {captureMessage ? <p className="message success">{captureMessage}</p> : null}
              {trainingMessage ? <p className="message success">{trainingMessage}</p> : null}
              {modelMessage ? <p className="message success">{modelMessage}</p> : null}
              {imageActionMessage ? <p className="message success">{imageActionMessage}</p> : null}
              {trainingError ? <p className="message error">{trainingError}</p> : null}
              {!collectionMode && models.length === 0 && modelsStatus === "ready" ? (
                <p className="message">
                  Train a model first with the captured dataset so the frontend has something to
                  classify.
                </p>
              ) : null}
            </div>
          </div>
        ) : null}
      </section>

      {activeGalleryItem ? (
        <div className="modal-backdrop" onClick={handleCloseImageModal} role="presentation">
          <div
            className="image-modal"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="image-modal-title"
          >
            <div className="modal-header">
              <div>
                <p className="panel-label">Edit Training Image</p>
                <h2 id="image-modal-title">{activeGalleryItem.label}</h2>
              </div>
              <button className="ghost-button compact-button" onClick={handleCloseImageModal} type="button">
                <span className="button-content"><EditIcon /> Close</span>
              </button>
            </div>

            <div className="modal-layout">
              <div className="modal-image-frame">
                <img
                  className="modal-image"
                  src={activeGalleryItem.previewUrl}
                  alt={activeGalleryItem.label}
                />
              </div>

              <div className="modal-controls">
                <label className="field">
                  <span className="field-title"><LabelIcon /> Label</span>
                  <input
                    className="text-input"
                    type="text"
                    value={modalLabelDraft}
                    onChange={(event) => setModalLabelDraft(event.target.value)}
                  />
                </label>

                <p className="sample-path">{activeGalleryItem.relativePath}</p>

                <div className="modal-meta-grid">
                  <div className="modal-meta-item">
                    <span className="status-label meta-line"><CalendarIcon /> Date</span>
                    <strong>{formatDateOnly(activeGalleryItem.capturedAt)}</strong>
                  </div>
                  <div className="modal-meta-item">
                    <span className="status-label meta-line"><ClockIcon /> Time</span>
                    <strong>{formatTimeOnly(activeGalleryItem.capturedAt)}</strong>
                  </div>
                  <div className="modal-meta-item">
                    <span className="status-label meta-line"><ImagesIcon /> Resolution</span>
                    <strong>
                      {formatResolution(
                        activeGalleryItem.width,
                        activeGalleryItem.height,
                        activeGalleryItem.resolution
                      )}
                    </strong>
                  </div>
                  <div className="modal-meta-item">
                    <span className="status-label meta-line"><CameraIcon /> File size</span>
                    <strong>{formatFileSize(activeGalleryItem.fileSizeBytes)}</strong>
                  </div>
                </div>

                <div className="button-row">
                  <button
                    className="primary-button"
                    onClick={() => handleRelabelImage(activeGalleryItem.relativePath, modalLabelDraft)}
                    type="button"
                    disabled={imageActionPath === activeGalleryItem.relativePath}
                  >
                    <span className="button-content">
                      {imageActionPath === activeGalleryItem.relativePath ? <ClockIcon /> : <EditIcon />}
                      {imageActionPath === activeGalleryItem.relativePath ? "Saving..." : "Save label"}
                    </span>
                  </button>

                  <button
                    className="danger-button"
                    onClick={() => handleDeleteImage(activeGalleryItem.relativePath)}
                    type="button"
                    disabled={imageActionPath === activeGalleryItem.relativePath}
                  >
                    <span className="button-content"><TrashIcon /> Delete image</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {pendingModelDelete ? (
        <div className="modal-backdrop" onClick={closeDeleteModelModal} role="presentation">
          <div
            className="image-modal model-delete-modal"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="delete-model-modal-title"
          >
            <div className="modal-header">
              <div>
                <p className="panel-label">Delete Model</p>
                <h2 id="delete-model-modal-title">Confirm deletion</h2>
              </div>
            </div>

            <p className="muted">
              Delete model <strong>{pendingModelDelete}</strong>? This action permanently removes
              its files from the models folder.
            </p>

            <div className="button-row">
              <button
                className="ghost-button"
                onClick={closeDeleteModelModal}
                type="button"
                disabled={isDeletingModel}
              >
                <span className="button-content">Cancel</span>
              </button>
              <button
                className="danger-button"
                onClick={() => handleDeleteModel(pendingModelDelete)}
                type="button"
                disabled={isDeletingModel}
              >
                <span className="button-content">
                  {isDeletingModel ? <ClockIcon /> : <TrashIcon />}
                  {isDeletingModel ? "Deleting..." : "Delete model"}
                </span>
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <section className="results-grid">
        <article className="results-card">
          <p className="panel-label">Top Result</p>
          {activePrediction ? (
            <>
              <h2>{activePrediction.label}</h2>
              <p className="score">{formatConfidence(activePrediction.confidence)}</p>
            </>
          ) : (
            <>
              <h2>Waiting for a frame</h2>
              <p className="muted">Capture a frame to see the model&apos;s best guess here.</p>
            </>
          )}
        </article>

        <article className="results-card">
          <p className="panel-label">Prediction Stack</p>
          <h2>Confidence breakdown</h2>
          {classification?.predictions?.length ? (
            <div className="prediction-list">
              {classification.predictions.map((prediction) => (
                <div className="prediction-row" key={prediction.label}>
                  <div className="prediction-copy">
                    <strong>{prediction.label}</strong>
                    <span>{formatConfidence(prediction.confidence)}</span>
                  </div>
                  <div className="meter">
                    <div
                      className="meter-fill"
                      style={{ width: `${Math.max(prediction.confidence * 100, 3)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">
              The full ranked list of predictions will appear here after classification.
            </p>
          )}
        </article>

        <article className="results-card">
          <div className="card-header-inline">
            <div>
              <p className="panel-label">Model Info</p>
              <h2>{selectedModelMeta?.name || "No active model"}</h2>
            </div>
            <button
              className="danger-button"
              onClick={openDeleteModelModal}
              type="button"
              disabled={!selectedModel || isDeletingModel}
            >
              <span className="button-content">
                {isDeletingModel ? <ClockIcon /> : <TrashIcon />}
                {isDeletingModel ? "Deleting..." : "Delete model"}
              </span>
            </button>
          </div>
          <dl className="meta-grid">
            <div>
              <dt>Input size</dt>
              <dd>{classification?.image_size || selectedModelMeta?.image_size || "--"}</dd>
            </div>
            <div>
              <dt>Available models</dt>
              <dd>{models.length}</dd>
            </div>
            <div>
              <dt>Dataset status</dt>
              <dd>{trainingSummary?.is_current_dataset_trained ? "up to date" : "needs training"}</dd>
            </div>
            <div>
              <dt>Train date</dt>
              <dd className="meta-line"><CalendarIcon /> {formatTimestamp(selectedModelMeta?.trained_at)}</dd>
            </div>
            <div>
              <dt>Train time</dt>
              <dd className="meta-line"><ClockIcon /> {formatDuration(selectedModelMeta?.training_duration_seconds)}</dd>
            </div>
          </dl>
        </article>

        <article className="results-card log-card">
          <p className="panel-label">Available Models</p>
          <h2>Model catalog</h2>
          {models.length ? (
            <div className="training-log-list">
              {models.map((model) => (
                <div className="training-log-item" key={`model-catalog-${model.name}`}>
                  <div className="training-log-head">
                    <strong>{model.name}</strong>
                    <div className="log-head-meta">
                      <span className="log-badge">
                        <ModelIcon className="icon chip-icon" />
                        <span>{model.model_type || "Keras CNN"}</span>
                      </span>
                      <span className="log-meta-item"><CalendarIcon /> {formatTimestamp(model.trained_at)}</span>
                      <span className="log-meta-item"><ClockIcon /> {formatDuration(model.training_duration_seconds)}</span>
                      <button
                        className="danger-button compact-button"
                        onClick={() => openDeleteModelModal(model.name)}
                        type="button"
                        disabled={isDeletingModel}
                      >
                        <span className="button-content"><TrashIcon /> Delete</span>
                      </button>
                    </div>
                  </div>

                  <div className="models-metadata-grid">
                    <div>
                      <dt>Images</dt>
                      <dd>{model.dataset_total_images ?? "--"}</dd>
                    </div>
                    <div>
                      <dt>Labels</dt>
                      <dd>{model.dataset_label_count ?? model.class_count ?? "--"}</dd>
                    </div>
                    <div>
                      <dt>Model size</dt>
                      <dd>{formatFileSize(model.run_size_bytes)}</dd>
                    </div>
                    <div>
                      <dt>Input size</dt>
                      <dd>{model.image_size ?? "--"}</dd>
                    </div>
                    <div>
                      <dt>Val Acc (best)</dt>
                      <dd>
                        {Number.isFinite(model?.accuracy?.val_accuracy_best)
                          ? formatConfidence(model.accuracy.val_accuracy_best)
                          : "--"}
                      </dd>
                    </div>
                    <div>
                      <dt>Val Acc (final)</dt>
                      <dd>
                        {Number.isFinite(model?.accuracy?.val_accuracy_final)
                          ? formatConfidence(model.accuracy.val_accuracy_final)
                          : "--"}
                      </dd>
                    </div>
                  </div>

                  <div className="label-chip-row">
                    {(model.dataset_labels ?? []).length
                      ? model.dataset_labels.map((label) => (
                        <span className="label-chip" key={`${model.name}-${label.name}`}>
                          <LabelIcon className="icon chip-icon" />
                          <span>{label.name} | {label.image_count}</span>
                        </span>
                      ))
                      : (model.labels ?? []).map((label) => (
                        <span className="label-chip" key={`${model.name}-${label}`}>
                          <LabelIcon className="icon chip-icon" />
                          <span>{label}</span>
                        </span>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">No models available yet.</p>
          )}
        </article>

        <article className="results-card log-card">
          <p className="panel-label">Training Log</p>
          <h2>Recorded training runs</h2>
          {trainingLogs.length ? (
            <div className="training-log-list">
              {trainingLogs.map((entry) => (
                <div className="training-log-item" key={`${entry.trained_at}-${entry.model_name}`}>
                  <div className="training-log-head">
                    <strong>{entry.model_name}</strong>
                    <div className="log-head-meta">
                      <span className={availableModelNames.has(entry.model_name) ? "log-badge" : "log-badge muted-badge"}>
                        {availableModelNames.has(entry.model_name) ? <ModelIcon className="icon chip-icon" /> : <TrashIcon className="icon chip-icon" />}
                        <span>{availableModelNames.has(entry.model_name) ? "available" : "deleted"}</span>
                      </span>
                      <span className="log-meta-item"><CalendarIcon /> {formatTimestamp(entry.trained_at)}</span>
                      <span className="log-meta-item"><ClockIcon /> {formatDuration(entry.training_duration_seconds)}</span>
                    </div>
                  </div>
                  <p className="muted compact-line">
                    {pluralize(entry.dataset?.total_images ?? 0, "image")} across{" "}
                    {pluralize(entry.dataset?.label_count ?? 0, "label")}
                  </p>
                  <div className="label-chip-row">
                    {(entry.dataset?.labels ?? []).map((label) => (
                      <span className="label-chip" key={`${entry.model_name}-${label.name}`}>
                        <LabelIcon className="icon chip-icon" />
                        <span>{label.name} | {label.image_count}</span>
                      </span>
                    ))}
                  </div>
                  <details className="log-details">
                    <summary>View images used for this training run</summary>
                    <div className="log-files">
                      {(entry.dataset?.labels ?? []).flatMap((label) =>
                        (label.images ?? []).map((imagePath) => (
                          <code key={`${entry.model_name}-${imagePath}`}>{imagePath}</code>
                        ))
                      )}
                    </div>
                  </details>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">
              No training runs have been recorded yet. Capture labeled images, then train a model.
            </p>
          )}
        </article>
      </section>
    </main>
  );
}
