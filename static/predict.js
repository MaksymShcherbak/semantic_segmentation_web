document.addEventListener("DOMContentLoaded", () => {
  const introSection = document.querySelector(".intro");

  const dropZone = document.querySelector(".drop-zone");
  dropZone.addEventListener("click", () => fileInput.click());

  const results = document.getElementById("results");
  const spinner = document.getElementById("spinner");
  const overlayRange = document.getElementById("overlay");

  const updateOverlay = () => {
    for (const model_name of model_names) {
      const overlayCanvas = document.getElementById(
        `mask-canvas-${model_name}`,
      );
      overlayCanvas.style.opacity = overlayRange.value;
    }
  };

  overlayRange.addEventListener("change", updateOverlay);
  overlayRange.checked = false;

  const getIoUColor = (iou) => {
    if (iou < 0.3) return "#cd1d1d";
    if (iou < 0.7) return "#8B8000";
    return "#177117";
  };

  let allModels = {};
  let selectedModel = null;
  const modal = document.getElementById("modal");
  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.style.display = "none";
    }
  });
  const modalClose = document.getElementById("modal-close");
  modalClose.onclick = () => {
    modal.style.display = "none";
  };
  const modalContent = document.getElementById("modal-content");

  const openOverviewTab = () => {
    modal.style.display = "flex";
    modalContent.innerHTML = `
      <h2>${selectedModel.description.name}</h2>
      <p>${selectedModel.description.summary}</p>
      <img class="summary" src="/static/${selectedModel.model_name}.png">
    `;
  };

  const makeTable = (headers, rows, highlight_column = "") => {
    let container = document.createElement("div");
    const headerRowDiv = document.createElement("div");
    headerRowDiv.className = "stats-row sts-header";

    for (const header of headers) {
      const div = document.createElement("div");
      div.className = `stats-header-title`;
      div.innerText = header.name;
      if (header.tooltip) {
        const questionMark = document.createElement("span");
        questionMark.className = "tooltip";
        questionMark.innerHTML =
          "? <span class='tooltip-text'>" + header.tooltip + "</span>";
        div.appendChild(questionMark);
      }

      if (highlight_column == header.name) {
        div.classList.add("highlighted");
      }
      headerRowDiv.appendChild(div);
    }

    container.appendChild(headerRowDiv);

    rows.forEach((row) => {
      const rowDiv = document.createElement("div");
      rowDiv.className = "legend-row";
      let bold = false;
      row.forEach((item, index) => {
        const header = headers[index];
        let div = document.createElement("div");
        if (item != "") div.className = `legend-${header.id}`;
        if (header.id == "color" && item != "") {
          div.style.backgroundColor = item;
        } else if (header.id == "iou") {
          div.style.color = getIoUColor(item);
          div.innerText = item;
        } else {
          div.innerText = item;
        }
        if (item == "Mean" || bold) {
          bold = true;
          div.classList.add("bold");
        }
        if (highlight_column == header.name) div.classList.add("highlighted");
        rowDiv.appendChild(div);
      });
      container.appendChild(rowDiv);
    });

    return container;
  };

  const openStatsTab = () => {
    modal.style.display = "flex";

    const headers = [
      {
        id: "color",
        name: "Color",
        tooltip: "",
      },
      { id: "name", name: "Name", tooltip: "" },
      {
        id: "iou",
        name: "IoU",
        tooltip:
          "Intersection over Union (IoU) measures the overlap between the predicted segmentation and the ground truth segmentation for each class.",
      },
      {
        id: "tp",
        name: "TP",
        tooltip:
          "True Positives (TP) are the number of pixels correctly predicted as belonging to a specific class.",
      },
      {
        id: "tn",
        name: "TN",
        tooltip:
          "True Negatives (TN) are the number of pixels correctly predicted as not belonging to a specific class.",
      },
      {
        id: "fp",
        name: "FP",
        tooltip:
          "False Positives (FP) are the number of pixels incorrectly predicted as belonging to a specific class.",
      },
      {
        id: "fn",
        name: "FN",
        tooltip:
          "False Negatives (FN) are the number of pixels incorrectly predicted as not belonging to a specific class.",
      },
      {
        id: "precision",
        name: "Precision",
        tooltip:
          "Precision is the ratio of true positive predictions to the total predicted positives (TP / (TP + FP)). It indicates how many of the predicted positive pixels are actually correct.",
      },
      {
        id: "recall",
        name: "Recall",
        tooltip:
          "Recall is the ratio of true positive predictions to the total actual positives (TP / (TP + FN)). It indicates how many of the actual positive pixels were correctly identified.",
      },
      {
        id: "f1_score",
        name: "F1 Score",
        tooltip:
          "The F1 Score is the harmonic mean of precision and recall (2 * (Precision * Recall) / (Precision + Recall)). It provides a single metric that balances both precision and recall.",
      },
      {
        id: "fraction",
        name: "Fraction",
        tooltip:
          "Fraction indicates the proportion of pixels in the predicted segmentation that are assigned to a specific class.",
      },
    ];

    const rows = selectedModel.stats.map((item) => {
      const {
        name,
        color,
        tp,
        tn,
        fp,
        fn,
        iou,
        precision,
        recall,
        f1_score,
        fraction,
      } = item;

      return [
        color,
        name,
        iou.toFixed(4),
        tp,
        tn,
        fp,
        fn,
        precision.toFixed(4),
        recall.toFixed(4),
        f1_score.toFixed(4),
        fraction.toFixed(2),
      ];
    });

    const container = makeTable(headers, rows);
    modalContent.innerHTML = `
      <h2>Training Statistics (${selectedModel.model_name})</h2>
      <p>${selectedModel.description.performance}</p>
      ${container.outerHTML}`;
  };

  const openComparisonTab = (current_model_name) => {
    console.log(current_model_name);
    modal.style.display = "flex";

    const headers = [
      {
        id: "color",
        name: "Color",
        tooltip: "",
      },
      { id: "name", name: "Name", tooltip: "" },
    ];

    const rows = [];

    for (let model_name in allModels) {
      headers.push({
        id: "iou",
        name: model_name + " IoU",
      });
    }

    for (let i = 0; i < allModels[model_names[0]].stats.length; i++) {
      const values = [];
      values.push(allModels[model_names[0]].stats[i].color);
      values.push(allModels[model_names[0]].stats[i].name);
      for (let model_name in allModels) {
        let model = allModels[model_name];
        values.push(model.stats[i].iou.toFixed(4));
      }
      rows.push(values);
    }

    const values = [];
    values.push("");
    values.push("Mean");
    for (let model_name in allModels) {
      let model = allModels[model_name];
      values.push(model.stats[0].mean_iou.toFixed(4));
    }
    rows.push(values);

    const container = makeTable(headers, rows, current_model_name + " IoU");
    modalContent.innerHTML = `
      <h2>Model Comparison (${selectedModel.model_name})</h2>
      <p>${selectedModel.description.comparison}</p>${container.outerHTML}`;
  };

  const overviewTab = document.getElementById("overview-tab");
  const statsTab = document.getElementById("stats-tab");
  const comparisonTab = document.getElementById("comparison-tab");
  overviewTab.onclick = openOverviewTab;
  statsTab.onclick = openStatsTab;
  comparisonTab.onclick = () => openComparisonTab(selectedModel.model_name);

  const drawImage = (canvas, img) => {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;

    canvas.width = w;
    canvas.height = h;

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = false;

    const scale = Math.min(w / img.width, h / img.height);
    const dw = img.width * scale;
    const dh = img.height * scale;
    const dx = (w - dw) / 2;
    const dy = (h - dh) / 2;

    ctx.drawImage(img, dx, dy, dw, dh);
  };
  const processResponse = async (blob) => {
    const zip = await JSZip.loadAsync(blob);

    for (const model_name of model_names) {
      const origBlob = await zip.file(`orig-${model_name}.png`).async("blob");
      const maskBlob = await zip.file(`mask-${model_name}.png`).async("blob");

      const stats = JSON.parse(
        await zip.file(`legend-${model_name}.json`).async("string"),
      );
      const description = JSON.parse(
        await zip.file(`description-${model_name}.json`).async("string"),
      );

      allModels[model_name] = { model_name, description, stats };

      const openInfo = () => {
        selectedModel = {
          model_name,
          description,
          stats,
        };
        const modalName = document.getElementById("modal-name");
        modalName.innerHTML = "Selected Model: <span>" + model_name + "</span>";
        openOverviewTab();
      };

      const detailsButton = document.getElementById(`details-${model_name}`);
      detailsButton.onclick = openInfo;

      const origImg = new Image();
      const maskImg = new Image();

      origImg.src = URL.createObjectURL(origBlob);
      maskImg.src = URL.createObjectURL(maskBlob);

      const origCanvasStandalone = document.getElementById("original-canvas");
      const origCanvas = document.getElementById(
        `original-canvas-${model_name}`,
      );
      const maskCanvas = document.getElementById(`mask-canvas-${model_name}`);
      maskCanvas.onclick = openInfo;

      origImg.onload = () => {
        drawImage(origCanvasStandalone, origImg);
        drawImage(origCanvas, origImg);
      };
      maskImg.onload = () => drawImage(maskCanvas, maskImg);
    }

    const legendContainer = document.getElementById("class-legend");
    legendContainer.innerHTML = "";

    const firstModelStats = allModels[model_names[0]].stats;

    firstModelStats.forEach((item) => {
      const legendItem = document.createElement("div");
      legendItem.className = "class-legend-item";

      const colorBox = document.createElement("div");
      colorBox.className = "class-legend-color";
      colorBox.style.backgroundColor = item.color;

      const label = document.createElement("span");
      label.innerText = item.name;

      legendItem.appendChild(colorBox);
      legendItem.appendChild(label);
      legendContainer.appendChild(legendItem);
    });
  };

  const predict = async (e) => {
    if (!fileInput.files[0]) return alert("Please select a file.");
    spinner.style.display = "inline-block";

    setTimeout(() => {
      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    }, 200);

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("image", file);

    dropZone.style.display = "none";
    introSection.style.display = "none";

    try {
      const response = await fetch("/api/predict_all", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const err = await response.json();
        return alert("Error: " + err.error);
      }

      const blob = await response.blob();
      spinner.style.display = "none";
      results.style.display = "block";
      await processResponse(blob);
    } catch (err) {
      console.error(err);
      alert("Prediction Failed: " + err.message);
    } finally {
    }
  };

  const fileInput = document.getElementById("image");
  fileInput.addEventListener("change", predict);

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");

    const files = e.dataTransfer.files;
    if (files.length === 0) return;
    if (!files[0].type.startsWith("image/")) {
      alert("Please drop an image file.");
      return;
    }

    fileInput.files = files;
    predict();
  });

  const retry = () => {
    results.style.display = "none";
    spinner.style.display = "none";
    dropZone.style.display = "flex";
    introSection.style.display = "block";

    fileInput.value = "";

    selectedModel = null;
    allModels = {};
  };

  const retryText = document.getElementById("retry");
  retryText.addEventListener("click", retry);

  function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  let imageIdsSet = new Set();

  while (imageIdsSet.size < 3) {
    imageIdsSet.add(getRandomInt(1, 13));
  }

  let imageIds = Array.from(imageIdsSet);

  let galleryImages = document.getElementById("gallery-images");

  for (let i = 0; i < imageIds.length; i++) {
    let img = new Image();
    img.src = `/static/${imageIds[i]}.jpg`;
    img.classList.add("gallery-image");

    img.style.cursor = "pointer";

    img.addEventListener("click", async () => {
      try {
        const response = await fetch(img.src);
        const blob = await response.blob();

        const file = new File([blob], `gallery-${imageIds[i]}.jpg`, {
          type: blob.type,
        });

        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;

        predict();
      } catch (err) {
        console.error(err);
        alert("Failed to load gallery image.");
      }
    });

    galleryImages.appendChild(img);
  }
});
