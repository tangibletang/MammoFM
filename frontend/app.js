const POLL_INTERVAL = 3000;
const VIEW_ORDER = ["LCC", "LMLO", "RCC", "RMLO"];
const STEP_MAP = {
  encoding: "step-encoding",
  stage1: "step-stage1",
  converting: "step-converting",
  stage2: "step-stage2",
};

const STATUS_PROGRESS_FALLBACK = {
  pending: 0.05,
  encoding: 0.2,
  stage1: 0.45,
  converting: 0.7,
  stage2: 0.85,
  done: 1,
  failed: 1,
};

let uploadFiles = [];
let lastPreview = { slots: {}, warnings: [], rows: [] };
let galleryObjectUrls = [];

function revokeGalleryUrls() {
  galleryObjectUrls.forEach((u) => URL.revokeObjectURL(u));
  galleryObjectUrls = [];
}

function setProgressFromStatus(status, apiProgress) {
  const bar = document.getElementById("jobProgress");
  if (!bar) return;
  const v =
    typeof apiProgress === "number"
      ? apiProgress
      : STATUS_PROGRESS_FALLBACK[status] ?? 0.05;
  bar.value = v;
}

function setStepActive(status) {
  Object.values(STEP_MAP).forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.className = "step";
  });
  if (STEP_MAP[status]) {
    const el = document.getElementById(STEP_MAP[status]);
    if (el) el.className = "step active";
  }
}

function markStepsDone(status) {
  const order = ["encoding", "stage1", "converting", "stage2"];
  const idx = order.indexOf(status);
  order.slice(0, idx).forEach((s) => {
    const el = document.getElementById(STEP_MAP[s]);
    if (el) el.className = "step done";
  });
}

function showError(msg) {
  document.getElementById("errorMsg").textContent = msg;
  document.getElementById("errorBox").classList.remove("hidden");
}

function formatErrorDetail(detail) {
  if (detail == null) return "Request failed.";
  if (typeof detail === "string") return detail;
  if (detail.message && typeof detail.message === "string") {
    const w = detail.warnings?.length
      ? ` ${detail.warnings.join(" ")}`
      : "";
    return detail.message + w;
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

function fileByName(name) {
  return uploadFiles.find((f) => f.name === name);
}

function renderChecklist(rows) {
  const tbody = document.getElementById("checklistBody");
  tbody.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => {
      const td = document.createElement("td");
      td.textContent = cell ?? "";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function renderGallery(slots) {
  const host = document.getElementById("thumbGallery");
  host.innerHTML = "";
  revokeGalleryUrls();

  VIEW_ORDER.forEach((view) => {
    const slot = document.createElement("div");
    slot.className = "thumb-slot";
    const label = document.createElement("span");
    label.className = "thumb-label";
    label.textContent = view;
    slot.appendChild(label);

    const name = slots[view];
    const file = name ? fileByName(name) : null;
    if (file) {
      const url = URL.createObjectURL(file);
      galleryObjectUrls.push(url);
      const img = document.createElement("img");
      img.src = url;
      img.alt = `${view}: ${file.name}`;
      slot.appendChild(img);
    } else {
      const ph = document.createElement("div");
      ph.className = "thumb-placeholder";
      ph.textContent = "—";
      slot.appendChild(ph);
    }
    host.appendChild(slot);
  });
}

function renderWarnings(warnings) {
  const box = document.getElementById("previewWarnings");
  if (!warnings.length) {
    box.classList.add("hidden");
    box.innerHTML = "";
    return;
  }
  box.classList.remove("hidden");
  box.innerHTML =
    "<strong>Upload checks</strong><ul>" +
    warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("") +
    "</ul>";
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function runPreview() {
  const fd = new FormData();
  uploadFiles.forEach((f) => fd.append("files", f));

  if (!uploadFiles.length) {
    lastPreview = { slots: {}, warnings: [], rows: [] };
    renderChecklist(lastPreview.rows);
    renderGallery({});
    renderWarnings([]);
    return;
  }

  try {
    const res = await fetch("/api/preview", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) {
      lastPreview = { slots: {}, warnings: [formatErrorDetail(data.detail)], rows: data.rows || [] };
    } else {
      lastPreview = data;
    }
  } catch (e) {
    lastPreview = { slots: {}, warnings: [e.message || String(e)], rows: [] };
  }

  renderChecklist(lastPreview.rows || []);
  renderGallery(lastPreview.slots || {});
  renderWarnings(lastPreview.warnings || []);
}

function syncFilesFromList(fileList) {
  uploadFiles = Array.from(fileList || []);
  runPreview();
}

async function pollStatus(jobId) {
  while (true) {
    const res = await fetch(`/api/status/${jobId}`);
    const data = await res.json();

    document.getElementById("statusMsg").textContent = data.message || data.status;
    setProgressFromStatus(data.status, data.progress);

    if (data.status === "done") {
      markStepsDone("done");
      Object.values(STEP_MAP).forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.className = "step done";
      });
      setProgressFromStatus("done", 1);
      const r = await fetch(`/api/results/${jobId}`);
      const reports = await r.json();
      if (!r.ok) {
        showError(formatErrorDetail(reports.detail) || "Could not load results.");
        return;
      }
      document.getElementById("prelimReport").textContent =
        reports.preliminary_report || "";
      document.getElementById("finalReport").textContent = reports.final_report || "";
      document.getElementById("results").classList.remove("hidden");
      return;
    }

    if (data.status === "failed") {
      showError(data.error || "Pipeline failed.");
      return;
    }

    setStepActive(data.status);
    markStepsDone(data.status);
    await new Promise((r) => setTimeout(r, POLL_INTERVAL));
  }
}

document.getElementById("fileInput").addEventListener("change", (e) => {
  syncFilesFromList(e.target.files);
});

const dropZone = document.getElementById("dropZone");
dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  syncFilesFromList(e.dataTransfer.files);
});

document.getElementById("refreshDetection").addEventListener("click", () => runPreview());

document.getElementById("reportForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const n = Object.keys(lastPreview.slots || {}).length;
  if (n !== 4) {
    showError("All four views (LCC, LMLO, RCC, RMLO) must be detected before running.");
    return;
  }

  document.getElementById("progress").classList.remove("hidden");
  document.getElementById("results").classList.add("hidden");
  document.getElementById("errorBox").classList.add("hidden");
  document.getElementById("submitBtn").disabled = true;
  document.getElementById("statusMsg").textContent = "Submitting…";
  setProgressFromStatus("pending", STATUS_PROGRESS_FALLBACK.pending);

  const fd = new FormData();
  uploadFiles.forEach((f) => fd.append("files", f));
  const pid = document.getElementById("patientId").value.trim();
  const eid = document.getElementById("examId").value.trim();
  if (pid) fd.append("patient_id", pid);
  if (eid) fd.append("exam_id", eid);
  const csv = document.getElementById("classifierCsv").files[0];
  if (csv) fd.append("classifier_csv", csv);

  try {
    const res = await fetch("/api/run", { method: "POST", body: fd });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      showError(formatErrorDetail(body.detail));
      document.getElementById("submitBtn").disabled = false;
      return;
    }
    document.getElementById("jobIdField").value = body.job_id || "";
    await pollStatus(body.job_id);
  } catch (err) {
    showError(err.message || String(err));
  }
  document.getElementById("submitBtn").disabled = false;
});

document.querySelectorAll(".btn-copy").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const id = btn.getAttribute("data-copy-target");
    const el = document.getElementById(id);
    const text = el?.textContent || "";
    try {
      await navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
      setTimeout(() => {
        btn.textContent = "Copy";
      }, 1500);
    } catch {
      showError("Clipboard not available.");
    }
  });
});

/* initial empty table */
renderChecklist([]);
