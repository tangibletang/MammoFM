const POLL_INTERVAL = 3000;

const STEP_MAP = {
  encoding:   "step-encoding",
  stage1:     "step-stage1",
  converting: "step-converting",
  stage2:     "step-stage2",
};

function setStepActive(status) {
  Object.values(STEP_MAP).forEach(id => {
    document.getElementById(id).className = "step";
  });
  if (STEP_MAP[status]) {
    document.getElementById(STEP_MAP[status]).className = "step active";
  }
}

function markStepsDone(status) {
  const order = ["encoding", "stage1", "converting", "stage2"];
  const idx = order.indexOf(status);
  order.slice(0, idx).forEach(s => {
    document.getElementById(STEP_MAP[s]).className = "step done";
  });
}

function showError(msg) {
  document.getElementById("errorMsg").textContent = msg;
  document.getElementById("errorBox").classList.remove("hidden");
}

async function pollStatus(jobId) {
  while (true) {
    const res = await fetch(`/api/status/${jobId}`);
    const data = await res.json();

    document.getElementById("statusMsg").textContent = data.message || data.status;

    if (data.status === "done") {
      markStepsDone("done");
      Object.values(STEP_MAP).forEach(id => {
        document.getElementById(id).className = "step done";
      });
      const r = await fetch(`/api/results/${jobId}`);
      const reports = await r.json();
      document.getElementById("prelimReport").textContent = reports.preliminary_report;
      document.getElementById("finalReport").textContent  = reports.final_report;
      document.getElementById("results").classList.remove("hidden");
      return;
    }

    if (data.status === "failed") {
      showError(data.error || "Pipeline failed.");
      return;
    }

    setStepActive(data.status);
    markStepsDone(data.status);
    await new Promise(r => setTimeout(r, POLL_INTERVAL));
  }
}

document.getElementById("reportForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  document.getElementById("progress").classList.remove("hidden");
  document.getElementById("results").classList.add("hidden");
  document.getElementById("errorBox").classList.add("hidden");
  document.getElementById("submitBtn").disabled = true;
  document.getElementById("statusMsg").textContent = "Submitting…";

  const fd = new FormData(e.target);
  try {
    const res = await fetch("/api/run", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json();
      showError(err.detail || "Submission failed.");
      document.getElementById("submitBtn").disabled = false;
      return;
    }
    const { job_id } = await res.json();
    await pollStatus(job_id);
  } catch (err) {
    showError(err.message);
  }
  document.getElementById("submitBtn").disabled = false;
});
