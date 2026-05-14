const state = {
  providers: [],
  activeProvider: "",
  facts: 0,
  episodes: 0,
  onboardingStep: 0,
  selectedFile: null,
};

const els = {};

document.addEventListener("DOMContentLoaded", async () => {
  bindElements();
  bindEvents();
  await loadDashboard();
  maybeShowOnboarding();
});

function bindElements() {
  els.providerSelect = document.getElementById("providerSelect");
  els.providerStatusDot = document.getElementById("providerStatusDot");
  els.providerStatusText = document.getElementById("providerStatusText");
  els.memoryFileInput = document.getElementById("memoryFileInput");
  els.importMemoryBtn = document.getElementById("importMemoryBtn");
  els.importStatus = document.getElementById("importStatus");
  els.copyProxyUrlBtn = document.getElementById("copyProxyUrlBtn");
  els.proxyBaseUrl = document.getElementById("proxyBaseUrl");
  els.openOnboardingBtn = document.getElementById("openOnboardingBtn");
  els.refreshDashboardBtn = document.getElementById("refreshDashboardBtn");

  els.heroFacts = document.getElementById("heroFacts");
  els.heroEpisodes = document.getElementById("heroEpisodes");
  els.heroProviders = document.getElementById("heroProviders");

  els.onboardingOverlay = document.getElementById("onboardingOverlay");
  els.closeOnboardingBtn = document.getElementById("closeOnboardingBtn");
  els.wizardBackBtn = document.getElementById("wizardBackBtn");
  els.wizardNextBtn = document.getElementById("wizardNextBtn");
  els.wizardSkipBtn = document.getElementById("wizardSkipBtn");
  els.wizardFinishBtn = document.getElementById("wizardFinishBtn");
  els.wizardCopyProxyBtn = document.getElementById("wizardCopyProxyBtn");
  els.wizardSteps = Array.from(document.querySelectorAll(".wizard-step"));
  els.wizardPanels = Array.from(document.querySelectorAll(".wizard-panel"));
  els.onboardingProgressBar = document.getElementById("onboardingProgressBar");
  els.onboardingProgressText = document.getElementById("onboardingProgressText");
  els.wizardProviderSelect = document.getElementById("wizardProviderSelect");
  els.wizardProviderName = document.getElementById("wizardProviderName");
  els.wizardProviderStatus = document.getElementById("wizardProviderStatus");
  els.wizardFileInput = document.getElementById("wizardFileInput");
  els.wizardFileName = document.getElementById("wizardFileName");
  els.wizardImportBtn = document.getElementById("wizardImportBtn");
  els.wizardImportStatus = document.getElementById("wizardImportStatus");
  els.wizardProxyBaseUrl = document.getElementById("wizardProxyBaseUrl");
  els.wizardDropzone = document.getElementById("wizardDropzone");
}

function bindEvents() {
  els.refreshDashboardBtn?.addEventListener("click", loadDashboard);
  els.openOnboardingBtn?.addEventListener("click", () => openOnboarding(0));
  els.copyProxyUrlBtn?.addEventListener("click", () => copyText("http://localhost:11435/v1", "Proxy base URL copied"));
  els.wizardCopyProxyBtn?.addEventListener("click", () => copyText("http://localhost:11435/v1", "Proxy base URL copied"));
  els.closeOnboardingBtn?.addEventListener("click", closeOnboarding);
  els.wizardSkipBtn?.addEventListener("click", closeOnboarding);
  els.wizardBackBtn?.addEventListener("click", () => setOnboardingStep(Math.max(0, state.onboardingStep - 1)));
  els.wizardNextBtn?.addEventListener("click", () => setOnboardingStep(Math.min(3, state.onboardingStep + 1)));
  els.wizardFinishBtn?.addEventListener("click", closeOnboarding);

  document.querySelectorAll("[data-close-onboarding]").forEach((el) => {
    el.addEventListener("click", closeOnboarding);
  });

  els.providerSelect?.addEventListener("change", async (e) => {
    await updateProvider(e.target.value);
    syncProviderSelections(e.target.value);
  });

  els.wizardProviderSelect?.addEventListener("change", async (e) => {
    await updateProvider(e.target.value);
    syncProviderSelections(e.target.value);
  });

  els.memoryFileInput?.addEventListener("change", (e) => {
    state.selectedFile = e.target.files?.[0] || null;
    updateSelectedFileUI();
  });

  els.wizardFileInput?.addEventListener("change", (e) => {
    state.selectedFile = e.target.files?.[0] || null;
    syncFileInputs("wizard");
    updateSelectedFileUI();
  });

  els.importMemoryBtn?.addEventListener("click", importSelectedFile);
  els.wizardImportBtn?.addEventListener("click", importSelectedFile);

  if (els.wizardDropzone) {
    ["dragenter", "dragover"].forEach((eventName) => {
      els.wizardDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        els.wizardDropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      els.wizardDropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        els.wizardDropzone.classList.remove("dragover");
      });
    });

    els.wizardDropzone.addEventListener("drop", (e) => {
      const file = e.dataTransfer?.files?.[0];
      if (!file) return;
      state.selectedFile = file;
      syncFileInputs();
      updateSelectedFileUI();
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !els.onboardingOverlay?.classList.contains("hidden")) {
      closeOnboarding();
    }
  });
}

async function loadDashboard() {
  const [rootData, statsData] = await Promise.allSettled([
    fetchJSON("/health").catch(() => fetchJSON("/")),
    fetchJSON("/api/facts/stats"),
  ]);

  if (rootData.status === "fulfilled") {
    const data = rootData.value || {};
    const providers = Array.isArray(data.providers) ? data.providers : [];
    state.providers = providers;
    state.activeProvider = data.active_provider || state.activeProvider || "";
    renderProviders();
  }

  if (statsData.status === "fulfilled") {
    const stats = statsData.value || {};
    state.facts = Number(stats.total_facts || stats.facts || 0);
    state.episodes = Number(stats.total_episodes || stats.episodes || 0);
  } else {
    state.facts = 0;
    state.episodes = 0;
  }

  updateHero();
  updateSelectedFileUI();
  updateProxyUrls();
}

function renderProviders() {
  const names = state.providers.length ? state.providers : ["ollama", "openai", "anthropic", "groq"];
  populateSelect(els.providerSelect, names, state.activeProvider);
  populateSelect(els.wizardProviderSelect, names, state.activeProvider);
  syncProviderSelections(state.activeProvider || names[0]);

  if (els.providerStatusDot) els.providerStatusDot.classList.add("online");
  if (els.providerStatusText) {
    els.providerStatusText.textContent = state.activeProvider
      ? `Velqua will forward requests to ${state.activeProvider}.`
      : "Choose a provider to begin.";
  }
}

function populateSelect(select, options, value) {
  if (!select) return;
  select.innerHTML = "";
  options.forEach((option) => {
    const el = document.createElement("option");
    el.value = option;
    el.textContent = titleCase(option);
    if (option === value) el.selected = true;
    select.appendChild(el);
  });
}

function syncProviderSelections(value) {
  if (!value) return;
  state.activeProvider = value;
  if (els.providerSelect) els.providerSelect.value = value;
  if (els.wizardProviderSelect) els.wizardProviderSelect.value = value;
  if (els.wizardProviderName) els.wizardProviderName.textContent = titleCase(value);
  if (els.wizardProviderStatus) els.wizardProviderStatus.textContent = "Ready";
  if (els.providerStatusText) {
    els.providerStatusText.textContent = `Velqua will forward requests to ${value}.`;
  }
}

async function updateProvider(providerName) {
  if (!providerName) return;
  state.activeProvider = providerName;
  syncProviderSelections(providerName);

  try {
    await fetch("/api/providers/active", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: providerName }),
    });
    toast(`Active provider set to ${titleCase(providerName)}`, "success");
  } catch {
    toast("Provider updated in the UI. Backend endpoint was unavailable.", "info");
  }
}

async function importSelectedFile() {
  if (!state.selectedFile) {
    setImportStatus("Choose a file first.", true);
    return;
  }

  setImportStatus(`Importing ${state.selectedFile.name}…`, false);

  const formData = new FormData();
  formData.append("file", state.selectedFile);

  try {
    const response = await fetch("/api/import", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Import failed (${response.status})`);
    }

    setImportStatus(`Imported ${state.selectedFile.name}`, false);
    toast("Memory file imported", "success");
    state.selectedFile = null;
    clearFileInputs();
    await loadDashboard();

    if (state.facts > 0 && state.onboardingStep === 1) {
      setOnboardingStep(2);
    }
  } catch {
    setImportStatus("Import endpoint unavailable. Wire your file import to /api/import.", true);
    toast("Import could not be completed", "error");
  }
}

function setImportStatus(message, isError = false) {
  if (els.importStatus) {
    els.importStatus.textContent = message;
    els.importStatus.classList.toggle("error", isError);
  }
  if (els.wizardImportStatus) {
    els.wizardImportStatus.textContent = message;
    els.wizardImportStatus.classList.toggle("error", isError);
  }
}

function updateSelectedFileUI() {
  const label = state.selectedFile
    ? state.selectedFile.name
    : "Drop a file here or choose one";

  if (els.wizardFileName) els.wizardFileName.textContent = label;

  if (!state.selectedFile) {
    setImportStatus("No file selected yet.", false);
  }
}

function syncFileInputs(source = "dashboard") {
  const dt = new DataTransfer();
  if (state.selectedFile) dt.items.add(state.selectedFile);

  if (source !== "dashboard" && els.memoryFileInput) {
    els.memoryFileInput.files = dt.files;
  }
  if (source !== "wizard" && els.wizardFileInput) {
    els.wizardFileInput.files = dt.files;
  }
}

function clearFileInputs() {
  if (els.memoryFileInput) els.memoryFileInput.value = "";
  if (els.wizardFileInput) els.wizardFileInput.value = "";
}

function updateHero() {
  if (els.heroFacts) els.heroFacts.textContent = state.facts;
  if (els.heroEpisodes) els.heroEpisodes.textContent = state.episodes;
  if (els.heroProviders) els.heroProviders.textContent = state.providers.length || "—";
}

function maybeShowOnboarding() {
  const dismissed = localStorage.getItem("velqua-onboarding-dismissed") === "true";
  if (state.facts === 0 && !dismissed) {
    openOnboarding(0);
  }
}

function openOnboarding(step = 0) {
  els.onboardingOverlay?.classList.remove("hidden");
  document.body.classList.add("modal-open");
  setOnboardingStep(step);
}

function closeOnboarding() {
  els.onboardingOverlay?.classList.add("hidden");
  document.body.classList.remove("modal-open");
  localStorage.setItem("velqua-onboarding-dismissed", "true");
}

function setOnboardingStep(step) {
  state.onboardingStep = step;

  els.wizardSteps.forEach((item, index) => {
    item.classList.toggle("active", index === step);
    item.classList.toggle("complete", index < step);
  });

  els.wizardPanels.forEach((panel, index) => {
    panel.classList.toggle("active", index === step);
  });

  const progress = ((step + 1) / 4) * 100;
  if (els.onboardingProgressBar) els.onboardingProgressBar.style.width = `${progress}%`;
  if (els.onboardingProgressText) els.onboardingProgressText.textContent = `Step ${step + 1} of 4`;

  if (els.wizardBackBtn) els.wizardBackBtn.disabled = step === 0;
  if (els.wizardNextBtn) els.wizardNextBtn.classList.toggle("hidden", step === 3);
  if (els.wizardFinishBtn) els.wizardFinishBtn.classList.toggle("hidden", step !== 3);
}

function updateProxyUrls() {
  const baseUrl = "http://localhost:11435/v1";
  if (els.proxyBaseUrl) els.proxyBaseUrl.textContent = baseUrl;
  if (els.wizardProxyBaseUrl) els.wizardProxyBaseUrl.textContent = baseUrl;
}

async function copyText(text, successMessage) {
  try {
    await navigator.clipboard.writeText(text);
    toast(successMessage, "success");
  } catch {
    toast("Could not copy to clipboard", "error");
  }
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed: ${response.status}`);
  return response.json();
}

function titleCase(value) {
  return String(value || "")
    .split(/[\s_-]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function toast(message, type = "info") {
  const root = document.getElementById("toastRoot");
  if (!root) return;

  const node = document.createElement("div");
  node.className = `toast toast-${type}`;
  node.textContent = message;
  root.appendChild(node);

  requestAnimationFrame(() => node.classList.add("show"));
  setTimeout(() => {
    node.classList.remove("show");
    setTimeout(() => node.remove(), 220);
  }, 2600);
}
