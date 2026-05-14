/**
 * Command palette — Ctrl+K / Cmd+K to open.
 * Raycast/Linear-inspired: glass modal, fuzzy filter, keyboard navigation.
 */
(function () {
  "use strict";

  let overlay = null;
  let input = null;
  let list = null;
  let hint = null;
  let activeIndex = 0;
  let filteredActions = [];

  /** Registry of all available actions. */
  const actions = [
    // Navigation
    { id: "nav-home", label: "Go to Home", group: "Navigate", icon: "\u2302", keywords: "home main dashboard overview", action: () => clickNav("home") },
    { id: "nav-mesh", label: "Go to Mesh", group: "Navigate", icon: "\u25CE", keywords: "agents network connected mesh", action: () => clickNav("mesh") },
    { id: "nav-activity", label: "Go to Activity", group: "Navigate", icon: "\u26A1", keywords: "activity log timeline events history feed", action: () => clickNav("activity") },

    // Actions
    { id: "search-facts", label: "Search Facts", group: "Actions", icon: "\u2315", keywords: "find memory query search", action: focusFactSearch },
    { id: "import-file", label: "Import Memory File", group: "Actions", icon: "\u2191", keywords: "upload add file conversation import", action: triggerImport },
    { id: "refresh", label: "Refresh Dashboard", group: "Actions", icon: "\u21BB", keywords: "reload update sync refresh", action: () => document.getElementById("refreshDashboardBtn")?.click() },
    { id: "copy-url", label: "Copy Proxy URL", group: "Actions", icon: "\u2398", keywords: "clipboard base url endpoint copy", action: copyProxyUrl },
    { id: "export-backup", label: "Create Backup", group: "Actions", icon: "\u2193", keywords: "export download save json backup", action: exportBackup },

    // Settings
    { id: "toggle-theme", label: "Toggle Theme", group: "Settings", icon: "\u263D", keywords: "dark light mode theme toggle appearance", action: () => document.getElementById("themeToggleBtn")?.click() },
    { id: "onboarding", label: "Getting Started Guide", group: "Settings", icon: "?", keywords: "onboarding setup wizard help guide", action: () => document.getElementById("openOnboardingBtn")?.click() },
  ];

  document.addEventListener("DOMContentLoaded", init);

  function init() {
    overlay = document.getElementById("commandPaletteOverlay");
    input = document.getElementById("commandPaletteInput");
    list = document.getElementById("commandPaletteList");
    hint = document.getElementById("commandPaletteHint");

    if (!overlay || !input || !list) return;

    // Global hotkey
    document.addEventListener("keydown", function (e) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        toggle();
      }
    });

    // Trigger button in topbar
    var triggerBtn = document.getElementById("commandPaletteBtn");
    if (triggerBtn) {
      triggerBtn.addEventListener("click", function () { open(); });
    }

    // Detect platform for modifier key label
    var modKey = document.querySelector(".cmd-mod-key");
    if (modKey) {
      modKey.textContent = /Mac|iPhone|iPad/.test(navigator.platform || "") ? "\u2318" : "Ctrl+";
    }

    // Close on backdrop click
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay || e.target.classList.contains("cmd-backdrop")) {
        close();
      }
    });

    // Input events
    input.addEventListener("input", function () {
      activeIndex = 0;
      render();
    });

    input.addEventListener("keydown", function (e) {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          activeIndex = Math.min(activeIndex + 1, filteredActions.length - 1);
          updateActive();
          break;
        case "ArrowUp":
          e.preventDefault();
          activeIndex = Math.max(activeIndex - 1, 0);
          updateActive();
          break;
        case "Enter":
          e.preventDefault();
          if (filteredActions[activeIndex]) {
            executeAction(filteredActions[activeIndex]);
          }
          break;
        case "Escape":
          e.preventDefault();
          close();
          break;
      }
    });
  }

  function toggle() {
    if (overlay.classList.contains("hidden")) {
      open();
    } else {
      close();
    }
  }

  function open() {
    overlay.classList.remove("hidden");
    overlay.classList.add("visible");
    document.body.classList.add("modal-open");
    input.value = "";
    activeIndex = 0;
    render();
    requestAnimationFrame(function () { input.focus(); });
  }

  function close() {
    overlay.classList.remove("visible");
    overlay.classList.add("hidden");
    document.body.classList.remove("modal-open");
    input.value = "";
  }

  function getFilteredActions() {
    var query = input.value.trim().toLowerCase();
    if (!query) return actions.slice();

    return actions.filter(function (a) {
      var haystack = (a.label + " " + a.group + " " + a.keywords).toLowerCase();
      return query.split(/\s+/).every(function (word) {
        return haystack.indexOf(word) !== -1;
      });
    });
  }

  function render() {
    filteredActions = getFilteredActions();
    list.innerHTML = "";

    if (filteredActions.length === 0) {
      var empty = document.createElement("div");
      empty.className = "cmd-empty";

      var iconWrap = document.createElement("div");
      iconWrap.className = "cmd-empty-icon";
      iconWrap.setAttribute("aria-hidden", "true");
      iconWrap.innerHTML = '<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/></svg>';
      empty.appendChild(iconWrap);

      var heading = document.createElement("h4");
      heading.textContent = "No matching actions";
      empty.appendChild(heading);

      var subtext = document.createElement("p");
      subtext.textContent = "Try a different search term or press Esc to close.";
      empty.appendChild(subtext);

      list.appendChild(empty);
      if (hint) hint.textContent = "";
      return;
    }

    var currentGroup = "";

    filteredActions.forEach(function (action, index) {
      // Group header
      if (action.group !== currentGroup) {
        currentGroup = action.group;
        var groupEl = document.createElement("div");
        groupEl.className = "cmd-group";
        groupEl.textContent = currentGroup;
        list.appendChild(groupEl);
      }

      var row = document.createElement("button");
      row.className = "cmd-row";
      row.type = "button";
      row.dataset.index = index;
      if (index === activeIndex) row.classList.add("active");

      var icon = document.createElement("span");
      icon.className = "cmd-icon";
      icon.textContent = action.icon;

      var label = document.createElement("span");
      label.className = "cmd-label";
      label.textContent = action.label;

      var badge = document.createElement("span");
      badge.className = "cmd-badge";
      badge.textContent = action.group;

      row.appendChild(icon);
      row.appendChild(label);
      row.appendChild(badge);

      row.addEventListener("click", function () { executeAction(action); });
      row.addEventListener("mouseenter", function () {
        activeIndex = index;
        updateActive();
      });

      list.appendChild(row);
    });

    if (hint) {
      hint.textContent = filteredActions.length + " action" + (filteredActions.length !== 1 ? "s" : "");
    }
  }

  function updateActive() {
    var rows = list.querySelectorAll(".cmd-row");
    rows.forEach(function (row, i) {
      row.classList.toggle("active", i === activeIndex);
    });

    var activeRow = list.querySelector(".cmd-row.active");
    if (activeRow) {
      activeRow.scrollIntoView({ block: "nearest" });
    }
  }

  function executeAction(action) {
    close();
    requestAnimationFrame(function () { action.action(); });
  }

  // --- Action implementations ---

  function clickNav(pageName) {
    var pill = document.querySelector('.nav-pill[data-page="' + pageName + '"]');
    if (pill) pill.click();
  }

  function focusFactSearch() {
    var searchInput = document.getElementById("factSearch");
    if (searchInput) {
      searchInput.focus();
      searchInput.select();
    }
  }

  function triggerImport() {
    var fileInput = document.getElementById("memoryFileInput");
    if (fileInput) fileInput.click();
  }

  function copyProxyUrl() {
    try {
      navigator.clipboard.writeText("http://localhost:11435/v1").then(function () {
        showToast("Proxy base URL copied", "success");
      });
    } catch (err) {
      showToast("Could not copy to clipboard", "error");
    }
  }

  function exportBackup() {
    fetch("/api/facts/list?limit=10000&offset=0")
      .then(function (response) {
        if (!response.ok) throw new Error("Export failed");
        return response.json();
      })
      .then(function (data) {
        var blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = "velqua-backup-" + new Date().toISOString().slice(0, 10) + ".json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        showToast("Backup downloaded", "success");
      })
      .catch(function () {
        showToast("Backup failed \u2014 could not reach the API", "error");
      });
  }

  function showToast(message, type) {
    var root = document.getElementById("toastRoot");
    if (!root) return;
    var node = document.createElement("div");
    node.className = "toast toast-" + type;
    node.textContent = message;
    root.appendChild(node);
    requestAnimationFrame(function () { node.classList.add("show"); });
    setTimeout(function () {
      node.classList.remove("show");
      setTimeout(function () { node.remove(); }, 220);
    }, 2600);
  }
})();
