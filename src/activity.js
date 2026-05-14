/**
 * Activity Page — chronological event timeline.
 *
 * Fetches events from GET /activity and renders a clean timeline
 * with icons per event type, date separators, and pagination.
 * Uses createElement throughout to avoid innerHTML XSS.
 */

(function () {
  "use strict";

  var API_BASE = window.location.origin;
  var PAGE_SIZE = 50;
  var currentPage = 0;
  var currentFilter = "";

  var ICONS = {
    fact_learned:       "\uD83E\uDDE0",
    fact_approved:      "\u2705",
    fact_rejected:      "\u274C",
    fact_deleted:       "\uD83D\uDDD1",
    fact_merged:        "\uD83D\uDD00",
    fact_edited:        "\u270F\uFE0F",
    import_completed:   "\uD83D\uDCE5",
    import_failed:      "\u26A0\uFE0F",
    backup_created:     "\uD83D\uDCBE",
    backup_restored:    "\uD83D\uDD04",
    provider_changed:   "\uD83D\uDD17",
    agent_connected:    "\uD83E\uDD16",
    agent_disconnected: "\uD83D\uDEAB",
    system_started:     "\u26A1",
  };

  var TYPE_LABELS = {
    fact_learned:       "Fact Learned",
    fact_approved:      "Approved",
    fact_rejected:      "Rejected",
    fact_deleted:       "Deleted",
    fact_merged:        "Merged",
    fact_edited:        "Edited",
    import_completed:   "Import",
    import_failed:      "Import Failed",
    backup_created:     "Backup",
    backup_restored:    "Restore",
    provider_changed:   "Provider",
    agent_connected:    "Agent",
    agent_disconnected: "Agent Left",
    system_started:     "System",
  };

  // ============================================================
  // Init
  // ============================================================

  var filterSelect = document.getElementById("activityFilter");
  var refreshBtn = document.getElementById("activityRefreshBtn");
  var prevBtn = document.getElementById("activityPrevBtn");
  var nextBtn = document.getElementById("activityNextBtn");

  if (filterSelect) {
    filterSelect.addEventListener("change", function () {
      currentFilter = filterSelect.value;
      currentPage = 0;
      loadActivity();
    });
  }

  if (refreshBtn) refreshBtn.addEventListener("click", loadActivity);

  if (prevBtn) {
    prevBtn.addEventListener("click", function () {
      if (currentPage > 0) {
        currentPage--;
        loadActivity();
      }
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener("click", function () {
      currentPage++;
      loadActivity();
    });
  }

  // Expose for mesh.js page routing
  window._loadActivity = loadActivity;

  // ============================================================
  // Data Fetching
  // ============================================================

  function loadActivity() {
    var feed = document.getElementById("activityFeed");
    var pageInfo = document.getElementById("activityPageInfo");
    if (!feed) return;

    var offset = currentPage * PAGE_SIZE;
    var url = API_BASE + "/activity?limit=" + PAGE_SIZE + "&offset=" + offset;
    if (currentFilter) url += "&event_type=" + encodeURIComponent(currentFilter);

    fetch(url)
      .then(function (r) { return r.json(); })
      .then(function (data) {
        renderTimeline(data.events || [], feed);

        var total = data.total || 0;
        var totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
        if (pageInfo) pageInfo.textContent = "Page " + (currentPage + 1) + " of " + totalPages;
        if (prevBtn) prevBtn.disabled = currentPage === 0;
        if (nextBtn) nextBtn.disabled = offset + PAGE_SIZE >= total;
      })
      .catch(function () {
        feed.textContent = "";
        var wrapper = document.createElement("div");
        wrapper.className = "activity-empty";
        var card = document.createElement("div");
        card.className = "empty-state-card";
        var iconWrap = document.createElement("div");
        iconWrap.className = "empty-state-icon";
        iconWrap.setAttribute("aria-hidden", "true");
        iconWrap.innerHTML = '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>';
        card.appendChild(iconWrap);
        var heading = document.createElement("h3");
        heading.textContent = "Could not load activity";
        card.appendChild(heading);
        var msg = document.createElement("p");
        msg.textContent = "The activity log is unavailable right now. Check that the backend is running and try again.";
        card.appendChild(msg);
        var retryBtn = document.createElement("button");
        retryBtn.className = "btn btn-primary";
        retryBtn.textContent = "Retry";
        retryBtn.addEventListener("click", loadActivity);
        card.appendChild(retryBtn);
        wrapper.appendChild(card);
        feed.appendChild(wrapper);
      });
  }

  // ============================================================
  // Timeline Rendering
  // ============================================================

  function renderTimeline(events, container) {
    container.textContent = "";

    if (!events || events.length === 0) {
      // Remove the timeline line when empty
      container.style.position = "static";

      var wrapper = document.createElement("div");
      wrapper.className = "activity-empty";

      var card = document.createElement("div");
      card.className = "empty-state-card";

      var iconWrap = document.createElement("div");
      iconWrap.className = "empty-state-icon";
      iconWrap.setAttribute("aria-hidden", "true");
      iconWrap.innerHTML = '<svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>';
      card.appendChild(iconWrap);

      var heading = document.createElement("h3");
      heading.textContent = "No activity yet";
      card.appendChild(heading);

      var text = document.createElement("p");
      text.textContent = "Events will appear here as you use Velqua \u2014 imports, learned facts, provider changes, and more.";
      card.appendChild(text);

      var cta = document.createElement("button");
      cta.className = "btn btn-primary";
      cta.textContent = "Import your first memory";
      cta.addEventListener("click", function () {
        var homePill = document.querySelector('.nav-pill[data-page="home"]');
        if (homePill) homePill.click();
      });
      card.appendChild(cta);

      wrapper.appendChild(card);
      container.appendChild(wrapper);
      return;
    }

    container.style.position = "relative";
    var lastDateStr = "";

    events.forEach(function (event) {
      var dateStr = formatDate(event.timestamp);

      if (dateStr !== lastDateStr) {
        lastDateStr = dateStr;
        var dateSep = document.createElement("div");
        dateSep.className = "activity-date-separator";
        var dateLabel = document.createElement("span");
        dateLabel.textContent = dateStr;
        dateSep.appendChild(dateLabel);
        container.appendChild(dateSep);
      }

      var item = document.createElement("div");
      item.className = "activity-item";

      // Icon
      var iconEl = document.createElement("div");
      iconEl.className = "activity-icon";
      iconEl.textContent = ICONS[event.event_type] || "\u2022";
      item.appendChild(iconEl);

      // Body
      var body = document.createElement("div");
      body.className = "activity-body";

      // Header row: badge + time
      var headerRow = document.createElement("div");
      headerRow.className = "activity-header-row";

      var badge = document.createElement("span");
      badge.className = "activity-badge activity-badge--" + eventCategory(event.event_type);
      badge.textContent = TYPE_LABELS[event.event_type] || event.event_type;
      headerRow.appendChild(badge);

      var time = document.createElement("span");
      time.className = "activity-time";
      time.textContent = formatTime(event.timestamp);
      headerRow.appendChild(time);

      body.appendChild(headerRow);

      // Title
      var title = document.createElement("p");
      title.className = "activity-title";
      title.textContent = event.title;
      body.appendChild(title);

      // Detail (optional)
      if (event.detail) {
        var detail = document.createElement("p");
        detail.className = "activity-detail";
        detail.textContent = event.detail;
        body.appendChild(detail);
      }

      item.appendChild(body);
      container.appendChild(item);
    });
  }

  // ============================================================
  // Helpers
  // ============================================================

  function eventCategory(type) {
    if (type.startsWith("fact_")) return "fact";
    if (type.startsWith("import")) return "import";
    if (type.startsWith("backup")) return "backup";
    if (type.startsWith("provider")) return "provider";
    if (type.startsWith("agent")) return "agent";
    return "system";
  }

  function formatDate(ts) {
    var d = new Date(ts * 1000);
    var now = new Date();
    var diff = now - d;
    if (diff < 86400000 && d.getDate() === now.getDate()) return "Today";
    var yesterday = new Date(now);
    yesterday.setDate(yesterday.getDate() - 1);
    if (diff < 172800000 && d.getDate() === yesterday.getDate()) return "Yesterday";
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
  }

  function formatTime(ts) {
    return new Date(ts * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  }

})();
