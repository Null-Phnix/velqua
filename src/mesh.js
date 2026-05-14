/**
 * Mesh Page — Live agent visualization via WebSocket.
 *
 * Connects to /mesh/stream for real-time agent heartbeat events.
 * Falls back to polling /mesh/agents when WebSocket is unavailable.
 * Renders glass cards with status dots, pulse animations, and live metadata.
 */

(function () {
  "use strict";

  // ============================================================
  // Config
  // ============================================================

  const API_BASE = window.location.origin;
  const MAX_WS_RETRIES = 3;
  const POLL_INTERVAL_MS = 8000;
  const ACTIVE_THRESHOLD_S = 60;   // green dot
  const IDLE_THRESHOLD_S = 120;    // yellow dot (> active, < idle = idle; > idle = disconnected)

  // ============================================================
  // State
  // ============================================================

  let ws = null;
  let wsRetries = 0;
  let pollTimer = null;
  let agents = {};          // agent_id -> agent data
  let currentPage = "home";

  // ============================================================
  // DOM References
  // ============================================================

  const grid = document.getElementById("meshAgentGrid");
  const emptyState = document.getElementById("meshEmptyState");
  const connectionDot = document.getElementById("meshConnectionDot");
  const connectionStatus = document.getElementById("meshConnectionStatus");
  const agentCount = document.getElementById("meshAgentCount");
  const navBadge = document.getElementById("meshNavBadge");

  // ============================================================
  // Page Navigation
  // ============================================================

  const homePage = document.querySelector("main.layout");
  const meshPage = document.getElementById("meshPage");
  const activityPage = document.getElementById("activityPage");
  const navPills = document.querySelectorAll(".nav-pill");

  navPills.forEach(function (pill) {
    pill.addEventListener("click", function () {
      var page = pill.dataset.page;
      if (!page || page === currentPage) return;
      switchPage(page);
    });
  });

  function switchPage(page) {
    currentPage = page;

    navPills.forEach(function (p) {
      var isActive = p.dataset.page === page;
      p.classList.toggle("active", isActive);
      p.setAttribute("aria-selected", String(isActive));
    });

    // Hide all pages
    homePage.classList.add("hidden");
    meshPage.classList.add("hidden");
    if (activityPage) activityPage.classList.add("hidden");

    if (page === "home") {
      homePage.classList.remove("hidden");
    } else if (page === "mesh") {
      meshPage.classList.remove("hidden");
      // Trigger initial load if we haven't connected yet
      if (!ws && wsRetries === 0) {
        connectWebSocket();
      }
    } else if (page === "activity") {
      activityPage.classList.remove("hidden");
      // Trigger activity load if available
      if (window._loadActivity) window._loadActivity();
    }
  }

  // ============================================================
  // WebSocket Connection
  // ============================================================

  function connectWebSocket() {
    var port = window.location.port || extractPort();
    var wsUrl = "ws://127.0.0.1:" + port + "/mesh/stream";

    try {
      ws = new WebSocket(wsUrl);
    } catch (e) {
      fallbackToPolling();
      return;
    }

    ws.onopen = function () {
      wsRetries = 0;
      setConnectionStatus("live");
    };

    ws.onmessage = function (evt) {
      try {
        var msg = JSON.parse(evt.data);
        handleWsEvent(msg);
      } catch (e) { /* ignore parse errors */ }
    };

    ws.onclose = function () {
      ws = null;
      setConnectionStatus("reconnecting");
      wsRetries++;
      if (wsRetries <= MAX_WS_RETRIES) {
        setTimeout(connectWebSocket, 2000 * wsRetries);
      } else {
        setConnectionStatus("polling");
        fallbackToPolling();
      }
    };

    ws.onerror = function () {
      // onclose will fire after this
    };
  }

  function extractPort() {
    var match = API_BASE.match(/:(\d+)/);
    return match ? match[1] : "8765";
  }

  function fallbackToPolling() {
    loadAgentsViaRest();
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(loadAgentsViaRest, POLL_INTERVAL_MS);
  }

  // ============================================================
  // WebSocket Event Handling
  // ============================================================

  function handleWsEvent(msg) {
    switch (msg.type) {
      case "snapshot":
        handleSnapshot(msg.data);
        break;
      case "agent_heartbeat":
        handleAgentHeartbeat(msg.data);
        break;
      case "ping":
        if (msg.data.agents) {
          handleSnapshot({ agents: msg.data.agents });
        } else {
          updateAgentCount(msg.data.active_agents || 0);
        }
        break;
    }
  }

  function handleSnapshot(data) {
    var agentList = data.agents || [];
    agents = {};
    agentList.forEach(function (a) {
      agents[a.id] = a;
    });
    renderAllCards();
  }

  function handleAgentHeartbeat(agentData) {
    if (!agentData || !agentData.id) return;
    var isNew = !agents[agentData.id];
    agents[agentData.id] = agentData;

    if (isNew) {
      renderAllCards();
    } else {
      updateCard(agentData);
      pulseCard(agentData.id);
    }

    updateAgentCount(Object.keys(agents).length);
  }

  // ============================================================
  // REST Fallback
  // ============================================================

  function loadAgentsViaRest() {
    fetch(API_BASE + "/mesh/agents?active_only=false")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        agents = {};
        (data.agents || []).forEach(function (a) {
          agents[a.id] = a;
        });
        renderAllCards();
      })
      .catch(function () {
        // Silently fail — will retry on next poll
      });
  }

  // ============================================================
  // Card Rendering
  // ============================================================

  function renderAllCards() {
    var ids = Object.keys(agents);
    updateAgentCount(ids.filter(function (id) { return agents[id].is_active; }).length);

    if (ids.length === 0) {
      grid.textContent = "";
      emptyState.classList.remove("hidden");
      return;
    }

    emptyState.classList.add("hidden");

    // Sort: active first, then by last_seen_ago ascending
    ids.sort(function (a, b) {
      var aa = agents[a], bb = agents[b];
      if (aa.is_active !== bb.is_active) return aa.is_active ? -1 : 1;
      return (aa.last_seen_ago || 0) - (bb.last_seen_ago || 0);
    });

    // Build new cards
    grid.textContent = "";
    ids.forEach(function (id, i) {
      var card = createCard(agents[id]);
      card.style.animationDelay = (i * 0.06) + "s";
      grid.appendChild(card);
    });
  }

  function createCard(agent) {
    var status = getAgentStatus(agent);

    var card = document.createElement("div");
    card.className = "mesh-card";
    card.dataset.agentId = agent.id;
    card.dataset.status = status;

    // Status dot column
    var statusCol = document.createElement("div");
    statusCol.className = "mesh-card-status";
    var dot = document.createElement("span");
    dot.className = "mesh-card-dot " + status;
    statusCol.appendChild(dot);
    card.appendChild(statusCol);

    // Body
    var body = document.createElement("div");
    body.className = "mesh-card-body";

    // Name
    var name = document.createElement("h4");
    name.className = "mesh-card-name";
    name.textContent = agent.name || agent.id;
    body.appendChild(name);

    // Meta row — last seen, message count
    var meta = document.createElement("div");
    meta.className = "mesh-card-meta";

    meta.appendChild(createMetaItem(
      "\u23F1",  // stopwatch
      formatAgo(agent.last_seen_ago),
      "mesh-card-seen"
    ));

    var msgCount = agent.message_count || 0;
    meta.appendChild(createMetaItem(
      "\u2709",  // envelope
      msgCount.toLocaleString() + " message" + (msgCount !== 1 ? "s" : ""),
      "mesh-card-messages"
    ));

    body.appendChild(meta);

    // Task
    if (agent.current_task) {
      var taskEl = document.createElement("div");
      taskEl.className = "mesh-card-task";

      var taskLabel = document.createElement("span");
      taskLabel.className = "mesh-card-task-label";
      taskLabel.textContent = "Current task";
      taskEl.appendChild(taskLabel);

      var taskText = document.createTextNode(
        agent.current_task.length > 140
          ? agent.current_task.slice(0, 140) + "\u2026"
          : agent.current_task
      );
      taskEl.appendChild(taskText);
      body.appendChild(taskEl);
    }

    card.appendChild(body);
    return card;
  }

  function createMetaItem(icon, text, className) {
    var item = document.createElement("span");
    item.className = "mesh-card-meta-item" + (className ? " " + className : "");

    var iconEl = document.createElement("span");
    iconEl.className = "mesh-card-meta-icon";
    iconEl.textContent = icon;
    item.appendChild(iconEl);

    var value = document.createElement("span");
    value.className = "mesh-card-meta-value";
    value.textContent = text;
    item.appendChild(value);

    return item;
  }

  // ============================================================
  // Card Updates (in-place, no full re-render)
  // ============================================================

  function updateCard(agent) {
    var card = grid.querySelector('[data-agent-id="' + agent.id + '"]');
    if (!card) {
      renderAllCards();
      return;
    }

    var status = getAgentStatus(agent);
    card.dataset.status = status;

    // Update dot
    var dot = card.querySelector(".mesh-card-dot");
    if (dot) dot.className = "mesh-card-dot " + status;

    // Update last seen
    var seenEl = card.querySelector(".mesh-card-seen .mesh-card-meta-value");
    if (seenEl) seenEl.textContent = formatAgo(agent.last_seen_ago);

    // Update message count
    var msgEl = card.querySelector(".mesh-card-messages .mesh-card-meta-value");
    if (msgEl) {
      var c = agent.message_count || 0;
      msgEl.textContent = c.toLocaleString() + " message" + (c !== 1 ? "s" : "");
    }

    // Update task
    var taskEl = card.querySelector(".mesh-card-task");
    if (agent.current_task) {
      if (!taskEl) {
        taskEl = document.createElement("div");
        taskEl.className = "mesh-card-task";
        var label = document.createElement("span");
        label.className = "mesh-card-task-label";
        label.textContent = "Current task";
        taskEl.appendChild(label);
        taskEl.appendChild(document.createTextNode(""));
        card.querySelector(".mesh-card-body").appendChild(taskEl);
      }
      // Update text node (last child)
      var textNode = taskEl.lastChild;
      if (textNode.nodeType === Node.TEXT_NODE) {
        textNode.textContent = agent.current_task.length > 140
          ? agent.current_task.slice(0, 140) + "\u2026"
          : agent.current_task;
      }
    } else if (taskEl) {
      taskEl.remove();
    }
  }

  // ============================================================
  // Pulse Animation
  // ============================================================

  function pulseCard(agentId) {
    var card = grid.querySelector('[data-agent-id="' + agentId + '"]');
    if (!card) return;

    // Remove then re-add to restart animation
    card.classList.remove("pulse");
    // Force reflow so the browser recognizes the class removal
    void card.offsetWidth;
    card.classList.add("pulse");

    // Clean up after animation completes
    setTimeout(function () {
      card.classList.remove("pulse");
    }, 1200);
  }

  // ============================================================
  // Status Helpers
  // ============================================================

  function getAgentStatus(agent) {
    if (!agent.is_active) return "disconnected";
    if (agent.last_seen_ago <= ACTIVE_THRESHOLD_S) return "active";
    if (agent.last_seen_ago <= IDLE_THRESHOLD_S) return "idle";
    return "disconnected";
  }

  function setConnectionStatus(status) {
    var labels = { live: "Live", reconnecting: "Reconnecting\u2026", polling: "Polling" };
    if (connectionStatus) connectionStatus.textContent = labels[status] || status;
    if (connectionDot) {
      connectionDot.className = "mesh-connection-dot " + status;
    }
  }

  function updateAgentCount(n) {
    if (agentCount) agentCount.textContent = String(n);
    // Update nav badge
    if (navBadge) {
      if (n > 0) {
        navBadge.textContent = String(n);
        navBadge.style.display = "inline-flex";
      } else {
        navBadge.style.display = "none";
      }
    }
  }

  function formatAgo(seconds) {
    if (seconds == null) return "unknown";
    if (seconds < 5) return "just now";
    if (seconds < 60) return seconds + "s ago";
    if (seconds < 3600) return Math.floor(seconds / 60) + "m ago";
    if (seconds < 86400) return Math.floor(seconds / 3600) + "h ago";
    return Math.floor(seconds / 86400) + "d ago";
  }

  // ============================================================
  // Auto-refresh last_seen_ago timestamps
  // ============================================================

  setInterval(function () {
    // Increment all last_seen_ago counters locally
    Object.keys(agents).forEach(function (id) {
      agents[id].last_seen_ago = (agents[id].last_seen_ago || 0) + 1;
    });

    // Update displayed timestamps without full re-render
    var cards = grid.querySelectorAll(".mesh-card");
    cards.forEach(function (card) {
      var id = card.dataset.agentId;
      var agent = agents[id];
      if (!agent) return;

      var seenEl = card.querySelector(".mesh-card-seen .mesh-card-meta-value");
      if (seenEl) seenEl.textContent = formatAgo(agent.last_seen_ago);

      // Update status dot if threshold crossed
      var newStatus = getAgentStatus(agent);
      if (card.dataset.status !== newStatus) {
        card.dataset.status = newStatus;
        var dot = card.querySelector(".mesh-card-dot");
        if (dot) dot.className = "mesh-card-dot " + newStatus;
      }
    });
  }, 1000);

  // ============================================================
  // Init — start WebSocket immediately so badge updates on Home too
  // ============================================================

  connectWebSocket();

})();
