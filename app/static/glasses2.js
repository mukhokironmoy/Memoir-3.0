/* glasses.js */
"use strict";

/* ============================
   CONFIG
============================ */



/* ============================
   DEBUG UTIL
============================ */
const DEBUG = true;
const dbg = (...a) => DEBUG && console.log("[Memoir][DBG]", ...a);
const warn = (...a) => console.warn("[Memoir][WARN]", ...a);
const err  = (...a) => console.error("[Memoir][ERR]", ...a);

/* Track last-seen flags to reduce log spam in loops */
const DebugFlags = {
  facePresent: null,
  currentView: null,
  convoActive: null,
};
let FACE_PRESENT_NOW = false;

/* ============================
   CAMERA (persistent)
============================ */
(async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    const videoEl = document.getElementById("videoBackground");
    if (videoEl) videoEl.srcObject = stream;
  } catch (e) { err("Camera error:", e); }
})();

async function loadFaceModels() {
  try {
    const MODELS_URL = "/static/face-models";
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_URL);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODELS_URL);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODELS_URL);
    console.log("[Memoir] Face models loaded.");
  } catch (e) {
    console.error("loadFaceModels failed:", e);
  }
}


async function startFaceDetection() {
  const video  = document.getElementById("videoBackground");
  const canvas = document.getElementById("overlayCanvas");
  if (!video || !canvas) return;

  video.addEventListener("playing", () => {
    const displaySize = { width: video.videoWidth, height: video.videoHeight };
    faceapi.matchDimensions(canvas, displaySize);

    setInterval(async () => {
      let detections = [];
      try {
        detections = await faceapi.detectAllFaces(
          video,
          new faceapi.TinyFaceDetectorOptions()
        );
      } catch (e) { err("detectAllFaces failed:", e); }

      // keep only biggest face
      let detection = null;
      if (detections.length > 0) {
        detection = detections.reduce((big, cur) => {
          const A = big.box.width * big.box.height;
          const B = cur.box.width * cur.box.height;
          return B > A ? cur : big;
        });
      }

      FACE_PRESENT_NOW = !!detection;
      if (DebugFlags.facePresent !== FACE_PRESENT_NOW) {
        DebugFlags.facePresent = FACE_PRESENT_NOW;
        dbg("Face present:", FACE_PRESENT_NOW);
        reflectHeader(); // update Recognize button enable/disable
      }

      // draw
      try {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (detection) {
          const resized = faceapi.resizeResults(detection, displaySize);
          faceapi.draw.drawDetections(canvas, [resized]);
        }
      } catch (e) { err("Canvas draw failed:", e); }
    }, 200);
  });
}

/* ============================
   CLOCK (persistent)
============================ */
function updateClock() {
  const now = new Date();
  const t = document.getElementById("timeText");
  const d = document.getElementById("dateText");
  if (t) t.textContent = now.toLocaleTimeString([], { hour:"2-digit", minute:"2-digit", second:"2-digit" });
  if (d) d.textContent = now.toLocaleDateString([], { weekday:"short", day:"2-digit", month:"short", year:"numeric" });
}
updateClock(); setInterval(updateClock, 1000);

function prettyDate(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString([], {
      weekday:"short", year:"numeric", month:"short", day:"2-digit",
      hour:"2-digit", minute:"2-digit"
    });
  } catch { return "—"; }
}

/* ============================
   PEOPLE (DB-backed)
============================ */
let PEOPLE = [];
const getPersonById = (id) => PEOPLE.find(p => p.id === id) || null;

function personToViewModel(p) {
  return {
    id: p.id,
    name: p.display_name,
    relation: p.relation || "—",
    avatar: p.photo_url || "https://via.placeholder.com/160x160.png?text=?",
    lastMetIso: p.last_met_at || null,
    lastMetPretty: p.last_met_at ? prettyDate(p.last_met_at) : "—",
  };
}

async function loadPeople() {
  try {
    const res = await fetch("/glasses/api/people", { headers: { "Accept":"application/json" }});
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    PEOPLE = await res.json();
  } catch (e) { warn("Failed to load people:", e); PEOPLE = []; }
}

async function fetchPerson(personId) {
  const res = await fetch(`/glasses/api/people/${personId}`, { headers: { "Accept":"application/json" }});
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function parseSummaryBullets(text) {
  if (!text) return [];
  return text.split("\n")
    .map(s => s.trim().replace(/^[•*\-\d\.\)\s]+/, ""))
    .filter(Boolean);
}

function personApiToViewModel(p) {
  return {
    id: p.id,
    name: p.display_name,
    relation: p.relation || "—",
    avatar: p.photo_url || "https://via.placeholder.com/160x160.png?text=?",
    lastMetIso: p.last_met_at || null,
    lastMetPretty: p.last_met_at ? prettyDate(p.last_met_at) : "—",
    summaryBullets: parseSummaryBullets(p.last_summary_cached || ""),
    latestConversation: p.latest_conversation || null,
  };
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;")
    .replaceAll("\"","&quot;").replaceAll("'","&#039;");
}

/* ============================
   WEB SPEECH STT ENGINE
============================ */
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

const STT = {
  recog: null,
  isRunning: false,
  isPaused: false,
  lang: "en-IN",
  interim: "",           // rolling interim
  finalBuf: "",          // accumulated finalized text for the *current* speaker
  lastConfidence: null,

  _renderLive() {
    const box = document.getElementById("transcriptArea");
    if (!box) return;
    const safe = (s) => s.replaceAll("&","&amp;").replaceAll("<","&lt;");
    box.innerHTML = `${safe(this.finalBuf)} <span class="text-white-50">${safe(this.interim)}</span>`;
  },

  _updateMicChip() {
    const chip = document.getElementById("micStatusChip");
    if (!chip) return;
    if (!this.isRunning) { chip.textContent = "Mic: Off"; chip.className = "badge bg-secondary"; return; }
    if (this.isPaused)   { chip.textContent = "Mic: Paused"; chip.className = "badge bg-warning text-dark"; return; }
    chip.textContent = "Mic: Live"; chip.className = "badge bg-success";
  },

  _ensureRecognizer() {
    if (!SpeechRecognition) { alert("Web Speech not supported in this browser."); return null; }
    const r = new SpeechRecognition();
    r.lang = this.lang;
    r.continuous = true;
    r.interimResults = true;
    r.maxAlternatives = 1;

    r.onresult = (ev) => {
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const res = ev.results[i];
        const alt = res[0];
        if (res.isFinal) {
          this.finalBuf += (this.finalBuf && !this.finalBuf.endsWith(" ")) ? " " : "";
          this.finalBuf += alt.transcript.trim();
          this.lastConfidence = alt.confidence ?? null;
          this.interim = "";
        } else {
          this.interim = alt.transcript;
        }
      }
      this._renderLive();
    };

    // Chrome will end even with continuous; restart if still running and not paused.
    r.onend = () => {
      if (this.isRunning && !this.isPaused) {
        try { r.start(); } catch {}
      }
    };

    r.onerror = (e) => {
      console.warn("STT error:", e?.error || e);
      // On some errors, try to recover
      if (this.isRunning && !this.isPaused) {
        try { r.stop(); } catch {}
        setTimeout(() => { try { r.start(); } catch {} }, 300);
      }
    };

    return r;
  },

  async start(lang = "en-IN") {
    this.lang = lang;
    this.recog = this._ensureRecognizer();
    if (!this.recog) return;
    this.isRunning = true;
    this.isPaused = false;
    this._updateMicChip();
    this._renderLive();
    try { this.recog.start(); } catch {}
  },

  async pause() {
    if (!this.isRunning || this.isPaused) return;
    this.isPaused = true;
    try { this.recog.stop(); } catch {}
    this._updateMicChip();
  },

  async resume() {
    if (!this.isRunning || !this.isPaused) return;
    this.isPaused = false;
    try { this.recog.start(); } catch {}
    this._updateMicChip();
  },

  async stop() {
    if (!this.isRunning) return;
    this.isRunning = false;
    this.isPaused = false;
    try { this.recog.stop(); } catch {}
    this._updateMicChip();
  },

  // Flushes buffered finalized text as a DB turn (if any), for the given speaker.
  async flushTurn(reason, speaker) {
    const text = (this.finalBuf || "").trim();
    if (!text) return; // nothing to save
    const payload = {
      conversation_id: AppState.conversationId,
      text,
      speaker,                          // "Patient" / "Visitor" (we convert server-side to enum)
      confidence: this.lastConfidence,
      lang: this.lang
    };
    try {
      await fetch("/glasses/api/turns/append", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify(payload)
      });
    } catch (e) {
      console.warn("appendTurn failed:", e);
    } finally {
      this.finalBuf = "";               // clear current-speaker buffer
      this._renderLive();
    }
  }
};


/* ============================
   APP STATE + PERSISTENCE
============================ */
const AppState = {
  activePersonId: null,
  conversationActive: false,
  conversationId: null,
  conversationPersonId: null,
  activeSpeaker: "Patient",
  recognizedPersonId: null,
};

function saveState() {
  try {
    localStorage.setItem("memoirState", JSON.stringify(AppState));
  } catch(e){ warn("saveState failed:", e); }
}
function loadState() {
  try {
    const raw = localStorage.getItem("memoirState");
    if (!raw) return;
    const s = JSON.parse(raw);
    Object.assign(AppState, s);
  } catch(e){ warn("loadState failed:", e); }
}

/* Warn on tab close/refresh while recording */
window.addEventListener("beforeunload", (e) => {
  if (AppState.conversationActive) { e.preventDefault(); e.returnValue = ""; }
});

/* ============================
   TEMPLATING + RENDERING
============================ */
function applyTemplate(html, dataObj) {
  let out = html;
  Object.entries(dataObj).forEach(([k,v]) => { out = out.replaceAll(`{{${k}}}`, String(v)); });
  return out;
}

function loadView(name, data = {}) {
  const tpl = document.getElementById(`tpl-${name}`);
  if (!tpl) return;
  window.__MEMOIR_CURRENT_VIEW = name;

  if (DebugFlags.currentView !== name) {
    DebugFlags.currentView = name;
    dbg("loadView:", name);
  }

  const mount = document.getElementById("sidebarContent");
  if (!mount) return;

  const person = data.person || (AppState.activePersonId ? getPersonById(AppState.activePersonId) : null);
  const merged = {
    id: person?.id ?? "",
    name: person?.name ?? "",
    relation: person?.relation ?? "",
    avatar: person?.avatar ?? "",
    lastMetPretty: person?.lastMetPretty ?? "—",
    activeSpeaker: AppState.activeSpeaker,
    ...data,
  };

  mount.innerHTML = applyTemplate(tpl.innerHTML, merged);

  // post-render
  if (name === "home") {
    reflectHomeButtons();
  } else if (name === "contacts") {
    renderContactsList();
  } else if (name === "profile") {
    reflectProfileButtons();
  } else if (name === "conversation") {
    reflectSpeakerUI();
  }

  reflectHeader(); // ensure header is correct for this view
}

/* Sidebar header (Back + Recognize + Title) */
function reflectHeader() {
  const header = document.getElementById("sidebarHeader");
  const titleEl = document.getElementById("sidebarTitle");
  if (!header || !titleEl) return;

  const view = window.__MEMOIR_CURRENT_VIEW;
  let backAction = null;
  let title = "Memoir";

  if (view === "home")      title = "Memoir Home";
  if (view === "contacts") { title = "Contacts"; backAction = "go-home"; }
  if (view === "profile")  { title = "Profile";  backAction = "go-contacts"; }
  if (view === "conversation") { title = "Active Conversation"; backAction = "go-profile"; }

  const showRecognize = !AppState.conversationActive;
  const recognizeDisabled = !DebugFlags.facePresent;

  // Fill buttons row
  header.innerHTML = `
    <div class="d-flex align-items-center justify-content-between">
      ${backAction ? `<button class="btn btn-outline-light btn-sm" data-action="${backAction}">⬅ Back</button>` : `<span></span>`}
      ${showRecognize ? `<button class="btn btn-warning btn-sm" data-action="recognize-person" ${recognizeDisabled ? "disabled" : ""}>Recognize/Refresh</button>` : ""}
    </div>
  `;

  // Centered title below
  titleEl.textContent = title;
}



/* Home: show "Return to Active Conversation" only when active */
function reflectHomeButtons() {
  const box = document.getElementById("homeActiveConvo");
  if (!box) return;
  const visible = AppState.conversationActive;
  box.classList.toggle("d-none", !visible);
}

/* Contacts list render */
function renderContactsList() {
  const wrap = document.getElementById("contactsList");
  if (!wrap) return;

  if (!PEOPLE.length) {
    wrap.innerHTML = `<div class="text-white-50 small">No contacts yet.</div>`;
    return;
  }

  wrap.innerHTML = PEOPLE.map((p) => {
    const c = personToViewModel(p);
    return `
      <div class="glass p-2 mb-2 d-flex align-items-center justify-content-between">
        <div class="d-flex align-items-center gap-2 cursor-pointer" data-action="open-profile" data-person-id="${c.id}">
          <img src="${c.avatar}" alt="pfp" class="rounded-circle" style="width:40px;height:40px;object-fit:cover;border:1px solid rgba(255,255,255,0.25);" onerror="this.onerror=null;this.src='https://via.placeholder.com/160x160.png?text=?';" />
          <div class="small">
            <div class="fw-semibold">${c.name}</div>
            <div class="text-white-50">${c.relation}</div>
          </div>
        </div>
        <button class="btn btn-outline-light btn-sm" data-action="open-profile" data-person-id="${c.id}">View</button>
      </div>
    `;
  }).join("");
}

/* Profile view: show chips/buttons depending on convo + recognition state */
function reflectProfileButtons() {
  const chip     = document.getElementById("convStatusChip");
  const idle     = document.getElementById("profileBtnsIdle");
  const active   = document.getElementById("profileBtnsActive");
  const startBtn = document.getElementById("startBtn");
  const enrollBtn = document.getElementById("enrollBtn");

  const isActiveProfileRecognized =
    !AppState.conversationActive &&
    AppState.activePersonId != null &&
    AppState.recognizedPersonId != null &&
    AppState.activePersonId === AppState.recognizedPersonId;

  if (AppState.conversationActive) {
    chip?.classList.remove("d-none");
    idle?.classList.add("d-none");
    active?.classList.remove("d-none");
  } else {
    chip?.classList.add("d-none");
    idle?.classList.remove("d-none");
    active?.classList.add("d-none");

    // toggle Start vs Enroll separately
    if (startBtn) startBtn.classList.toggle("d-none", !isActiveProfileRecognized);
    if (enrollBtn) enrollBtn.classList.remove("d-none"); // always visible
  }
}


/* Conversation view: set chip + radio highlighting */
function reflectSpeakerUI() {
  const chip = document.getElementById("activeSpeakerChip");
  if (chip) chip.textContent = `Active: ${AppState.activeSpeaker}`;

  const isPatient = AppState.activeSpeaker === "Patient";
  const patientInput = document.getElementById("speakerPatient");
  const visitorInput = document.getElementById("speakerVisitor");
  if (patientInput && visitorInput) {
    patientInput.checked = isPatient;
    visitorInput.checked = !isPatient;
  }

  const patientLabel = document.querySelector('label[for="speakerPatient"]');
  const visitorLabel = document.querySelector('label[for="speakerVisitor"]');
  if (patientLabel && visitorLabel) {
    patientLabel.classList.toggle("active", isPatient);
    visitorLabel.classList.toggle("active", !isPatient);
  }

    // also refresh mic chip & transcript when the view mounts
  STT._updateMicChip();
  STT._renderLive();

}

/* Guard dialog when navigating away during active conversation */
function confirmLeaveActiveConversation() {
  return window.confirm(
    "Conversation is still running. Do you want to leave this tab? Conversation will keep running in the background."
  );
}

/* Utility: navigate to a profile by id */
async function openProfile(personId) {
  try {
    AppState.activePersonId = personId;
    saveState();

    const p = await fetchPerson(personId);
    const vm = personApiToViewModel(p);

    loadView("profile", { person: vm });

    const ul = document.getElementById("summaryList");
    if (ul) {
      if (vm.summaryBullets.length) {
        ul.innerHTML = vm.summaryBullets.map(line => `<li>${escapeHtml(line)}</li>`).join("");
      } else {
        ul.innerHTML = `<li class="text-white-50">No summary available yet.</li>`;
      }
    }
  } catch (e) {
    warn("Failed to open profile:", e);
    const fallback = getPersonById(personId);
    if (fallback) {
      const vm = personToViewModel(fallback);
      loadView("profile", { person: vm });
      const ul = document.getElementById("summaryList");
      if (ul) ul.innerHTML = `<li class="text-white-50">Unable to load summary.</li>`;
    } else {
      alert("Could not load profile.");
    }
  }
}

async function getCurrentFaceDescriptor() {
  const video = document.getElementById("videoBackground");
  if (!video) return null;

  const det = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!det) return null;
  return Array.from(det.descriptor); // convert Float32Array → plain array
}


/* ============================
   ACTION HANDLERS (delegation)
============================ */
document.addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-action]");
  if (!btn) return;

  const action = btn.getAttribute("data-action");
  dbg("click:", action);

  switch (action) {
    /** NAV */
case "view-contacts":
  if (AppState.conversationActive && !confirmLeaveActiveConversation()) return;
  await loadPeople();
  loadView("contacts");
  break;

case "visit-memory-bank":
  window.location.href = "/memory_bank/";
  break;

  case "view-more-info": {
  const ctx = window.__MEMOIR_ACTIVE_PERSON || {};
  const pid = Number(ctx.id || AppState.activePersonId);
  if (!pid) { alert("No profile selected."); break; }
  window.location.href = `/memory_bank/person/${pid}`;
  break;
}



    case "return-to-conversation":
      loadView("conversation"); break;

    case "go-home":
      if (AppState.conversationActive && !confirmLeaveActiveConversation()) return;
      loadView("home"); break;

    case "open-profile": {
      const id = Number(btn.getAttribute("data-person-id"));
      if (!Number.isNaN(id)) openProfile(id);
      break;
    }

    case "go-contacts":
      if (AppState.conversationActive && !confirmLeaveActiveConversation()) return;
      loadView("contacts"); break;

case "start-conversation": {
  const eligible =
    !AppState.conversationActive &&
    AppState.activePersonId != null &&
    AppState.recognizedPersonId != null &&
    AppState.activePersonId === AppState.recognizedPersonId;

  if (!eligible) { alert("Recognize the person on screen first to start a conversation."); return; }

  const personId = Number(btn.getAttribute("data-person-id")) || AppState.activePersonId || null;

  // Ask backend to open conversation
  const resp = await fetch("/glasses/api/conversations/start", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ person_id: personId, stt_lang: "en-IN" })
  });
  const data = await resp.json();
  if (!data.ok) { alert("Could not start conversation."); return; }

  AppState.conversationId = data.conversation_id;
  AppState.conversationActive = true;
  AppState.conversationPersonId = personId;
  saveState();

  // Start STT
  await STT.start("en-IN");

  loadView("conversation");
  break;
}

case "pause-resume-conversation": {
  if (!AppState.conversationActive) return;
  if (!STT.isPaused) {
    // Pause STT (UI keeps the current speaker; no flush)
    try { await fetch("/glasses/api/conversations/pause", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ conversation_id: AppState.conversationId })
    }); } catch {}
    await STT.pause();
    btn.textContent = "Resume";
  } else {
    try { await fetch("/glasses/api/conversations/resume", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({ conversation_id: AppState.conversationId })
    }); } catch {}
    await STT.resume();
    btn.textContent = "Pause";
  }
  break;
}


case "stop-conversation": {
  // Final flush for whoever is active
  await STT.flushTurn("stop", AppState.activeSpeaker);
  await STT.stop();

  if (AppState.conversationId) {
    try {
      await fetch("/glasses/api/conversations/stop", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ conversation_id: AppState.conversationId })
      });
    } catch {}
  }

  AppState.conversationActive = false;
  const prev = AppState.conversationPersonId;
  AppState.conversationId = null;
  AppState.conversationPersonId = null;
  saveState();

  if (prev) openProfile(prev); else loadView("home");
  break;
}


    case "go-profile": {
      const pid = AppState.conversationPersonId ?? AppState.activePersonId;
      if (pid) openProfile(pid); else loadView("home");
      break;
    }

    /** SPEAKER TOGGLE */
case "set-speaker": {
  const newSpeaker = btn.getAttribute("data-speaker");
  if (newSpeaker !== "Patient" && newSpeaker !== "Visitor") return;

  const prevSpeaker = AppState.activeSpeaker;         // who just finished speaking
  // Commit whatever we have so far as a turn for the previous speaker
  await STT.flushTurn("speaker-switch", prevSpeaker);

  // Now switch active speaker
  AppState.activeSpeaker = newSpeaker;
  saveState();
  reflectSpeakerUI();
  break;
}


    /** RECOGNIZE (constant button in header) */
    case "recognize-person": {
  if (!FACE_PRESENT_NOW) { alert("No face detected to recognize."); return; }
  const vec = await getCurrentFaceDescriptor();
  if (!vec) { alert("Couldn’t read a clean face. Hold still."); return; }

  const res = await fetch("/glasses/api/face/recognize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vector: vec, provider: "local", threshold: 0.58 }),
  });
  const data = await res.json();

  if (data.ok && data.match) {
    const pid = data.person.id;
    AppState.recognizedPersonId = pid;
    AppState.activePersonId = pid;
    saveState();
    openProfile(pid);
    } else {
    try {
      const r2 = await fetch("/glasses/api/unknown/ensure", { method: "POST" });
      if (!r2.ok) throw new Error(`HTTP ${r2.status}`);
      const unk = await r2.json();

      AppState.recognizedPersonId = unk.id;
      AppState.activePersonId = unk.id;
      saveState();
      openProfile(unk.id);
    } catch (e) {
      alert("I don’t recognize this face and couldn’t prepare an Unknown profile.");
      console.error(e);
    }
  }

  break;
}


        /** ENROLL (temporary button in profile) */
    case "enroll-current-face": {
      const personId = Number(btn.getAttribute("data-person-id"));
      if (!FACE_PRESENT_NOW) {
        alert("No face detected on camera.");
        return;
      }
      const vec = await getCurrentFaceDescriptor();
      if (!vec) {
        alert("Couldn’t read a clean face descriptor. Try again.");
        return;
      }

      const res = await fetch("/glasses/api/face/enroll", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ person_id: personId, vector: vec, provider: "local" }),
      });
      const data = await res.json();
      if (data.ok) {
        alert("Face enrolled ✅");
      } else {
        alert("Enroll failed: " + (data.error || "unknown"));
      }
      break;
    }


    default: break;
  }
});

/* ============================
   BOOT
============================ */
document.addEventListener("DOMContentLoaded", async () => {
  loadState();
  // require a fresh Recognize click each session
  AppState.recognizedPersonId = null;
  saveState();

  try {
    await loadFaceModels();
    await loadPeople();
  } catch (e) { warn("BOOT: init failure", e); }

  loadView("home");
  startFaceDetection();
});
