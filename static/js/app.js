/**
 * Academic Stress & Clinical Referral Ecosystem - ChatGPT Interface Logic
 * Features:
 * - ChatGPT Conversation Stream (User reflections + Clinical Triage Assistant turns)
 * - Auto-expanding composer with Enter-to-submit and Shift+Enter for new line
 * - AI4Bharat Indic Voice Pipeline (Audio Recording, Waveform visualizer, Presets)
 * - Encrypted Vault Sessions in Collapsible Sidebar
 * - Interactive CBT Modules (4-7-8 Breathing Pacer, Thought Reframing, 5-4-3-2-1 Grounding)
 * - Priority Clinical Referral Slip Generator & SQLite Vault Inspector
 */

document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

let vaultState = {
    is_initialized: false,
    is_unlocked: false,
    total_reflections: 0
};

let voiceState = {
    mediaRecorder: null,
    audioStream: null,
    audioChunks: [],
    recordedSamples: [],
    scriptProcessor: null,
    currentAudioBase64: null,
    clientTranscript: "",
    isRecording: false,
    recordStartTime: 0,
    timerInterval: null,
    audioContext: null,
    analyser: null,
    animFrameId: null,
    speechRecognizer: null
};

let breathingInterval = null;
let currentBreathingPhase = "idle";
let breathingCycleCount = 0;
let cbtAudioPlaying = false;
let allVaultEntries = [];
let currentAuthUser = null;

function initApp() {
    setupSidebarControls();
    setupComposerControls();
    setupVoicePipelineUI();
    checkUserSession();
    fetchVaultStatus();
    fetchVaultEntries();
    fetchModelMetrics();
}

async function checkUserSession() {
    try {
        const res = await fetch("/api/auth/me", { credentials: "include" });
        if (res.ok) {
            const data = await res.json();
            if (data.authenticated && data.user) {
                currentAuthUser = data.user;
                updateAuthUI(data.user);
                return;
            }
        }
    } catch (e) {
        console.warn("Auth check error:", e);
    }
    // Strictly prevent accessing main chat without authentication
    window.location.href = "/login";
}

function updateAuthUI(user) {
    const loginBtn = document.getElementById("btn-topbar-login");
    const userBadge = document.getElementById("topbar-user-badge");
    const userName = document.getElementById("topbar-user-name");
    const logoutBtn = document.getElementById("btn-topbar-logout");
    const sidebarAvatar = document.getElementById("sidebar-avatar");
    const sidebarName = document.getElementById("sidebar-user-fullname");
    const sidebarDept = document.getElementById("sidebar-user-dept");

    if (user) {
        if (loginBtn) loginBtn.style.display = "none";
        if (userBadge) userBadge.style.display = "flex";
        if (userName) userName.textContent = `${user.full_name} (${user.student_id})`;

        if (sidebarAvatar) {
            const parts = user.full_name.trim().split(" ");
            const initials = parts.length > 1 ? (parts[0][0] + parts[1][0]).toUpperCase() : parts[0].substring(0, 2).toUpperCase();
            sidebarAvatar.textContent = initials || "U";
        }
        if (sidebarName) sidebarName.textContent = `${user.full_name} (${user.student_id})`;
        if (sidebarDept) sidebarDept.textContent = `${user.department} • Central Univ of Karnataka`;

        if (logoutBtn) {
            logoutBtn.onclick = handleLogout;
        }

        // Vault is automatically unlocked for the authenticated session
        vaultState.is_unlocked = true;
        updateVaultUI();
    } else {
        if (loginBtn) loginBtn.style.display = "inline-flex";
        if (userBadge) userBadge.style.display = "none";
        if (sidebarAvatar) sidebarAvatar.textContent = "SM";
        if (sidebarName) sidebarName.textContent = "Guest / Student (Click to Sign In)";
        if (sidebarDept) sidebarDept.textContent = "Guide: Prof. Srikantaiah K C";

        const profileBox = document.getElementById("sidebar-profile-box");
        if (profileBox) {
            profileBox.style.cursor = "pointer";
            profileBox.onclick = () => { window.location.href = "/login"; };
        }
    }
}

async function handleLogout() {
    try {
        await fetch("/api/auth/logout", { method: "POST", credentials: "include" });
    } catch (e) {}
    sessionStorage.removeItem("cuk_vault_passphrase");
    sessionStorage.removeItem("cuk_user_profile");
    window.location.reload();
}

/* ==========================================================================
   Sidebar & Navigation
   ========================================================================== */
function setupSidebarControls() {
    const sidebar = document.getElementById("sidebar");
    const toggleBtn = document.getElementById("btn-toggle-sidebar");
    const collapseBtn = document.getElementById("btn-collapse-sidebar");
    const newChatBtn = document.getElementById("btn-new-chat");

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener("click", () => {
            sidebar.classList.toggle("collapsed");
        });
    }

    if (collapseBtn && sidebar) {
        collapseBtn.addEventListener("click", () => {
            sidebar.classList.add("collapsed");
        });
    }

    if (newChatBtn) {
        newChatBtn.addEventListener("click", startNewChat);
    }

    // Vault Inspector Modal Trigger
    const inspectBtn = document.getElementById("btn-open-vault-inspector");
    if (inspectBtn) {
        inspectBtn.addEventListener("click", openVaultInspectorModal);
    }

    // Vault Unlock Trigger
    const vaultToggleBtn = document.getElementById("btn-vault-toggle");
    if (vaultToggleBtn) {
        vaultToggleBtn.addEventListener("click", openVaultModal);
    }

    // Vault Lock Trigger
    const lockBtn = document.getElementById("btn-lock-vault");
    if (lockBtn) {
        lockBtn.addEventListener("click", handleVaultLock);
    }

    // Modal Close buttons
    document.querySelectorAll(".modal-close-btn").forEach(btn => {
        btn.addEventListener("click", closeAllModals);
    });

    // Vault Auth Form Submit
    const vaultForm = document.getElementById("vault-auth-form");
    if (vaultForm) {
        vaultForm.addEventListener("submit", handleVaultAuthSubmit);
    }
}

function startNewChat() {
    const welcomeHero = document.getElementById("welcome-hero");
    const chatThread = document.getElementById("chat-thread");
    const textInput = document.getElementById("reflection-input");

    if (welcomeHero) welcomeHero.style.display = "flex";
    if (chatThread) {
        chatThread.style.display = "none";
        chatThread.innerHTML = "";
    }
    if (textInput) {
        textInput.value = "";
        textInput.style.height = "auto";
        textInput.focus();
    }
    resetVoiceAudio();

    // Deselect any active history item in sidebar
    document.querySelectorAll(".history-item").forEach(item => item.classList.remove("active"));
}

/* ==========================================================================
   Composer & Input Handling
   ========================================================================== */
function setupComposerControls() {
    const textInput = document.getElementById("reflection-input");
    const composerForm = document.getElementById("chat-composer-form");
    const sendBtn = document.getElementById("btn-send");

    if (!textInput) return;

    // Auto-expanding textarea (ChatGPT style)
    textInput.addEventListener("input", () => {
        textInput.style.height = "auto";
        const newHeight = Math.min(textInput.scrollHeight, 180);
        textInput.style.height = newHeight + "px";

        // Enable / disable send button
        const hasContent = textInput.value.trim().length > 0 || voiceState.currentAudioBase64;
        if (sendBtn) sendBtn.disabled = !hasContent;
    });

    // Enter to send, Shift+Enter for newline
    textInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitCurrentReflection();
        }
    });

    // Form submit
    if (composerForm) {
        composerForm.addEventListener("submit", (e) => {
            e.preventDefault();
            submitCurrentReflection();
        });
    }

    // Prompt Chips from Hero or Presets
    document.querySelectorAll(".prompt-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const promptText = chip.getAttribute("data-prompt");
            if (promptText && textInput) {
                textInput.value = promptText;
                textInput.style.height = "auto";
                textInput.style.height = Math.min(textInput.scrollHeight, 180) + "px";
                submitCurrentReflection();
            }
        });
    });

    // Presets dropdown menu toggle
    const presetsToggleBtn = document.getElementById("btn-toggle-presets");
    const presetsDropdown = document.getElementById("presets-dropdown");
    if (presetsToggleBtn && presetsDropdown) {
        presetsToggleBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            presetsDropdown.style.display = presetsDropdown.style.display === "none" ? "flex" : "none";
        });

        document.addEventListener("click", () => {
            if (presetsDropdown) presetsDropdown.style.display = "none";
        });
    }
}

/* ==========================================================================
   Reflection Submission (Text or Voice)
   ========================================================================== */
async function ensureVaultUnlocked() {
    if (vaultState.is_initialized && !vaultState.is_unlocked) {
        const cached = sessionStorage.getItem("cuk_vault_passphrase");
        if (cached) {
            try {
                const res = await fetch("/api/vault/unlock", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ passphrase: cached })
                });
                if (res.ok) {
                    const statusRes = await fetch("/api/vault/status");
                    if (statusRes.ok) {
                        vaultState = await statusRes.json();
                        updateVaultUI();
                    }
                }
            } catch (e) {
                console.warn("Auto-unlock error:", e);
            }
        }
    }
}

async function submitCurrentReflection() {
    const textInput = document.getElementById("reflection-input");
    const rawText = textInput ? textInput.value.trim() : "";
    const hasAudio = voiceState.currentAudioBase64 !== null;

    if (!rawText && !hasAudio) return;

    // Ensure vault is unlocked if user already entered passphrase in this browser session
    await ensureVaultUnlocked();

    // Check if voice mode or text mode
    if (hasAudio) {
        await submitVoiceReflection(rawText);
    } else {
        await submitTextReflection(rawText);
    }
}

async function parseResponseError(res, defaultMsg = "Operation failed.") {
    try {
        const text = await res.text();
        try {
            const data = JSON.parse(text);
            return data.detail || data.message || defaultMsg;
        } catch (_) {
            return (text && text.trim().length > 0) ? text.trim() : defaultMsg;
        }
    } catch (_) {
        return defaultMsg;
    }
}

async function submitTextReflection(text) {
    const sendBtn = document.getElementById("btn-send");
    const textInput = document.getElementById("reflection-input");

    // 1. Append User Message Bubble with Turn ID
    const userTurnId = "user-turn-" + Date.now();
    appendUserMessageTurn(userTurnId, text, null);

    // Reset textarea
    if (textInput) {
        textInput.value = "";
        textInput.style.height = "auto";
    }
    if (sendBtn) sendBtn.disabled = true;

    // 2. Append Assistant Thinking / Loading Turn
    const assistantTurnId = "turn-" + Date.now();
    appendLoadingTurn(assistantTurnId);

    try {
        const res = await fetch("/api/analyze", {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                text: text,
                save_to_vault: true
            })
        });

        if (!res.ok) {
            const errorMsg = await parseResponseError(res, "Screening analysis failed.");
            throw new Error(errorMsg);
        }

        const data = await res.json();

        // If text was in Indic dialect (Hindi/Kannada etc.), display translation in user bubble
        if (data.is_translated && data.translated_text) {
            updateUserTurnWithTextTranslation(userTurnId, data.original_text || text, data.translated_text, true);
        }

        // 3. Render Assistant Response
        renderAssistantResponse(assistantTurnId, data);

        // 4. Update Vault & Sidebar
        await fetchVaultEntries();
        await fetchVaultStatus();

    } catch (err) {
        replaceTurnWithError(assistantTurnId, err.message);
    }
}

async function submitVoiceReflection(optionalCaption) {
    const sendBtn = document.getElementById("btn-send");
    const textInput = document.getElementById("reflection-input");
    const langSelect = document.getElementById("voice-language-select");
    const lang = langSelect ? langSelect.value : "en-IN";

    // 1. Determine initial spoken transcript text (from live recognition, preset, or caption)
    const initialTranscript = voiceState.clientTranscript || optionalCaption || (textInput ? textInput.value.trim() : "");
    const audioLabel = initialTranscript || `[Voice Reflection - ${lang}]`;
    const userTurnId = "user-turn-" + Date.now();

    appendUserMessageTurn(userTurnId, audioLabel, {
        audioSrc: document.getElementById("voice-audio-element")?.src || null,
        language: lang,
        transcription: initialTranscript
    });

    const audioBase64 = voiceState.currentAudioBase64;
    const clientTranscript = initialTranscript;
    resetVoiceAudio();

    if (textInput) {
        textInput.value = "";
        textInput.style.height = "auto";
    }
    if (sendBtn) sendBtn.disabled = true;

    // 2. Append Assistant Thinking Turn
    const assistantTurnId = "turn-" + Date.now();
    appendLoadingTurn(assistantTurnId, "Transcribing with AI4Bharat Indic ASR & Translating...");

    try {
        const res = await fetch("/api/voice/analyze", {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                audio_base64: audioBase64,
                language: lang,
                model_choice: "auto",
                save_to_vault: true,
                text_transcript: clientTranscript || null
            })
        });

        if (!res.ok) {
            const errorMsg = await parseResponseError(res, "Voice screening analysis failed.");
            throw new Error(errorMsg);
        }

        const data = await res.json();

        // 3. Immediately update the User Bubble with the exact query text recognized & translated by server!
        const finalTranscription = data.transcription || clientTranscript || "Spoken Reflection";
        updateUserTurnWithTranscription(
            userTurnId,
            finalTranscription,
            data.translated_text || null,
            data.is_translated || false,
            lang
        );

        // 4. Render Assistant Response (includes explicit recognized query card + translation)
        renderAssistantResponse(assistantTurnId, data);

        // 5. Update Vault & Sidebar
        await fetchVaultEntries();
        await fetchVaultStatus();

    } catch (err) {
        replaceTurnWithError(assistantTurnId, err.message);
    }
}

/* ==========================================================================
   Chat Stream Rendering (ChatGPT Style)
   ========================================================================== */
function appendUserMessageTurn(turnId, text, voiceMeta) {
    const welcomeHero = document.getElementById("welcome-hero");
    const chatThread = document.getElementById("chat-thread");
    const viewport = document.getElementById("chat-viewport");

    if (welcomeHero) welcomeHero.style.display = "none";
    if (chatThread) chatThread.style.display = "flex";

    let voiceBadgeHtml = "";
    if (voiceMeta) {
        voiceBadgeHtml = `
            <div class="voice-note-pill">
                <span>🎙️ AI4Bharat Indic Voice (${escapeHtml(voiceMeta.language)})</span>
                ${voiceMeta.audioSrc ? `<audio src="${voiceMeta.audioSrc}" controls style="height: 26px; vertical-align: middle; margin-left: 0.5rem;"></audio>` : ""}
            </div>
        `;
    }

    let messageBodyHtml = "";
    if (voiceMeta && voiceMeta.transcription) {
        messageBodyHtml = `
            <div class="user-spoken-card">
                <span class="user-spoken-label">🗣️ Spoken Reflection (Query Input):</span>
                <span class="user-spoken-text">"${escapeHtml(voiceMeta.transcription)}"</span>
            </div>
        `;
    } else {
        messageBodyHtml = `<div class="user-query-text">${escapeHtml(text)}</div>`;
    }

    const userTurnHtml = `
        <div class="message-turn user-turn" id="${turnId}">
            <div class="user-message-bubble">
                ${voiceBadgeHtml}
                <div class="user-message-body-wrap">
                    ${messageBodyHtml}
                </div>
            </div>
        </div>
    `;

    if (chatThread) {
        chatThread.insertAdjacentHTML("beforeend", userTurnHtml);
    }
    scrollToBottom();
}

function updateUserTurnWithTranscription(turnId, originalText, translatedText, isTranslated, language) {
    const turnEl = document.getElementById(turnId);
    if (!turnEl) return;
    const bodyWrap = turnEl.querySelector(".user-message-body-wrap");
    if (!bodyWrap) return;

    let translationHtml = "";
    if (isTranslated && translatedText && translatedText.toLowerCase() !== originalText.toLowerCase()) {
        translationHtml = `
            <div class="user-translated-card">
                <span class="user-translated-label">🌐 Evaluated English Translation:</span>
                <span class="user-translated-text">"${escapeHtml(translatedText)}"</span>
            </div>
        `;
    }

    bodyWrap.innerHTML = `
        <div class="user-spoken-card">
            <span class="user-spoken-label">🗣️ Spoken Reflection (Query Input):</span>
            <span class="user-spoken-text">"${escapeHtml(originalText)}"</span>
            ${translationHtml}
        </div>
    `;
    scrollToBottom();
}

function updateUserTurnWithTextTranslation(turnId, originalText, translatedText, isTranslated) {
    const turnEl = document.getElementById(turnId);
    if (!turnEl) return;
    const bodyWrap = turnEl.querySelector(".user-message-body-wrap");
    if (!bodyWrap) return;

    if (isTranslated && translatedText && translatedText.toLowerCase() !== originalText.toLowerCase()) {
        bodyWrap.innerHTML = `
            <div class="user-query-text">${escapeHtml(originalText)}</div>
            <div class="user-translated-card">
                <span class="user-translated-label">🌐 Evaluated English Translation:</span>
                <span class="user-translated-text">"${escapeHtml(translatedText)}"</span>
            </div>
        `;
        scrollToBottom();
    }
}

function appendLoadingTurn(turnId, customMsg = "Screening on-device with DistilBERT...") {
    const chatThread = document.getElementById("chat-thread");
    if (!chatThread) return;

    const loadingHtml = `
        <div class="message-turn assistant-turn" id="${turnId}">
            <div class="assistant-avatar">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <path d="M9 12l2 2 4-4"/>
                </svg>
            </div>
            <div class="assistant-message-content">
                <div style="display: flex; align-items: center; gap: 0.65rem; color: var(--text-muted); font-size: 0.88rem;">
                    <svg class="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                        <path d="M12 2a10 10 0 0 1 10 10"></path>
                    </svg>
                    <span>${customMsg}</span>
                </div>
            </div>
        </div>
    `;

    chatThread.insertAdjacentHTML("beforeend", loadingHtml);
    scrollToBottom();
}

function replaceTurnWithError(turnId, errorMsg) {
    const turnEl = document.getElementById(turnId);
    if (!turnEl) return;

    turnEl.querySelector(".assistant-message-content").innerHTML = `
        <div style="background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.35); padding: 0.85rem; border-radius: var(--radius-sm); color: #fb7185; font-size: 0.85rem;">
            ⚠️ <strong>Error during on-device screening:</strong> ${escapeHtml(errorMsg)}
        </div>
    `;
    scrollToBottom();
}

function renderAssistantResponse(turnId, data) {
    const turnEl = document.getElementById(turnId);
    if (!turnEl) return;

    const classification = data.classification;
    const triage = data.triage;
    const referral = data.referral_token;
    const acoustic = data.acoustic_biomarkers || null;
    const tier = (data.fused_tier || classification.predicted_tier || "Medium").toLowerCase();

    // 0. Query Input Recognized Card (Always displays user's spoken/text query + English translation)
    let transcriptCardHtml = "";
    const rawQuery = data.transcription || data.original_text || null;
    const translatedQuery = data.translated_text || null;
    const isTranslated = data.is_translated || false;

    if (rawQuery && rawQuery.trim().length > 0) {
        transcriptCardHtml = `
            <div class="spoken-transcript-card">
                <div class="spoken-transcript-header">
                    <div class="spoken-transcript-badge">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                            <line x1="12" y1="19" x2="12" y2="23"/>
                            <line x1="8" y1="23" x2="16" y2="23"/>
                        </svg>
                        <span>Query Input Recognized</span>
                    </div>
                    <span class="spoken-transcript-meta">${escapeHtml(data.language || data.source_language || 'en-IN')} • ${escapeHtml(data.model_used || 'AI4Bharat Indic Ingestion')}</span>
                </div>
                <div class="spoken-transcript-quote">
                    “${escapeHtml(rawQuery)}”
                </div>
                ${isTranslated && translatedQuery && translatedQuery.toLowerCase() !== rawQuery.toLowerCase() ? `
                    <div class="spoken-transcript-translation">
                        <span class="trans-label">🌐 Evaluated English Translation:</span>
                        <span class="trans-text">“${escapeHtml(translatedQuery)}”</span>
                    </div>
                ` : ""}
            </div>
        `;
    }

    // 0.5 Empathetic Dialogue Reply (Native Language + English Toggle)
    let conversationalReplyHtml = "";
    if (data.conversational_reply) {
        const reply = data.conversational_reply;
        const nativeText = reply.native_text || "";
        const englishText = reply.english_text || "";
        const isTrans = reply.is_translated || false;
        const targetLang = reply.target_language || data.language || "hi-IN";
        const replyId = "reply-" + turnId;

        conversationalReplyHtml = `
            <div class="empathetic-reply-card">
                <div class="empathetic-reply-header">
                    <div class="empathetic-reply-badge">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                        </svg>
                        <span>Empathetic Dialogue • ${escapeHtml(reply.model_name || 'Clinical Assistant')}</span>
                    </div>
                    <div class="empathetic-reply-actions">
                        <button class="btn-tts" data-speech="${escapeHtml(nativeText)}" data-lang="${escapeHtml(targetLang)}" onclick="handleSpeakBtn(this)" title="Listen to spoken response">
                            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                                <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
                            </svg>
                            <span>Listen</span>
                        </button>
                        ${isTrans ? `
                            <button class="btn-toggle-lang" onclick="toggleLanguageView('${replyId}')" id="btn-toggle-${replyId}">
                                🌐 View English
                            </button>
                        ` : ""}
                    </div>
                </div>
                <div class="empathetic-reply-body" id="text-native-${replyId}">
                    ${escapeHtml(nativeText)}
                </div>
                ${isTrans ? `
                    <div class="empathetic-reply-english" id="text-english-${replyId}" style="display: none;">
                        <div class="english-badge-label">Original English Prompt Generation:</div>
                        <div class="english-quote">“${escapeHtml(englishText)}”</div>
                    </div>
                ` : ""}
            </div>
        `;
    }

    // 1. Triage Alert Banner
    const tierIcons = {
        low: "🟢",
        medium: "🟡",
        high: "🔴"
    };

    const displayTier = data.fused_tier || triage.tier || classification.predicted_tier || "Medium";

    let alertHtml = `
        <div class="triage-alert-banner tier-${tier}">
            <div class="triage-alert-icon">${tierIcons[tier] || "🟡"}</div>
            <div class="triage-alert-body">
                <div class="triage-tier-headline">${displayTier} Stress Tier: ${triage.tier_label}</div>
                <div class="triage-urgency-tag">${classification.clinical_urgency || triage.urgency}</div>
                <div class="triage-summary-text">${escapeHtml(triage.summary)}</div>
                
                <div class="stressor-chips-wrap">
                    ${classification.primary_domain ? `<span class="stressor-chip" style="background: rgba(56, 189, 248, 0.15); border-color: rgba(56, 189, 248, 0.4); color: #7dd3fc;">🎯 ${escapeHtml(classification.primary_domain)}</span>` : ""}
                    ${(classification.stressors || []).map(s => `<span class="stressor-chip">📌 ${escapeHtml(s)}</span>`).join("")}
                    ${(classification.clinical_markers || []).map(m => `<span class="stressor-chip" style="background: rgba(244, 63, 94, 0.12); border-color: rgba(244, 63, 94, 0.3); color: #fda4af;">🩺 ${escapeHtml(m)}</span>`).join("")}
                </div>
            </div>
        </div>
    `;

    // 2. Softmax Radar Bars
    const probs = classification.probabilities || { Low: 0.05, Medium: 0.90, High: 0.05 };
    const lowPct = Math.round(probs.Low * 100);
    const medPct = Math.round(probs.Medium * 100);
    const highPct = Math.round(probs.High * 100);

    let radarHtml = `
        <div class="probability-radar-box">
            <div class="radar-title">Transformer Softmax Distribution</div>
            <div class="radar-bars-grid">
                <div class="radar-bar-row">
                    <span class="radar-label" style="color: #34d399;">Low Stress</span>
                    <div class="radar-track"><div class="radar-fill low" style="width: ${lowPct}%;"></div></div>
                    <span class="radar-val" style="color: #34d399;">${lowPct}%</span>
                </div>
                <div class="radar-bar-row">
                    <span class="radar-label" style="color: #fbbf24;">Medium Stress</span>
                    <div class="radar-track"><div class="radar-fill med" style="width: ${medPct}%;"></div></div>
                    <span class="radar-val" style="color: #fbbf24;">${medPct}%</span>
                </div>
                <div class="radar-bar-row">
                    <span class="radar-label" style="color: #fb7185;">High / Crisis</span>
                    <div class="radar-track"><div class="radar-fill high" style="width: ${highPct}%;"></div></div>
                    <span class="radar-val" style="color: #fb7185;">${highPct}%</span>
                </div>
            </div>
            ${classification.entropy !== undefined ? `
                <div style="margin-top: 0.65rem; padding-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.08); font-size: 0.72rem; color: var(--text-muted); display: flex; justify-content: space-between;">
                    <span>📐 Shannon Entropy: <strong>${classification.entropy} bits</strong></span>
                    <span>🔬 Telemetry: <strong>${escapeHtml(classification.ambiguity_status || 'Calibrated')}</strong></span>
                </div>
            ` : ""}
        </div>
    `;

    // 4. Interventions (CBT 4-7-8 Breathing Pacer or Priority Referral Slip)
    let interventionsHtml = "";
    if (triage.requires_clinical_escalation && referral) {
        interventionsHtml = `
            <div class="referral-slip-card">
                <div class="referral-header">
                    <div>
                        <div class="referral-case-id">${referral.case_id}</div>
                        <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.15rem;">
                            Verified Sealed Referral Pass • ${referral.issued_at}
                        </div>
                    </div>
                    <span class="referral-badge-alert">PRIORITY CRISIS ESCALATION</span>
                </div>

                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 0.75rem; margin-bottom: 0.75rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 0.78rem; font-weight: 600; color: #94a3b8;">🔑 Confidential Consultation Passcode:</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 800; color: #38bdf8; background: rgba(56, 189, 248, 0.15); padding: 0.2rem 0.5rem; border-radius: 4px;">${escapeHtml(referral.passcode || 'Present Account Password')}</span>
                    </div>
                    <p style="font-size: 0.72rem; color: #cbd5e1; margin: 0.4rem 0 0 0; line-height: 1.4;">
                        Share this passcode in-person with the campus counselor during your appointment to decrypt your reflection diary for clinical evaluation.
                    </p>
                </div>

                <p style="font-size: 0.85rem; margin-bottom: 0.65rem;">
                    <strong>Target Office:</strong> ${referral.target_office}<br>
                    <strong>Campus Location:</strong> ${referral.campus_location}
                </p>

                <div style="font-size: 0.78rem; background: rgba(0,0,0,0.3); padding: 0.65rem; border-radius: 6px; border-left: 3px solid #f43f5e; margin-bottom: 0.75rem;">
                    🛡️ <strong>Zero Diary Disclosure Guarantee:</strong> ${referral.privacy_attestation}
                </div>

                <div class="helpline-box">
                    <div style="font-size: 0.72rem; text-transform: uppercase; color: #fb7185; font-weight: 700; margin-bottom: 0.4rem;">
                        24x7 Emergency Helplines
                    </div>
                    <div class="helpline-item">
                        <span>Tele-MANAS (Govt of India Helpline)</span>
                        <a href="tel:14416" class="helpline-phone">📞 14416</a>
                    </div>
                    <div class="helpline-item">
                        <span>KIRAN Mental Health Toll-Free</span>
                        <a href="tel:18005990019" class="helpline-phone">📞 1800-599-0019</a>
                    </div>
                    <div class="helpline-item">
                        <span>Central University of Karnataka Health Centre</span>
                        <span style="font-family: var(--font-mono); font-size: 0.78rem; color: #cbd5e1;">Admin Block</span>
                    </div>
                </div>

                <button type="button" class="btn-print-slip" onclick="window.print()">
                    🖨️ Print Confidential Referral Pass
                </button>
            </div>
        `;
    } else if (triage.cbt_exercises && triage.cbt_exercises.length > 0) {
        // Filter out cluttered secondary modules; keep the clean, primary 4-7-8 Breathing Pacer
        const activeExercises = triage.cbt_exercises.filter(ex => ex.id === "breathing_478");
        if (activeExercises.length > 0) {
            interventionsHtml = `
                <div class="cbt-modules-container">
                    ${activeExercises.map(ex => `
                        <div class="cbt-card">
                            <div class="cbt-card-top">
                                <span class="cbt-card-title">${escapeHtml(ex.title)}</span>
                                <span class="cbt-category">${escapeHtml(ex.category)}</span>
                            </div>
                            <p class="cbt-card-desc">${escapeHtml(ex.description)}</p>
                            <div class="breathing-box">
                                <div class="breathing-circle-wrap">
                                    <div id="breath-circle" class="breath-circle">
                                        <span id="breath-instruction" class="breath-instruction">READY</span>
                                        <span id="breath-timer" class="breath-timer">4s</span>
                                    </div>
                                </div>
                                <div id="breath-cycle-label" style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.75rem;">
                                    Completed Cycles: 0 / 4
                                </div>
                                <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; justify-content: center;">
                                    <button id="btn-toggle-breathing" class="btn btn-primary" onclick="toggleBreathingPacer()">
                                        Start 4-7-8 Breathing
                                    </button>
                                    <button type="button" class="btn-cbt-audio" onclick="toggleCBTVoiceAudio('breathing_478')">
                                        <span id="cbt-audio-icon">🔊</span>
                                        <span id="cbt-audio-label">AI4Bharat Voice Guide & 432Hz Pacer</span>
                                    </button>
                                </div>
                                <audio id="cbt-harmonic-player" src="/api/voice/guidance/breathing_478/audio" style="display:none;" preload="none"></audio>
                            </div>
                        </div>
                    `).join("")}
                </div>
            `;
        }
    }

    // 5. Metadata Footer
    const latencyMs = data.total_latency_ms || classification.latency_ms || 2.5;
    let vaultFooterHtml = "";
    if (data.vault && data.vault.saved) {
        vaultFooterHtml = `<span>🔒 AES-256-GCM Encrypted in Local SQLite Vault (Session #${data.vault.entry_id})</span>`;
    } else if (vaultState && !vaultState.is_unlocked) {
        vaultFooterHtml = `<span class="vault-locked-hint" style="color: #f59e0b; cursor: pointer;" onclick="openVaultModal()" title="Click to unlock vault">⚠️ Ephemeral (Vault Locked — Click to Unlock &amp; Persist)</span>`;
    } else {
        vaultFooterHtml = `<span>🛡️ Zero Plaintext Ephemeral Analysis</span>`;
    }

    const metaFooterHtml = `
        <div class="assistant-meta-footer">
            <span>⚡ CPU Latency: ${latencyMs} ms (&lt;300ms SLA)</span>
            <span>•</span>
            ${vaultFooterHtml}
        </div>
    `;

    // Render into turn container (clean stream: transcript + empathetic dialogue + triage + softmax + 4-7-8 pacer)
    turnEl.querySelector(".assistant-message-content").innerHTML = `
        ${transcriptCardHtml}
        ${conversationalReplyHtml}
        ${alertHtml}
        ${radarHtml}
        ${interventionsHtml}
        ${metaFooterHtml}
    `;

    scrollToBottom();
}

function scrollToBottom() {
    const viewport = document.getElementById("chat-viewport");
    if (viewport) {
        viewport.scrollTo({ top: viewport.scrollHeight, behavior: "smooth" });
    }
}

/* ==========================================================================
   Interactive CBT 4-7-8 Breathing Pacer Logic
   ========================================================================== */
function toggleBreathingPacer() {
    const btn = document.getElementById("btn-toggle-breathing");
    if (!btn) return;

    if (breathingInterval) {
        clearInterval(breathingInterval);
        breathingInterval = null;
        currentBreathingPhase = "idle";
        breathingCycleCount = 0;
        btn.textContent = "Start 4-7-8 Breathing";
        btn.classList.remove("btn-danger");
        btn.classList.add("btn-primary");
        resetBreathingVisuals();
    } else {
        breathingCycleCount = 0;
        btn.textContent = "Stop Breathing Session";
        btn.classList.remove("btn-primary");
        btn.classList.add("btn-danger");
        runBreathingSequence();
    }
}

function resetBreathingVisuals() {
    const circle = document.getElementById("breath-circle");
    const inst = document.getElementById("breath-instruction");
    const timer = document.getElementById("breath-timer");
    const label = document.getElementById("breath-cycle-label");

    if (circle) circle.className = "breath-circle";
    if (inst) inst.textContent = "READY";
    if (timer) timer.textContent = "4s";
    if (label) label.textContent = `Completed Cycles: ${breathingCycleCount} / 4`;
}

function runBreathingSequence() {
    const circle = document.getElementById("breath-circle");
    const inst = document.getElementById("breath-instruction");
    const timer = document.getElementById("breath-timer");
    const label = document.getElementById("breath-cycle-label");

    let phase = "inhale";
    let timeLeft = 4;

    function tick() {
        if (!circle) return;

        if (phase === "inhale") {
            circle.className = "breath-circle inhale";
            inst.textContent = "INHALE";
            timer.textContent = `${timeLeft}s`;
        } else if (phase === "hold") {
            circle.className = "breath-circle hold";
            inst.textContent = "HOLD";
            timer.textContent = `${timeLeft}s`;
        } else if (phase === "exhale") {
            circle.className = "breath-circle exhale";
            inst.textContent = "EXHALE";
            timer.textContent = `${timeLeft}s`;
        }

        timeLeft--;

        if (timeLeft < 0) {
            if (phase === "inhale") {
                phase = "hold";
                timeLeft = 7;
            } else if (phase === "hold") {
                phase = "exhale";
                timeLeft = 8;
            } else if (phase === "exhale") {
                breathingCycleCount++;
                if (label) label.textContent = `Completed Cycles: ${breathingCycleCount} / 4`;
                if (breathingCycleCount >= 4) {
                    toggleBreathingPacer();
                    alert("Well done! 4 cycles of parasympathetic 4-7-8 breathing completed.");
                    return;
                }
                phase = "inhale";
                timeLeft = 4;
            }
        }
    }

    tick();
    breathingInterval = setInterval(tick, 1000);
}

function toggleCBTVoiceAudio(exerciseId) {
    const player = document.getElementById("cbt-harmonic-player");
    const label = document.getElementById("cbt-audio-label");
    const icon = document.getElementById("cbt-audio-icon");

    if (!player) return;

    if (!cbtAudioPlaying) {
        player.play().then(() => {
            cbtAudioPlaying = true;
            if (label) label.textContent = "Pause 432Hz Harmonic Pacer";
            if (icon) icon.textContent = "⏸️";
        }).catch(err => console.warn(err));
    } else {
        player.pause();
        cbtAudioPlaying = false;
        if (label) label.textContent = "AI4Bharat Voice Guide & 432Hz Pacer";
        if (icon) icon.textContent = "🔊";
    }

    player.onended = () => {
        cbtAudioPlaying = false;
        if (label) label.textContent = "AI4Bharat Voice Guide & 432Hz Pacer";
        if (icon) icon.textContent = "🔊";
    };
}

/* ==========================================================================
   AI4Bharat Voice Pipeline (Audio Recording & Presets)
   ========================================================================== */
function setupVoicePipelineUI() {
    const recordBtn = document.getElementById("btn-voice-record");
    const cancelBtn = document.getElementById("btn-cancel-voice");
    const fileInput = document.getElementById("voice-file-input");
    const removeAudioBtn = document.getElementById("btn-remove-audio");

    if (recordBtn) {
        recordBtn.addEventListener("click", toggleVoiceRecording);
    }
    if (cancelBtn) {
        cancelBtn.addEventListener("click", resetVoiceAudio);
    }
    if (fileInput) {
        fileInput.addEventListener("change", handleVoiceFileUpload);
    }
    if (removeAudioBtn) {
        removeAudioBtn.addEventListener("click", resetVoiceAudio);
    }

    // Preset chips in dropdown
    document.querySelectorAll(".voice-preset-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const scenario = chip.getAttribute("data-scenario");
            const lang = chip.getAttribute("data-lang") || "en-IN";
            loadVoiceScenarioPreset(scenario, lang);
            const dropdown = document.getElementById("presets-dropdown");
            if (dropdown) dropdown.style.display = "none";
        });
    });
}

async function toggleVoiceRecording() {
    if (!voiceState.isRecording) {
        await startVoiceRecording();
    } else {
        stopVoiceRecording();
    }
}

const PRESET_SCENARIOS = {
    viva_panic: {
        text: "I have three lab exams and two project viva rounds back to back tomorrow. I haven't completed my records and my backlogs are piling up. I am panicking and cannot breathe.",
        name: "Viva & Backlog Panic",
        baseFreq: 265,
        jitterAmp: 0.12,
        wpmCadence: 195
    },
    placement_anxiety: {
        text: "Campus placement aptitude results came out today and I got rejected again. All my batchmates are getting placed while I have arrears to clear. Feeling extremely anxious and stressed.",
        name: "Placement Rejection Stress",
        baseFreq: 240,
        jitterAmp: 0.08,
        wpmCadence: 185
    },
    hindi_viva: {
        text: "कल कंपाइलर डिज़ाइन का लैब वाइवा और इंटरनल एग्जाम है। मेरा रिकॉर्ड भी अधूरा है और बैकलाग की बहुत टेंशन हो रही है, रात को नींद नहीं आ रही।",
        name: "वाइवा और एग्जाम टेंशन",
        baseFreq: 250,
        jitterAmp: 0.10,
        wpmCadence: 180
    },
    kannada_viva: {
        text: "ನಾಳೆ ಸೆಮಿಸ್ಟರ್ ಲ್ಯಾಬ್ ವೈವಾ ಮತ್ತು ಪ್ರಾಜೆಕ್ಟ್ ರಿವ್ಯೂ ಇದೆ. ರೆಕಾರ್ಡ್ಸ್ ಇನ್ನೂ ಪೂರ್ಣವಾಗಿಲ್ಲ, ತುಂಬಾ ಆತಂಕ ಮತ್ತು ಒತ್ತಡವಾಗುತ್ತಿದೆ.",
        name: "ಲ್ಯಾಬ್ ಪರೀಕ್ಷೆಯ ಆತಂಕ (CUK)",
        baseFreq: 255,
        jitterAmp: 0.11,
        wpmCadence: 175
    },
    project_milestone: {
        text: "Finished reviewing the seminar presentation with my project guide. The feedback was constructive and I am feeling on track for the final submission.",
        name: "Milestone Review Approved",
        baseFreq: 145,
        jitterAmp: 0.01,
        wpmCadence: 120
    }
};

async function startVoiceRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });

        voiceState.audioStream = stream;
        voiceState.audioChunks = [];
        voiceState.recordedSamples = [];
        voiceState.clientTranscript = "";
        voiceState.isRecording = true;
        voiceState.recordStartTime = Date.now();

        // 1. AudioContext & Analyser for Waveform
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        voiceState.audioContext = new AudioContextClass();
        const source = voiceState.audioContext.createMediaStreamSource(stream);
        voiceState.analyser = voiceState.audioContext.createAnalyser();
        voiceState.analyser.fftSize = 256;
        source.connect(voiceState.analyser);

        // 2. Direct PCM Sample Collector for Pristine 16kHz WAV Generation
        const bufferSize = 4096;
        const scriptProcessor = voiceState.audioContext.createScriptProcessor(bufferSize, 1, 1);
        scriptProcessor.onaudioprocess = (e) => {
            if (!voiceState.isRecording) return;
            const channelData = e.inputBuffer.getChannelData(0);
            voiceState.recordedSamples.push(new Float32Array(channelData));
        };
        source.connect(scriptProcessor);
        scriptProcessor.connect(voiceState.audioContext.destination);
        voiceState.scriptProcessor = scriptProcessor;

        // 3. Fallback MediaRecorder for browser safety
        try {
            let mimeType = "audio/webm;codecs=opus";
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = MediaRecorder.isTypeSupported("audio/ogg") ? "audio/ogg" : "";
            }
            voiceState.mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
            voiceState.mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) voiceState.audioChunks.push(e.data);
            };
            voiceState.mediaRecorder.start(100);
        } catch (mrErr) {
            console.warn("MediaRecorder fallback notice:", mrErr);
        }

        // 4. Live Real-Time Speech Recognition (Web Speech API)
        const SpeechRecognitionClass = window.SpeechRecognition || window.webkitSpeechRecognition;
        const liveStrip = document.getElementById("voice-live-transcript-strip");
        const liveText = document.getElementById("voice-live-text");
        if (liveStrip) liveStrip.style.display = "flex";
        if (liveText) liveText.textContent = "Listening... Speak into your microphone";

        if (SpeechRecognitionClass) {
            try {
                const recognizer = new SpeechRecognitionClass();
                recognizer.continuous = true;
                recognizer.interimResults = true;
                const langSelect = document.getElementById("voice-language-select");
                recognizer.lang = langSelect ? langSelect.value : "en-IN";

                recognizer.onresult = (event) => {
                    let interim = "";
                    let final = "";
                    for (let i = 0; i < event.results.length; ++i) {
                        if (event.results[i].isFinal) {
                            final += event.results[i][0].transcript + " ";
                        } else {
                            interim += event.results[i][0].transcript;
                        }
                    }
                    const fullText = (final + interim).trim();
                    if (fullText) {
                        voiceState.clientTranscript = fullText;
                        if (liveText) liveText.textContent = fullText;
                    }
                };

                recognizer.onerror = (e) => {
                    console.log("Web Speech API notice:", e.error);
                };

                recognizer.start();
                voiceState.speechRecognizer = recognizer;
            } catch (srErr) {
                console.warn("Speech recognition initialization skipped:", srErr);
            }
        }

        // 5. UI Updates & Waveform Canvas
        const recordBtn = document.getElementById("btn-voice-record");
        const recordingHud = document.getElementById("voice-recording-hud");
        const timerEl = document.getElementById("voice-timer");

        if (recordBtn) recordBtn.classList.add("recording");
        if (recordingHud) recordingHud.style.display = "flex";

        voiceState.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - voiceState.recordStartTime) / 1000);
            const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
            const secs = String(elapsed % 60).padStart(2, '0');
            if (timerEl) timerEl.textContent = `${mins}:${secs} / 01:00`;
            if (elapsed >= 60) stopVoiceRecording();
        }, 500);

        drawLiveWaveform();

    } catch (err) {
        console.warn("Microphone access error:", err);
        alert("Microphone Access Error: Could not access recording device. You can test using the simulated Voice Presets or upload an audio file.");
    }
}

function stopVoiceRecording() {
    if (!voiceState.isRecording) return;
    voiceState.isRecording = false;

    if (voiceState.timerInterval) {
        clearInterval(voiceState.timerInterval);
        voiceState.timerInterval = null;
    }

    if (voiceState.speechRecognizer) {
        try {
            voiceState.speechRecognizer.stop();
        } catch (e) {}
        voiceState.speechRecognizer = null;
    }

    if (voiceState.scriptProcessor) {
        try {
            voiceState.scriptProcessor.disconnect();
        } catch (e) {}
        voiceState.scriptProcessor = null;
    }

    if (voiceState.mediaRecorder && voiceState.mediaRecorder.state !== "inactive") {
        try {
            voiceState.mediaRecorder.stop();
        } catch (e) {}
    }

    if (voiceState.audioStream) {
        voiceState.audioStream.getTracks().forEach(track => track.stop());
        voiceState.audioStream = null;
    }

    if (voiceState.animFrameId) {
        cancelAnimationFrame(voiceState.animFrameId);
        voiceState.animFrameId = null;
    }

    const recordBtn = document.getElementById("btn-voice-record");
    const recordingHud = document.getElementById("voice-recording-hud");
    const liveStrip = document.getElementById("voice-live-transcript-strip");

    if (recordBtn) recordBtn.classList.remove("recording");
    if (recordingHud) recordingHud.style.display = "none";
    if (liveStrip) liveStrip.style.display = "none";

    // Trigger audio compilation & UI update
    handleRecordingComplete();
}

function resampleFloat32Array(samples, srcSr, dstSr) {
    if (srcSr === dstSr || !srcSr || !dstSr || samples.length === 0) return samples;
    const ratio = srcSr / dstSr;
    const targetLength = Math.round(samples.length / ratio);
    const result = new Float32Array(targetLength);
    for (let i = 0; i < targetLength; i++) {
        const origIndex = i * ratio;
        const indexFloor = Math.floor(origIndex);
        const indexCeil = Math.min(indexFloor + 1, samples.length - 1);
        const frac = origIndex - indexFloor;
        result[i] = samples[indexFloor] * (1 - frac) + samples[indexCeil] * frac;
    }
    return result;
}

function handleRecordingComplete() {
    let wavBlob = null;

    // A: Compile from raw PCM float32 samples into standard 16kHz 16-bit Mono WAV
    if (voiceState.recordedSamples && voiceState.recordedSamples.length > 0) {
        let totalLen = 0;
        for (let i = 0; i < voiceState.recordedSamples.length; i++) {
            totalLen += voiceState.recordedSamples[i].length;
        }
        const merged = new Float32Array(totalLen);
        let offset = 0;
        for (let i = 0; i < voiceState.recordedSamples.length; i++) {
            merged.set(voiceState.recordedSamples[i], offset);
            offset += voiceState.recordedSamples[i].length;
        }

        const srcSr = voiceState.audioContext ? voiceState.audioContext.sampleRate : 44100;
        const resampled = resampleFloat32Array(merged, srcSr, 16000);
        wavBlob = encodeWAV16Bit(resampled, 16000);
    } else if (voiceState.audioChunks && voiceState.audioChunks.length > 0) {
        wavBlob = new Blob(voiceState.audioChunks, { type: voiceState.mediaRecorder?.mimeType || "audio/webm" });
    }

    if (!wavBlob) return;

    const audioUrl = URL.createObjectURL(wavBlob);
    const audioElement = document.getElementById("voice-audio-element");
    const attachedTag = document.getElementById("audio-attached-tag");
    const attachedName = document.getElementById("audio-attached-name");
    const attachedTranscript = document.getElementById("audio-attached-transcript");
    const attachedTranscriptContent = document.getElementById("attached-transcript-content");
    const sendBtn = document.getElementById("btn-send");

    if (audioElement) audioElement.src = audioUrl;
    if (attachedTag) attachedTag.style.display = "flex";
    if (attachedName) attachedName.textContent = "microphone_reflection.wav";

    if (voiceState.clientTranscript && voiceState.clientTranscript.trim().length > 0) {
        if (attachedTranscript) attachedTranscript.style.display = "flex";
        if (attachedTranscriptContent) attachedTranscriptContent.textContent = `“${voiceState.clientTranscript}”`;
    } else {
        if (attachedTranscript) attachedTranscript.style.display = "none";
    }

    if (sendBtn) sendBtn.disabled = false;

    const reader = new FileReader();
    reader.onloadend = () => {
        voiceState.currentAudioBase64 = reader.result;
    };
    reader.readAsDataURL(wavBlob);
}

function handleVoiceFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
        voiceState.currentAudioBase64 = reader.result;

        const audioElement = document.getElementById("voice-audio-element");
        const attachedTag = document.getElementById("audio-attached-tag");
        const attachedName = document.getElementById("audio-attached-name");
        const attachedTranscript = document.getElementById("audio-attached-transcript");
        const sendBtn = document.getElementById("btn-send");

        if (audioElement) audioElement.src = reader.result;
        if (attachedTag) attachedTag.style.display = "flex";
        if (attachedName) attachedName.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;
        if (attachedTranscript) attachedTranscript.style.display = "none";
        if (sendBtn) sendBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function loadVoiceScenarioPreset(scenario, lang) {
    const langSelect = document.getElementById("voice-language-select");
    if (langSelect && lang) {
        langSelect.value = lang;
    }

    const scn = PRESET_SCENARIOS[scenario] || PRESET_SCENARIOS.viva_panic;
    voiceState.clientTranscript = scn.text;

    const sr = 16000;
    const durationSec = 3.5;
    const totalSamples = Math.floor(sr * durationSec);
    const audioData = new Float32Array(totalSamples);

    const baseFreq = scn.baseFreq || 220;
    const jitterAmp = scn.jitterAmp || 0.05;
    const wpmCadence = scn.wpmCadence || 140;

    for (let i = 0; i < totalSamples; i++) {
        const t = i / sr;
        const tremor = jitterAmp * Math.sin(2 * Math.PI * 12 * t);
        const f0 = baseFreq * (1 + tremor);
        const fund = 0.5 * Math.sin(2 * Math.PI * f0 * t);
        const h2 = 0.25 * Math.sin(2 * Math.PI * f0 * 2 * t);
        const envelope = 0.5 + 0.5 * Math.sin(2 * Math.PI * (wpmCadence / 60) * t);
        audioData[i] = (fund + h2) * envelope * 0.7;
    }

    const wavBlob = encodeWAV16Bit(audioData, sr);
    const audioUrl = URL.createObjectURL(wavBlob);

    const audioElement = document.getElementById("voice-audio-element");
    const attachedTag = document.getElementById("audio-attached-tag");
    const attachedName = document.getElementById("audio-attached-name");
    const attachedTranscript = document.getElementById("audio-attached-transcript");
    const attachedTranscriptContent = document.getElementById("attached-transcript-content");
    const sendBtn = document.getElementById("btn-send");

    if (audioElement) audioElement.src = audioUrl;
    if (attachedTag) attachedTag.style.display = "flex";
    if (attachedName) attachedName.textContent = `Preset: ${scn.name} (${lang})`;

    if (attachedTranscript) attachedTranscript.style.display = "flex";
    if (attachedTranscriptContent) attachedTranscriptContent.textContent = `“${scn.text}”`;

    if (sendBtn) sendBtn.disabled = false;

    const reader = new FileReader();
    reader.onloadend = () => {
        voiceState.currentAudioBase64 = reader.result;
    };
    reader.readAsDataURL(wavBlob);
}

function resetVoiceAudio() {
    stopVoiceRecording();
    voiceState.audioChunks = [];
    voiceState.recordedSamples = [];
    voiceState.currentAudioBase64 = null;
    voiceState.clientTranscript = "";

    const audioElement = document.getElementById("voice-audio-element");
    const attachedTag = document.getElementById("audio-attached-tag");
    const attachedTranscript = document.getElementById("audio-attached-transcript");
    const attachedTranscriptContent = document.getElementById("attached-transcript-content");
    const liveStrip = document.getElementById("voice-live-transcript-strip");
    const sendBtn = document.getElementById("btn-send");
    const textInput = document.getElementById("reflection-input");

    if (audioElement) audioElement.src = "";
    if (attachedTag) attachedTag.style.display = "none";
    if (attachedTranscript) attachedTranscript.style.display = "none";
    if (attachedTranscriptContent) attachedTranscriptContent.textContent = "";
    if (liveStrip) liveStrip.style.display = "none";

    const hasText = textInput && textInput.value.trim().length > 0;
    if (sendBtn) sendBtn.disabled = !hasText;
}

function encodeWAV16Bit(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }
    return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

function drawLiveWaveform() {
    const canvas = document.getElementById("voice-waveform-canvas");
    if (!canvas || !voiceState.analyser || !voiceState.isRecording) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    const bufferLength = voiceState.analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    voiceState.analyser.getByteTimeDomainData(dataArray);

    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.fillRect(0, 0, width, height);

    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ef4444";
    ctx.beginPath();

    const sliceWidth = width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * height / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
    }

    ctx.lineTo(width, height / 2);
    ctx.stroke();

    voiceState.animFrameId = requestAnimationFrame(drawLiveWaveform);
}

/* ==========================================================================
   Vault Management & Modals
   ========================================================================== */
async function fetchVaultStatus() {
    try {
        const res = await fetch("/api/vault/status", { credentials: "include" });
        if (!res.ok) return;

        vaultState = await res.json();

        // If initialized but locked, auto-restore from browser session storage if available
        if (vaultState.is_initialized && !vaultState.is_unlocked) {
            const cachedPass = sessionStorage.getItem("cuk_vault_passphrase");
            if (cachedPass) {
                try {
                    const unlockRes = await fetch("/api/vault/unlock", {
                        method: "POST",
                        credentials: "include",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ passphrase: cachedPass })
                    });
                    if (unlockRes.ok) {
                        const refreshed = await fetch("/api/vault/status", { credentials: "include" });
                        if (refreshed.ok) vaultState = await refreshed.json();
                    }
                } catch (e) {
                    // Ignore background restore error
                }
            }
        }

        updateVaultUI();
    } catch (err) {
        console.warn("Vault status error:", err);
    }
}

function updateVaultUI() {
    const pulseDot = document.getElementById("vault-pulse-dot");
    const statusText = document.getElementById("vault-sidebar-status");
    const unlockBtn = document.getElementById("btn-vault-toggle");
    const lockBtn = document.getElementById("btn-lock-vault");
    const topbarText = document.getElementById("topbar-privacy-text");

    if (vaultState.is_unlocked) {
        if (pulseDot) { pulseDot.className = "status-dot emerald"; }
        if (statusText) statusText.textContent = "AES-256 Unlocked";
        if (unlockBtn) unlockBtn.style.display = "none";
        if (lockBtn) lockBtn.style.display = "inline-flex";
        if (topbarText) topbarText.textContent = "Vault Unlocked • AES-256-GCM";
    } else if (vaultState.is_initialized) {
        if (pulseDot) { pulseDot.className = "status-dot amber"; }
        if (statusText) statusText.textContent = "AES-256 Locked";
        if (unlockBtn) { unlockBtn.style.display = "inline-flex"; unlockBtn.textContent = "Unlock"; }
        if (lockBtn) lockBtn.style.display = "none";
        if (topbarText) topbarText.textContent = "Vault Locked • Zero Plaintext";
    } else {
        if (pulseDot) { pulseDot.className = "status-dot amber"; }
        if (statusText) statusText.textContent = "Vault Uninitialized";
        if (unlockBtn) { unlockBtn.style.display = "inline-flex"; unlockBtn.textContent = "Set Passphrase"; }
        if (lockBtn) lockBtn.style.display = "none";
    }
}

async function fetchVaultEntries() {
    try {
        const res = await fetch("/api/vault/entries", { credentials: "include" });
        if (!res.ok) return;

        const data = await res.json();
        allVaultEntries = data.entries || [];

        // 1. Render in Sidebar
        renderSidebarHistory(allVaultEntries);

        // 2. Render in Inspector Modal Table
        renderVaultInspectorTable(allVaultEntries);

    } catch (err) {
        console.warn("Failed loading vault entries:", err);
    }
}

function parseVaultEntryText(rawText) {
    if (!rawText) return { originalText: "", translatedText: "", isVoice: false, language: "en-IN", biomarkers: null };

    const isVoice = rawText.startsWith("[Voice Reflection");
    let originalText = rawText;
    let translatedText = "";
    let biomarkers = null;
    let lang = "en-IN";

    if (isVoice) {
        const langMatch = rawText.match(/\[Voice Reflection - [^\]]+ - ([a-zA-Z-]+)\]/);
        if (langMatch) lang = langMatch[1];

        const origMatch = rawText.match(/Original:\s*(.*?)(?=\nEnglish Translation:|\n\[Acoustic Biomarkers:|$)/s);
        if (origMatch) originalText = origMatch[1].trim();

        const transMatch = rawText.match(/English Translation:\s*(.*?)(?=\n\[Acoustic Biomarkers:|$)/s);
        if (transMatch) translatedText = transMatch[1].trim();

        const bioMatch = rawText.match(/\[Acoustic Biomarkers:\s*(.*?)\]/);
        if (bioMatch) biomarkers = bioMatch[1].trim();
    } else if (rawText.startsWith("Original:")) {
        const origMatch = rawText.match(/Original:\s*(.*?)(?=\nEnglish Translation:|$)/s);
        if (origMatch) originalText = origMatch[1].trim();

        const transMatch = rawText.match(/English Translation:\s*(.*)/s);
        if (transMatch) translatedText = transMatch[1].trim();
    }

    return {
        originalText: originalText,
        translatedText: translatedText,
        isVoice: isVoice,
        language: lang,
        biomarkers: biomarkers
    };
}

let pendingSessionToLoad = null;

function renderSidebarHistory(entries) {
    const listEl = document.getElementById("sidebar-history-list");
    const countEl = document.getElementById("sidebar-entry-count");

    if (countEl) countEl.textContent = entries.length;
    if (!listEl) return;

    if (entries.length === 0) {
        listEl.innerHTML = `<div class="history-empty">No reflections logged in local vault yet.</div>`;
        return;
    }

    listEl.innerHTML = entries.map(entry => {
        const date = new Date(entry.created_at);
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const tier = (entry.tier || "Medium").toLowerCase();
        
        let snippet = `Reflection #${entry.id}`;
        if (entry.is_decrypted) {
            const parsed = parseVaultEntryText(entry.text);
            snippet = parsed.originalText || snippet;
            if (snippet.startsWith("[DECRYPTION_ERROR") || snippet.startsWith("🔒")) {
                snippet = `🔒 Session #${entry.id}`;
            } else if (snippet.length > 28) {
                snippet = snippet.slice(0, 28) + "...";
            }
        } else {
            snippet = `🔒 Session #${entry.id}`;
        }

        return `
            <div class="history-item" data-id="${entry.id}" onclick="loadSessionIntoChat(${entry.id})" title="${entry.is_decrypted ? 'Click to open in chat' : 'Click to unlock & load into chat'}">
                <div class="history-item-top">
                    <span class="history-item-title">${escapeHtml(snippet)}</span>
                    <span class="history-tier-pill ${tier}">${entry.tier}</span>
                </div>
                <span class="history-item-date">${timeStr}</span>
            </div>
        `;
    }).join("");
}

function loadSessionIntoChat(entryId) {
    const entry = allVaultEntries.find(e => e.id === entryId);
    if (!entry) return;

    if (!entry.is_decrypted) {
        pendingSessionToLoad = entryId;
        openVaultModal();
        const desc = document.getElementById("vault-auth-desc");
        if (desc) desc.textContent = `🔒 Session #${entryId} is encrypted with AES-256-GCM. Enter your master passphrase to decrypt and open it in chat.`;
        return;
    }

    // Highlight active in sidebar
    document.querySelectorAll(".history-item").forEach(item => {
        item.classList.toggle("active", item.getAttribute("data-id") == entryId);
    });

    const welcomeHero = document.getElementById("welcome-hero");
    const chatThread = document.getElementById("chat-thread");

    if (welcomeHero) welcomeHero.style.display = "none";
    if (chatThread) {
        chatThread.style.display = "flex";
        chatThread.innerHTML = "";
    }

    const parsed = parseVaultEntryText(entry.text);
    const userTurnId = "history-user-" + entry.id;

    // Append User message with proper 3 arguments (turnId, text, voiceMeta)
    appendUserMessageTurn(
        userTurnId,
        parsed.originalText,
        parsed.isVoice ? { language: parsed.language, transcription: parsed.originalText } : null
    );

    if (parsed.translatedText && parsed.translatedText !== parsed.originalText) {
        updateUserTurnWithTranscription(
            userTurnId,
            parsed.originalText,
            parsed.translatedText,
            true,
            parsed.language
        );
    }

    // Reconstruct full assistant response so triage, distribution, and dialogue render completely
    const mockData = {
        status: "SUCCESS",
        original_text: parsed.originalText,
        transcription: parsed.isVoice ? parsed.originalText : null,
        translated_text: parsed.translatedText || parsed.originalText,
        is_translated: Boolean(parsed.translatedText && parsed.translatedText !== parsed.originalText),
        language: parsed.language,
        classification: {
            predicted_tier: entry.tier,
            confidence: entry.confidence || 0.95,
            probabilities: {
                Low: entry.tier === "Low" ? 0.85 : 0.05,
                Medium: entry.tier === "Medium" ? 0.88 : 0.10,
                High: entry.tier === "High" ? 0.92 : 0.02
            },
            stressors: entry.stressors || ["General Academic Distress"],
            latency_ms: 1.85
        },
        triage: {
            tier: entry.tier,
            tier_label: entry.tier === "Low" ? "Mild Academic Tension" : (entry.tier === "Medium" ? "Moderate Academic Distress" : "Severe Distress / Crisis Escalation"),
            urgency: entry.tier === "High" ? "Immediate Clinical Referral Handoff" : (entry.tier === "Medium" ? "Active Self-Intervention (CBT Guided)" : "Routine / Self-Care"),
            summary: entry.tier === "High" 
                ? "Deep distress and severe emotional burden detected. Standard self-care exercises are insufficient; an anonymous clinical referral token has been prepared." 
                : (entry.tier === "Medium" ? "Notable anxiety detected. Structured CBT exercises are recommended to prevent burnout." : "Stress levels are within normal academic bounds."),
            requires_clinical_escalation: entry.tier === "High",
            cbt_exercises: [
                {
                    id: "breathing_478",
                    title: "4-7-8 Parasympathetic Vagus Breathing",
                    category: "Physiological Regulation",
                    description: "Activates parasympathetic relaxation to lower cortisol prior to exams."
                }
            ]
        },
        referral_token: entry.referral_id ? {
            case_id: entry.referral_id,
            issued_at: entry.created_at,
            target_office: "University Health Centre & Student Counseling Cell",
            campus_location: "Admin Block, Kadaganchi, Kalaburagi",
            privacy_attestation: "Verified On-Device Screening: Zero diary disclosure guarantee."
        } : null,
        conversational_reply: {
            native_text: entry.tier === "High"
                ? "I want you to know that your feelings are deeply valid, and you do not have to carry this distress by yourself. Please pause right now and take three slow, grounding breaths with me. Our campus support team and counselors are here to help you through this safely."
                : (entry.tier === "Medium"
                    ? "I understand you are experiencing intense academic pressure. Let's break your workload down into manageable 15-minute study sprints with breathing pauses to restore your focus."
                    : "Thank you for checking in with your thoughts today. Your stress level is low and manageable. Continue taking mindful breaks."),
            english_text: entry.tier === "High"
                ? "I want you to know that your feelings are deeply valid, and you do not have to carry this distress by yourself."
                : "I understand you are experiencing intense academic pressure.",
            is_translated: false,
            target_language: parsed.language,
            model_name: "Decrypted-AES-256-Vault"
        }
    };

    const turnId = "saved-turn-" + entry.id;
    appendLoadingTurn(turnId, "Decrypted from local AES-256-GCM vault...");
    setTimeout(() => {
        renderAssistantResponse(turnId, mockData);
    }, 150);
}

function renderVaultInspectorTable(entries) {
    const tableBody = document.getElementById("vault-entries-body");
    const emptyNotice = document.getElementById("vault-empty-notice");
    if (!tableBody) return;

    if (entries.length === 0) {
        tableBody.innerHTML = "";
        if (emptyNotice) emptyNotice.style.display = "block";
        return;
    }

    if (emptyNotice) emptyNotice.style.display = "none";
    tableBody.innerHTML = entries.map(entry => {
        const dateStr = new Date(entry.created_at).toLocaleString();
        const tierBadge = `<span style="font-weight:700;">${entry.tier}</span>`;
        const referralLink = entry.referral_id 
            ? `<span style="font-family: var(--font-mono); color: #fb7185; font-size: 0.75rem;">${entry.referral_id}</span>`
            : `-`;

        let contentHtml = "";
        let actionBtn = "";

        if (entry.is_decrypted) {
            const parsed = parseVaultEntryText(entry.text);
            contentHtml = `
                <div style="font-size: 0.88rem; color: #f8fafc; margin-bottom: 0.25rem;">
                    ${escapeHtml(parsed.originalText)}
                </div>
                ${parsed.translatedText ? `
                    <div style="font-size: 0.76rem; color: #34d399; font-style: italic; margin-bottom: 0.2rem;">
                        🌐 Evaluated Translation: "${escapeHtml(parsed.translatedText)}"
                    </div>
                ` : ""}
                ${parsed.biomarkers ? `
                    <div style="font-size: 0.7rem; color: #38bdf8; font-family: var(--font-mono);">
                        📊 ${escapeHtml(parsed.biomarkers)}
                    </div>
                ` : ""}
            `;
            actionBtn = `<button type="button" class="btn btn-outline" style="font-size: 0.72rem; padding: 0.25rem 0.55rem; white-space: nowrap;" onclick="loadSessionIntoChat(${entry.id}); closeAllModals();">💬 Open in Chat</button>`;
        } else {
            contentHtml = `
                <span style="color: #fbbf24; font-family: var(--font-mono); font-size: 0.76rem;">
                    🔒 ${escapeHtml(entry.text)}
                </span>
            `;
            actionBtn = `<button type="button" class="btn btn-primary" style="font-size: 0.72rem; padding: 0.25rem 0.55rem; white-space: nowrap;" onclick="closeAllModals(); openVaultModal();">🔓 Unlock</button>`;
        }

        return `
            <tr>
                <td style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted);">${dateStr}</td>
                <td>${tierBadge}</td>
                <td class="cipher-cell">
                    ${contentHtml}
                </td>
                <td style="font-family: var(--font-mono); font-size: 0.72rem; color: var(--cyan);">${entry.cipher_meta?.tag_hex ? entry.cipher_meta.tag_hex.slice(0, 8) + '...' : '16B Tag'}</td>
                <td>${referralLink}</td>
                <td style="text-align: right;">${actionBtn}</td>
            </tr>
        `;
    }).join("");
}

function openVaultModal() {
    const modal = document.getElementById("vault-modal");
    const title = document.getElementById("vault-modal-title");
    const desc = document.getElementById("vault-modal-desc");
    const submitBtn = document.getElementById("vault-modal-submit-btn");

    if (!modal) return;
    modal.style.display = "flex";

    if (!vaultState.is_initialized) {
        if (title) title.textContent = "Initialize Encrypted SQLite Vault";
        if (desc) desc.textContent = "Create a master passphrase to secure your reflections with AES-256-GCM authenticated encryption (PBKDF2HMAC with 100,000 iterations).";
        if (submitBtn) submitBtn.textContent = "Create Vault Key";
    } else {
        if (title) title.textContent = "Unlock Encrypted Vault";
        if (desc) desc.textContent = "Enter your master passphrase to derive the AES-256 key and decrypt your past reflections in memory.";
        if (submitBtn) submitBtn.textContent = "Unlock Vault";
    }

    const input = document.getElementById("vault-passphrase-input");
    if (input) {
        input.value = "";
        input.focus();
    }
}

async function openVaultInspectorModal() {
    await fetchVaultEntries();
    const modal = document.getElementById("vault-inspector-modal");
    if (modal) modal.style.display = "flex";
}

function closeAllModals() {
    document.querySelectorAll(".modal-overlay").forEach(m => m.style.display = "none");
}

async function handleVaultAuthSubmit(e) {
    e.preventDefault();
    const input = document.getElementById("vault-passphrase-input");
    const passphrase = input ? input.value : "";
    if (!passphrase) return;

    const endpoint = vaultState.is_initialized ? "/api/vault/unlock" : "/api/vault/init";

    try {
        const res = await fetch(endpoint, {
            method: "POST",
            credentials: "include",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ passphrase: passphrase })
        });

        if (!res.ok) {
            const errorMsg = await parseResponseError(res, "Vault authentication failed.");
            throw new Error(errorMsg);
        }

        const data = await res.json();
        sessionStorage.setItem("cuk_vault_passphrase", passphrase);
        closeAllModals();
        await fetchVaultStatus();
        await fetchVaultEntries();

        // If user clicked an encrypted session to view it, auto-load it directly into chat now!
        if (pendingSessionToLoad) {
            const toLoad = pendingSessionToLoad;
            pendingSessionToLoad = null;
            loadSessionIntoChat(toLoad);
        } else if (allVaultEntries.length > 0) {
            // Otherwise auto-load the most recent session
            loadSessionIntoChat(allVaultEntries[0].id);
        }

    } catch (err) {
        alert("Vault Auth Error: " + err.message);
    }
}

async function handleVaultLock() {
    try {
        sessionStorage.removeItem("cuk_vault_passphrase");
        sessionStorage.removeItem("cuk_user_profile");
        const res = await fetch("/api/vault/lock", { method: "POST", credentials: "include" });
        const data = await res.json();
        await fetchVaultStatus();
        await fetchVaultEntries();
        alert(data.message);
    } catch (err) {
        alert("Error locking vault: " + err.message);
    }
}

async function fetchModelMetrics() {
    try {
        const res = await fetch("/api/metrics");
        if (res.ok) {
            const metrics = await res.json();
            const slaTag = document.getElementById("meter-latency-sla");
            if (slaTag) {
                slaTag.textContent = `CPU Latency: ${metrics.avg_latency_ms} ms (<300ms SLA)`;
            }
        }
    } catch (err) {
        console.warn("Failed fetching metrics:", err);
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/* ==========================================================================
   Language View Toggle & Speech Synthesis (TTS)
   ========================================================================== */
function toggleLanguageView(replyId) {
    const englishEl = document.getElementById("text-english-" + replyId);
    const btn = document.getElementById("btn-toggle-" + replyId);
    if (!englishEl || !btn) return;

    if (englishEl.style.display === "none" || !englishEl.style.display) {
        englishEl.style.display = "flex";
        btn.textContent = "🌐 Hide English";
    } else {
        englishEl.style.display = "none";
        btn.textContent = "🌐 View English";
    }
}

function handleSpeakBtn(btn) {
    const text = btn.getAttribute("data-speech");
    const lang = btn.getAttribute("data-lang") || "hi-IN";
    if (!text || !window.speechSynthesis) return;

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = lang;
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
}
