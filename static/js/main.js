/**
 * HealthCare AI - Main JavaScript
 * Handles UI interactions and API calls
 * Phase 1: Added Theme Manager, Disclaimer Flow, Health Tips, Prediction Feedback
 * Phase 2: Added Profile, Vitals Tracker, Timeline, Health Goals
 */

// ============================================
// THEME MANAGER (Feature #15)
// ============================================
const themeManager = {
    init() {
        const saved = localStorage.getItem('theme') || 'dark';
        this.apply(saved);
    },
    apply(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        const sunIcon = document.getElementById('sunIcon');
        const moonIcon = document.getElementById('moonIcon');
        if (sunIcon && moonIcon) {
            if (theme === 'light') {
                sunIcon.classList.remove('hidden');
                moonIcon.classList.add('hidden');
            } else {
                sunIcon.classList.add('hidden');
                moonIcon.classList.remove('hidden');
            }
        }
        localStorage.setItem('theme', theme);
    },
    toggle() {
        const current = localStorage.getItem('theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        this.apply(next);
    },
    get() {
        return localStorage.getItem('theme') || 'dark';
    }
};

// Initialize theme immediately to prevent flash
themeManager.init();

// State management
const state = {
    selectedSymptoms: new Set(),
    activeCategory: null,
    allSymptoms: [],
    categories: {},
    diseases: {},
    modelMetrics: null,
    predictionCount: parseInt(localStorage.getItem('predictionCount') || '0'),
    currentPredictionId: null,
    // Phase 2 state
    vitalsCharts: {},
    activeVitalsTab: 'log',
    activeRecordsTab: 'list',
    timelineFilters: { severity: null, dateFrom: null, dateTo: null, symptom: null }
};

// Initialize the application
async function initApp() {
    console.log('Initializing HealthCare AI...');

    await Promise.all([
        loadModelMetrics(),
        loadSymptoms(),
        loadDiseases()
    ]);

    updateStats();
    loadHealthTip();

    // Profile completeness indicator
    if (typeof updateProfileIndicator === 'function') {
        updateProfileIndicator();
    }

    // Auto-fill forms from profile
    if (typeof autoFillFormsFromProfile === 'function') {
        setTimeout(autoFillFormsFromProfile, 300);
    }

    setTimeout(() => {
        if (!localStorage.getItem('onboarding_complete')) {
            if (typeof OnboardingManager !== 'undefined') {
                window.onboardingManager = new OnboardingManager();
                window.onboardingManager.start();
            }
        }
    }, 1000);

    console.log('Application initialized successfully');
}

// API Helper — injects profile data automatically
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' }
    };

    if (data) {
        // Inject profile context into predict calls
        if (endpoint === '/api/predict' && typeof window.profileManager !== 'undefined') {
            const profileData = window.profileManager.getApiData();
            data = { ...data, profile: profileData };
        }
        options.body = JSON.stringify(data);
    }

    try {
        const response = await fetch(endpoint, options);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API call failed: ${endpoint}`, error);
        throw error;
    }
}

// Load model metrics
async function loadModelMetrics() {
    try {
        const metrics = await apiCall('/api/model/metrics');
        state.modelMetrics = metrics;
        const statusEl = document.getElementById('modelStatus');
        if (metrics.is_trained) {
            statusEl.innerHTML = `
                <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span class="text-gray-400">ONLINE</span>`;
        }
        document.getElementById('accuracyRate').textContent = `${metrics.cross_val_mean}%`;
    } catch (error) {
        console.error('Failed to load model metrics:', error);
        document.getElementById('modelStatus').innerHTML = `
            <span class="w-2 h-2 rounded-full bg-red-500"></span>
            <span class="text-gray-400">Model Error</span>`;
    }
}

async function loadSymptoms() {
    try {
        const data = await apiCall('/api/symptoms');
        state.allSymptoms = data.symptoms || [];
        state.categories = data.categories || {};
    } catch (error) {
        console.error('Failed to load symptoms:', error);
    }
}

async function loadDiseases() {
    try {
        const data = await apiCall('/api/diseases');
        state.diseases = data.diseases || {};
    } catch (error) {
        console.error('Failed to load diseases:', error);
    }
}

function updateStats() {
    document.getElementById('diseaseCount').textContent = Object.keys(state.diseases).length || '--';
    document.getElementById('symptomCount').textContent = state.allSymptoms.length || '--';
    document.getElementById('predictionCount').textContent = state.predictionCount;
}

function sanitize(str) {
    if (typeof str !== 'string') return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================
// DISCLAIMER FLOW (Feature #21)
// ============================================
function onDisclaimerCheckboxChange() {
    const checkbox = document.getElementById('disclaimerCheckbox');
    const btn = document.getElementById('disclaimerAcceptBtn');
    btn.disabled = !checkbox.checked;
}

function acceptDisclaimer() {
    const checkbox = document.getElementById('disclaimerCheckbox');
    if (!checkbox || !checkbox.checked) return;
    const modal = document.getElementById('disclaimerModal');
    modal.classList.add('animate-fade-out');
    setTimeout(() => {
        modal.classList.add('hidden');
        modal.style.display = 'none';
        document.getElementById('mainApp').classList.remove('hidden');
        localStorage.setItem('disclaimer_accepted', 'true');
        localStorage.setItem('disclaimerAccepted', 'true');
        initApp();
    }, 300);
}

function showDisclaimer() {
    const modal = document.getElementById('disclaimerModal');
    modal.classList.remove('hidden', 'animate-fade-out');
    modal.style.display = 'flex';
}

// Emergency
function showEmergency(message) {
    const banner = document.getElementById('emergencyBanner');
    document.getElementById('emergencyMessage').textContent = message;
    banner.classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function hideEmergency() {
    document.getElementById('emergencyBanner').classList.add('hidden');
}

// ============================================
// MODAL FUNCTIONS
// ============================================
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;
    modal.classList.remove('hidden');
    modal.classList.add('visible');
    document.body.style.overflow = 'hidden';

    if (modalId === 'symptomChecker') {
        initSymptomChecker();
    } else if (modalId === 'diseaseDb') {
        initDiseaseDatabase();
    } else if (modalId === 'records') {
        loadHealthRecords();
    } else if (modalId === 'vitalsModal') {
        initVitalsModal();
    } else if (modalId === 'goalsModal') {
        if (typeof renderGoalsModal === 'function') renderGoalsModal();
    } else if (modalId === 'profileModal') {
        if (typeof populateProfileForm === 'function') populateProfileForm();
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (!modal) return;
    modal.classList.add('hidden');
    modal.classList.remove('visible');
    document.body.style.overflow = '';
}

// ============================================
// SYMPTOM CHECKER
// ============================================
function initSymptomChecker() {
    const categoryFilters = document.getElementById('categoryFilters');
    const categories = Object.keys(state.categories);

    categoryFilters.innerHTML = `
        <button onclick="filterByCategory(null)" class="category-btn ${state.activeCategory === null ? 'active' : ''}">
            All Symptoms
        </button>
        ${categories.map(cat => `
            <button onclick="filterByCategory('${sanitize(cat)}')" class="category-btn ${state.activeCategory === cat ? 'active' : ''}">
                ${sanitize(cat)}
            </button>
        `).join('')}`;

    renderSymptomTags();
    updateSelectedDisplay();

    // Auto-fill age from profile
    if (typeof window.profileManager !== 'undefined') {
        const age = window.profileManager.get('age');
        const patientAge = document.getElementById('patientAge');
        if (patientAge && age && !patientAge.value) {
            patientAge.value = age;
        }
    }
}

function filterByCategory(category) {
    state.activeCategory = category;
    initSymptomChecker();
}

function renderSymptomTags() {
    const container = document.getElementById('symptomTags');
    let symptomsToShow = state.allSymptoms;
    if (state.activeCategory && state.categories[state.activeCategory]) {
        symptomsToShow = state.categories[state.activeCategory];
    }
    container.innerHTML = symptomsToShow.map(symptom => `
        <button onclick="toggleSymptom('${sanitize(symptom)}')"
            class="symptom-tag ${state.selectedSymptoms.has(symptom) ? 'selected' : ''}">
            ${sanitize(symptom)}
        </button>`).join('');
}

function toggleSymptom(symptom) {
    if (state.selectedSymptoms.has(symptom)) {
        state.selectedSymptoms.delete(symptom);
    } else {
        state.selectedSymptoms.add(symptom);
    }
    renderSymptomTags();
    updateSelectedDisplay();
}

function updateSelectedDisplay() {
    const container = document.getElementById('selectedSymptoms');
    const countSpan = document.getElementById('selectedCount');
    countSpan.textContent = state.selectedSymptoms.size;

    if (state.selectedSymptoms.size === 0) {
        container.innerHTML = '<span class="text-gray-500 text-sm">No symptoms selected yet</span>';
    } else {
        container.innerHTML = Array.from(state.selectedSymptoms).map(symptom => `
            <span class="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm bg-gradient-to-r from-blue-500/20 to-purple-500/20 border border-blue-500/30">
                ${sanitize(symptom)}
                <button onclick="toggleSymptom('${sanitize(symptom)}')" class="hover:text-red-400 transition text-lg leading-none">&times;</button>
            </span>`).join('');
    }
}

function generatePredictionId() {
    return 'pred_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

async function analyzeSymptoms() {
    if (state.selectedSymptoms.size === 0) {
        alert('Please select at least one symptom');
        return;
    }

    const symptoms = Array.from(state.selectedSymptoms);
    const duration = document.getElementById('symptomDuration').value;
    const ageInput = document.getElementById('patientAge').value;
    const age = ageInput ? parseInt(ageInput) : null;

    const btn = document.getElementById('analyzeBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="flex items-center justify-center gap-2"><span class="loading-spinner"></span> Analyzing...</span>';
    btn.disabled = true;

    state.currentPredictionId = generatePredictionId();

    try {
        const response = await apiCall('/api/predict', 'POST', { symptoms, duration, age });

        if (response.emergency && response.emergency.is_emergency) {
            showEmergency(response.emergency.message);
        }

        state.predictionCount++;
        localStorage.setItem('predictionCount', state.predictionCount.toString());
        updateStats();
        displayResults(response);
    } catch (error) {
        console.error('Prediction failed:', error);
        alert('Failed to analyze symptoms. Please try again.');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

function displayResults(response) {
    const container = document.getElementById('analysisResults');
    container.classList.remove('hidden');

    const predictions = response.predictions || [];
    const emergency = response.emergency;

    if (predictions.length === 0) {
        container.innerHTML = `
            <div class="result-card text-center py-8">
                <div class="text-5xl mb-4">🔍</div>
                <h3 class="text-xl font-semibold mb-2">No Strong Matches Found</h3>
                <p class="text-gray-400">Your symptoms don't strongly match any conditions in our database. Please consult a healthcare professional.</p>
            </div>`;
        return;
    }

    const severityColors = {
        'low': 'from-emerald-500 to-green-600',
        'moderate': 'from-amber-500 to-yellow-600',
        'high': 'from-orange-500 to-red-500',
        'critical': 'from-red-500 to-red-700'
    };

    let emergencyHtml = '';
    if (emergency && emergency.is_emergency) {
        emergencyHtml = `
            <div class="bg-red-900/40 border border-red-500/50 rounded-xl p-5 mb-6 emergency-alert">
                <h3 class="font-bold text-red-400 mb-3 flex items-center gap-2">
                    <span class="text-xl">🚨</span> ${emergency.message}
                </h3>
                <ul class="text-sm space-y-2">
                    ${emergency.recommendations.map(r => `<li class="flex items-start gap-2"><span class="text-red-400">•</span> ${sanitize(r)}</li>`).join('')}
                </ul>
            </div>`;
    }

    container.innerHTML = `
        ${emergencyHtml}
        <div class="bg-blue-900/20 border border-blue-500/30 rounded-xl p-4 mb-6">
            <p class="text-sm flex items-start gap-2">
                <span class="text-blue-400">ℹ️</span>
                <span><strong>Important:</strong> These are ML model suggestions, not medical diagnoses. Always consult a healthcare professional.</span>
            </p>
        </div>
        <h3 class="text-lg font-semibold mb-4">Possible Conditions (${predictions.length} matches)</h3>
        ${predictions.map((pred, idx) => `
        <div class="disease-card ${pred.severity}">
            <div class="flex items-start justify-between mb-4">
                <div>
                    <h4 class="text-lg font-bold flex items-center gap-2">
                        <span class="w-6 h-6 rounded-full bg-gradient-to-r ${severityColors[pred.severity]} flex items-center justify-center text-xs font-bold text-white">${idx + 1}</span>
                        ${sanitize(pred.disease)}
                    </h4>
                    <span class="text-xs text-gray-400">${sanitize(pred.category)}</span>
                </div>
                <div class="text-right">
                    <div class="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">${pred.confidence}%</div>
                    <div class="text-xs text-gray-500">confidence</div>
                </div>
            </div>
            <div class="confidence-bar mb-4">
                <div class="confidence-fill bg-gradient-to-r ${severityColors[pred.severity]}" style="width: ${pred.confidence}%"></div>
            </div>
            <div class="flex flex-wrap gap-2 mb-4">
                <span class="severity-badge severity-${pred.severity}">${pred.severity.toUpperCase()}</span>
                <span class="px-2 py-1 rounded-full text-xs bg-white/10 border border-white/20">Urgency: ${pred.urgency}/5</span>
            </div>
            <div class="space-y-3 text-sm">
                <p><strong class="text-blue-400">Matched Symptoms:</strong> <span class="text-gray-300">${pred.matched_symptoms.map(s => sanitize(s)).join(', ')}</span></p>
                <p><strong class="text-emerald-400">Recommendations:</strong> <span class="text-gray-300">${sanitize(pred.recommendations)}</span></p>
                <p><strong class="text-amber-400">When to Seek Help:</strong> <span class="text-gray-300">${sanitize(pred.when_to_seek_help)}</span></p>
            </div>
            <div class="mt-4 p-3 bg-white/5 rounded-lg border border-white/5">
                <p class="text-xs font-semibold text-gray-400 mb-2">PRECAUTIONS:</p>
                <ul class="text-xs text-gray-300 space-y-1">
                    ${pred.precautions.map(p => `<li class="flex items-start gap-2"><span class="text-gray-500">•</span> ${sanitize(p)}</li>`).join('')}
                </ul>
            </div>
        </div>`).join('')}

        <div class="bg-red-900/20 border border-red-500/30 rounded-xl p-4 mt-6">
            <p class="text-sm text-red-300 flex items-start gap-2">
                <span>⚠</span>
                <span><strong>Reminder:</strong> This is not a medical diagnosis. Please consult a qualified healthcare provider.</span>
            </p>
        </div>

        <!-- Prediction Feedback Section (Feature #18) -->
        <div id="feedbackSection" class="feedback-card">
            <h4 class="text-lg font-semibold mb-4 flex items-center gap-2">
                <span>📝</span> Was this prediction helpful?
            </h4>
            <div class="mb-4">
                <p class="text-sm text-gray-400 mb-2">Rate this prediction:</p>
                <div class="star-rating" id="starRating">
                    ${[1,2,3,4,5].map(i => `<span class="star" data-rating="${i}" onclick="setFeedbackRating(${i})" onmouseenter="hoverStars(${i})" onmouseleave="unhoverStars()">☆</span>`).join('')}
                </div>
            </div>
            <div class="mb-4">
                <p class="text-sm text-gray-400 mb-2">Was this accurate for you?</p>
                <div class="flex gap-3">
                    <button onclick="setFeedbackAccuracy(true)" id="accuracyYes" class="feedback-accuracy-btn">👍 Yes</button>
                    <button onclick="setFeedbackAccuracy(false)" id="accuracyNo" class="feedback-accuracy-btn">👎 No</button>
                </div>
            </div>
            <div class="mb-4">
                <textarea id="feedbackComment" class="feedback-textarea" rows="2" maxlength="200" placeholder="Tell us more (optional) — max 200 characters"></textarea>
                <p class="text-xs text-gray-500 mt-1 text-right"><span id="feedbackCharCount">0</span>/200</p>
            </div>
            <button onclick="submitFeedback()" id="feedbackSubmitBtn" class="btn-primary px-6 py-2 text-sm">Submit Feedback</button>
        </div>`;

    const commentEl = document.getElementById('feedbackComment');
    if (commentEl) {
        commentEl.addEventListener('input', () => {
            document.getElementById('feedbackCharCount').textContent = commentEl.value.length;
        });
    }

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================
// PREDICTION FEEDBACK (Feature #18)
// ============================================
let feedbackState = { rating: 0, wasAccurate: null, comment: '' };

function setFeedbackRating(rating) {
    feedbackState.rating = rating;
    document.querySelectorAll('#starRating .star').forEach((star, idx) => {
        star.classList.toggle('selected', idx < rating);
        star.textContent = idx < rating ? '★' : '☆';
    });
}

function hoverStars(rating) {
    document.querySelectorAll('#starRating .star').forEach((star, idx) => {
        if (idx < rating) { star.classList.add('hovered'); star.textContent = '★'; }
    });
}

function unhoverStars() {
    document.querySelectorAll('#starRating .star').forEach((star, idx) => {
        star.classList.remove('hovered');
        star.textContent = idx < feedbackState.rating ? '★' : '☆';
    });
}

function setFeedbackAccuracy(isAccurate) {
    feedbackState.wasAccurate = isAccurate;
    document.getElementById('accuracyYes').classList.toggle('selected-yes', isAccurate);
    document.getElementById('accuracyNo').classList.toggle('selected-no', !isAccurate);
}

async function submitFeedback() {
    const comment = document.getElementById('feedbackComment')?.value || '';
    feedbackState.comment = comment;
    if (feedbackState.rating === 0) { alert('Please select a star rating'); return; }

    const btn = document.getElementById('feedbackSubmitBtn');
    btn.disabled = true;
    btn.textContent = 'Submitting...';

    try {
        await apiCall('/api/feedback', 'POST', {
            prediction_id: state.currentPredictionId,
            rating: feedbackState.rating,
            was_accurate: feedbackState.wasAccurate,
            comment: feedbackState.comment
        });
        document.getElementById('feedbackSection').innerHTML = `
            <div class="feedback-thank-you">
                <div class="text-4xl mb-3">🎉</div>
                <h4 class="text-lg font-semibold mb-2">Thank you for your feedback!</h4>
                <p class="text-sm text-gray-400">Your input helps us improve our predictions.</p>
            </div>`;
    } catch (error) {
        console.error('Feedback submission failed:', error);
        btn.disabled = false;
        btn.textContent = 'Submit Feedback';
        alert('Failed to submit feedback. Please try again.');
    }
    feedbackState = { rating: 0, wasAccurate: null, comment: '' };
}

// ============================================
// DAILY HEALTH TIP (Feature #19)
// ============================================
async function loadHealthTip() {
    const lastDismissed = localStorage.getItem('healthTipDismissed');
    if (lastDismissed) {
        const hoursSince = (Date.now() - parseInt(lastDismissed)) / (1000 * 60 * 60);
        if (hoursSince < 24) return;
    }
    try {
        const data = await apiCall('/api/health-tip');
        if (data && data.text) {
            document.getElementById('tipIcon').textContent = data.icon || '💡';
            document.getElementById('tipText').textContent = data.text;
            document.getElementById('tipDetail').textContent = data.detail || '';
            document.getElementById('tipCategoryBadge').textContent = data.category || 'Health';
            document.getElementById('healthTipCard').classList.remove('hidden');
        }
    } catch (error) {
        console.error('Failed to load health tip:', error);
    }
}

function toggleTipDetail() {
    const detail = document.getElementById('tipDetail');
    const arrow = document.getElementById('tipLearnMoreArrow');
    const text = document.getElementById('tipLearnMoreText');
    if (detail.classList.contains('expanded')) {
        detail.classList.remove('expanded');
        detail.classList.add('hidden');
        arrow.style.transform = 'rotate(0deg)';
        text.textContent = 'Learn More';
    } else {
        detail.classList.remove('hidden');
        requestAnimationFrame(() => detail.classList.add('expanded'));
        arrow.style.transform = 'rotate(180deg)';
        text.textContent = 'Show Less';
    }
}

function dismissHealthTip() {
    const tipCard = document.getElementById('healthTipCard');
    tipCard.classList.add('animate-fade-out');
    setTimeout(() => tipCard.classList.add('hidden'), 300);
    localStorage.setItem('healthTipDismissed', Date.now().toString());
}

// ============================================
// ONBOARDING REPLAY (Feature #17)
// ============================================
function replayTutorial() {
    localStorage.removeItem('onboarding_complete');
    if (typeof OnboardingManager !== 'undefined') {
        window.onboardingManager = new OnboardingManager();
        window.onboardingManager.start();
    }
}

// ============================================
// BMI CALCULATOR
// ============================================
async function calculateBMI() {
    const weight = parseFloat(document.getElementById('bmiWeight').value);
    const height = parseFloat(document.getElementById('bmiHeight').value);
    const age = parseInt(document.getElementById('bmiAge').value);
    const gender = document.getElementById('bmiGender').value;
    const activity = parseFloat(document.getElementById('bmiActivity').value);

    if (!weight || !height || !age) {
        alert('Please fill in all required fields');
        return;
    }

    try {
        const response = await apiCall('/api/bmi', 'POST', { weight, height, age, gender, activity });
        displayBMIResults(response);
    } catch (error) {
        console.error('BMI calculation failed:', error);
        alert('Failed to calculate BMI. Please check your inputs.');
    }
}

function displayBMIResults(data) {
    const resultsDiv = document.getElementById('bmiResults');
    resultsDiv.classList.remove('hidden');
    const colorMap = {
        'Underweight': { gradient: 'from-blue-400 to-cyan-400', text: 'text-blue-400' },
        'Normal weight': { gradient: 'from-emerald-400 to-green-400', text: 'text-emerald-400' },
        'Overweight': { gradient: 'from-amber-400 to-yellow-400', text: 'text-amber-400' },
        'Obese': { gradient: 'from-red-400 to-orange-400', text: 'text-red-400' }
    };
    const colors = colorMap[data.category] || colorMap['Normal weight'];
    resultsDiv.innerHTML = `
        <div class="result-card">
            <div class="text-center mb-6">
                <div class="text-6xl font-bold bg-gradient-to-r ${colors.gradient} bg-clip-text text-transparent">${data.bmi}</div>
                <div class="text-xl ${colors.text} font-medium mt-1">${data.category}</div>
            </div>
            <div class="space-y-3">
                <div class="flex justify-between py-3 border-b border-white/10">
                    <span class="text-gray-400">Ideal Weight Range</span>
                    <span class="font-medium">${data.ideal_weight_range}</span>
                </div>
                <div class="flex justify-between py-3 border-b border-white/10">
                    <span class="text-gray-400">Daily Calorie Needs (TDEE)</span>
                    <span class="font-medium">${data.tdee} kcal</span>
                </div>
                <div class="flex justify-between py-3 border-b border-white/10">
                    <span class="text-gray-400">Basal Metabolic Rate</span>
                    <span class="font-medium">${data.bmr} kcal</span>
                </div>
            </div>
            <div class="mt-6 p-4 bg-white/5 rounded-xl border border-white/10">
                <h4 class="font-semibold mb-2 flex items-center gap-2">💡 Personalized Advice</h4>
                <p class="text-gray-300 text-sm">${sanitize(data.advice)}</p>
            </div>
            <p class="text-xs text-gray-500 mt-4 text-center">⚠️ BMI is a general indicator and may not account for muscle mass, bone density, or other factors.</p>
        </div>`;
}

// ============================================
// CHATBOT
// ============================================
async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    const chatContainer = document.getElementById('chatMessages');
    chatContainer.innerHTML += `<div class="chat-message user">${sanitize(message)}</div>`;
    input.value = '';
    chatContainer.innerHTML += `<div class="chat-message bot" id="loadingMsg"><div class="loading-spinner"></div></div>`;
    chatContainer.scrollTop = chatContainer.scrollHeight;

    try {
        const response = await apiCall('/api/chat', 'POST', { message });
        document.getElementById('loadingMsg').remove();
        chatContainer.innerHTML += `<div class="chat-message bot">${response.response}</div>`;
    } catch (error) {
        document.getElementById('loadingMsg').remove();
        chatContainer.innerHTML += `<div class="chat-message bot bg-red-900/30 border-red-500/30">Sorry, I encountered an error. Please try again.</div>`;
    }
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// ============================================
// DISEASE DATABASE
// ============================================
function initDiseaseDatabase() {
    const categories = [...new Set(Object.values(state.diseases).map(d => d.category))].sort();
    const categoryFilters = document.getElementById('diseaseCategoryFilters');
    categoryFilters.innerHTML = `
        <button onclick="filterDiseases(null, event)" class="category-btn active">All</button>
        ${categories.map(cat => `
            <button onclick="filterDiseases('${sanitize(cat)}', event)" class="category-btn">
                ${sanitize(cat)}
            </button>`).join('')}`;
    displayDiseases(Object.entries(state.diseases));
}

function filterDiseases(category, event) {
    const filtered = Object.entries(state.diseases).filter(([name, info]) =>
        !category || info.category === category);
    displayDiseases(filtered);
    document.querySelectorAll('#diseaseCategoryFilters button').forEach(btn => btn.classList.remove('active'));
    if (event && event.target) event.target.classList.add('active');
}

function searchDiseases() {
    const query = document.getElementById('diseaseSearch').value.toLowerCase();
    const filtered = Object.entries(state.diseases).filter(([name, info]) =>
        name.toLowerCase().includes(query) ||
        info.symptoms.some(s => s.toLowerCase().includes(query)) ||
        info.category.toLowerCase().includes(query) ||
        (info.description && info.description.toLowerCase().includes(query)));
    displayDiseases(filtered);
}

function displayDiseases(diseases) {
    const container = document.getElementById('diseaseList');
    if (diseases.length === 0) {
        container.innerHTML = '<p class="text-gray-400 text-center py-8">No diseases found matching your search.</p>';
        return;
    }
    container.innerHTML = diseases.map(([name, info]) => `
        <div class="disease-card ${info.severity}">
            <div class="flex items-start justify-between mb-3">
                <div>
                    <h4 class="text-lg font-semibold">${sanitize(name)}</h4>
                    <span class="text-xs text-gray-400">${sanitize(info.category)}</span>
                </div>
                <span class="severity-badge severity-${info.severity}">${info.severity.toUpperCase()}</span>
            </div>
            <p class="text-gray-300 text-sm mb-3">${sanitize(info.description || '')}</p>
            <div class="text-sm space-y-2">
                <p><strong class="text-blue-400">Symptoms:</strong> <span class="text-gray-400">${info.symptoms.slice(0, 6).map(s => sanitize(s)).join(', ')}${info.symptoms.length > 6 ? '...' : ''}</span></p>
                <p><strong class="text-emerald-400">Recommendations:</strong> <span class="text-gray-400">${sanitize(info.recommendations)}</span></p>
            </div>
        </div>`).join('');
}

// ============================================
// HEALTH RECORDS
// ============================================
async function loadHealthRecords() {
    const activeTab = state.activeRecordsTab || 'list';
    switchRecordsTab(activeTab);
}

function switchRecordsTab(tab) {
    state.activeRecordsTab = tab;

    ['list', 'timeline'].forEach(t => {
        const btn = document.getElementById(`recordsTab-${t}`);
        const panel = document.getElementById(`recordsPanel-${t}`);
        if (btn) btn.classList.toggle('active', t === tab);
        if (panel) panel.classList.toggle('hidden', t !== tab);
    });

    if (tab === 'list') {
        loadRecordsList();
    } else if (tab === 'timeline') {
        loadTimeline();
    }
}

async function loadRecordsList() {
    try {
        const data = await apiCall('/api/records');
        const predictionContainer = document.getElementById('predictionHistory');
        const bmiContainer = document.getElementById('bmiHistory');

        if (!data.health_records || data.health_records.length === 0) {
            predictionContainer.innerHTML = '<p class="text-gray-500 text-sm py-4">No prediction records yet.</p>';
        } else {
            predictionContainer.innerHTML = data.health_records.map(record => `
                <div class="record-item">
                    <div class="text-xs text-gray-500 mb-1">${record.timestamp}</div>
                    <div class="font-medium text-blue-400">${sanitize(record.disease || record.top_prediction || '')}</div>
                    <div class="text-xs text-gray-400 mt-1">Symptoms: ${record.symptoms.map(s => sanitize(s)).join(', ')}</div>
                </div>`).join('');
        }

        if (!data.bmi_records || data.bmi_records.length === 0) {
            bmiContainer.innerHTML = '<p class="text-gray-500 text-sm py-4">No BMI records yet.</p>';
        } else {
            bmiContainer.innerHTML = data.bmi_records.map(record => `
                <div class="record-item">
                    <div class="text-xs text-gray-500 mb-1">${record.timestamp}</div>
                    <div class="font-medium">BMI: <span class="text-emerald-400">${record.bmi}</span> <span class="text-gray-400">(${sanitize(record.category)})</span></div>
                </div>`).join('');
        }
    } catch (error) {
        console.error('Failed to load records:', error);
    }
}

async function loadTimeline() {
    const container = document.getElementById('timelineContainer');
    if (!container) return;
    container.innerHTML = '<div class="flex justify-center py-8"><div class="loading-spinner"></div></div>';

    try {
        const data = await apiCall('/api/records/timeline');
        renderTimeline(data.timeline || [], data.total || 0);
        renderPatternCharts(data.timeline || []);
    } catch (error) {
        console.error('Timeline load failed:', error);
        container.innerHTML = '<p class="text-gray-500 text-center py-8">Failed to load timeline.</p>';
    }
}

function renderTimeline(timeline, total) {
    const container = document.getElementById('timelineContainer');
    if (!container) return;

    if (total === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <div class="text-4xl mb-3">📋</div>
                <p>No symptom history yet.</p>
                <p class="text-sm mt-2">Analyze your symptoms to start building your timeline.</p>
            </div>`;
        return;
    }

    const severityBorderMap = {
        low: 'border-l-emerald-500',
        moderate: 'border-l-yellow-400',
        high: 'border-l-orange-500',
        critical: 'border-l-red-500'
    };

    const severityBadgeMap = {
        low: 'severity-low',
        moderate: 'severity-moderate',
        high: 'severity-high',
        critical: 'severity-critical'
    };

    let html = '';
    timeline.forEach(week => {
        html += `
            <div class="mb-6">
                <div class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3 px-2">
                    ${week.week_label}
                </div>`;

        week.records.forEach(record => {
            const severity = record.severity || 'low';
            const borderClass = severityBorderMap[severity] || 'border-l-gray-500';
            const badgeClass = severityBadgeMap[severity] || '';
            const ts = record.timestamp ? new Date(record.timestamp).toLocaleString() : 'Unknown';
            const confidence = record.confidence ? `${record.confidence}%` : '--';

            html += `
                <div class="timeline-entry record-item border-l-4 ${borderClass} pl-4 mb-3 cursor-pointer"
                    onclick="toggleTimelineEntry('${record.id}')">
                    <div class="flex items-start justify-between">
                        <div class="flex-1">
                            <div class="flex items-center gap-2 mb-2">
                                <span class="text-xs text-gray-500">${ts}</span>
                                <span class="severity-badge ${badgeClass} text-[10px] px-2 py-0.5">${severity.toUpperCase()}</span>
                            </div>
                            <div class="font-medium text-blue-400 mb-2">${sanitize(record.top_prediction || record.disease || '')}</div>
                            <!-- Confidence bar -->
                            <div class="flex items-center gap-2 mb-2">
                                <div class="flex-1 confidence-bar h-1.5">
                                    <div class="confidence-fill bg-gradient-to-r from-blue-400 to-purple-400"
                                        style="width: ${record.confidence || 0}%"></div>
                                </div>
                                <span class="text-xs text-gray-400">${confidence}</span>
                            </div>
                            <!-- Symptom chips -->
                            <div class="flex flex-wrap gap-1">
                                ${(record.symptoms || []).slice(0, 5).map(s =>
                                    `<span class="px-2 py-0.5 rounded-full text-xs bg-white/10 border border-white/10">${sanitize(s)}</span>`
                                ).join('')}
                                ${(record.symptoms || []).length > 5 ? `<span class="text-xs text-gray-500">+${record.symptoms.length - 5} more</span>` : ''}
                            </div>
                        </div>
                        <span class="text-gray-600 ml-3 text-lg" id="timelineArrow-${record.id}">▸</span>
                    </div>
                    <!-- Expanded detail -->
                    <div id="timelineDetail-${record.id}" class="hidden mt-3 pt-3 border-t border-white/10">
                        <p class="text-xs text-gray-400 mb-2">All predictions:</p>
                        ${(record.all_predictions || []).map((p, i) => `
                            <div class="flex items-center justify-between text-sm py-1">
                                <span class="${i === 0 ? 'text-white font-medium' : 'text-gray-400'}">${sanitize(p.disease)}</span>
                                <span class="text-gray-400">${p.confidence}%</span>
                            </div>`).join('')}
                        ${record.duration ? `<p class="text-xs text-gray-500 mt-2">Duration: ${sanitize(record.duration)}</p>` : ''}
                        ${record.age ? `<p class="text-xs text-gray-500">Age at time: ${record.age}</p>` : ''}
                    </div>
                </div>`;
        });

        html += '</div>';
    });

    container.innerHTML = html;
}

function toggleTimelineEntry(recordId) {
    const detail = document.getElementById(`timelineDetail-${recordId}`);
    const arrow = document.getElementById(`timelineArrow-${recordId}`);
    if (!detail) return;
    const isHidden = detail.classList.contains('hidden');
    detail.classList.toggle('hidden', !isHidden);
    if (arrow) arrow.textContent = isHidden ? '▾' : '▸';
}

function renderPatternCharts(timeline) {
    // Flatten all records
    const allRecords = timeline.flatMap(w => w.records);

    // Most frequent symptoms this month
    const symCount = {};
    allRecords.forEach(r => {
        (r.symptoms || []).forEach(s => {
            symCount[s] = (symCount[s] || 0) + 1;
        });
    });
    const topSymptoms = Object.entries(symCount)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 7);

    // Most common predictions
    const predCount = {};
    allRecords.forEach(r => {
        const p = r.top_prediction || r.disease || '';
        if (p) predCount[p] = (predCount[p] || 0) + 1;
    });
    const topPreds = Object.entries(predCount)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6);

    // Render symptom bar chart
    const symCtx = document.getElementById('symptomPatternChart');
    if (symCtx && topSymptoms.length > 0 && typeof Chart !== 'undefined') {
        if (state.vitalsCharts['symptomPattern']) {
            state.vitalsCharts['symptomPattern'].destroy();
        }
        state.vitalsCharts['symptomPattern'] = new Chart(symCtx, {
            type: 'bar',
            data: {
                labels: topSymptoms.map(([s]) => s),
                datasets: [{
                    label: 'Frequency',
                    data: topSymptoms.map(([, c]) => c),
                    backgroundColor: 'rgba(59, 130, 246, 0.6)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 1,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                }
            }
        });
    }

    // Render prediction donut chart
    const predCtx = document.getElementById('predictionPatternChart');
    if (predCtx && topPreds.length > 0 && typeof Chart !== 'undefined') {
        if (state.vitalsCharts['predictionPattern']) {
            state.vitalsCharts['predictionPattern'].destroy();
        }
        const colors = ['#3b82f6','#8b5cf6','#22c55e','#f59e0b','#ef4444','#06b6d4'];
        state.vitalsCharts['predictionPattern'] = new Chart(predCtx, {
            type: 'doughnut',
            data: {
                labels: topPreds.map(([p]) => p),
                datasets: [{
                    data: topPreds.map(([, c]) => c),
                    backgroundColor: colors,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { position: 'bottom', labels: { color: '#94a3b8', padding: 12, font: { size: 11 } } }
                }
            }
        });
    }
}

async function clearAllRecords() {
    if (confirm('Are you sure you want to clear all health records? This cannot be undone.')) {
        try {
            await apiCall('/api/records/clear', 'POST');
            loadHealthRecords();
        } catch (error) {
            console.error('Failed to clear records:', error);
        }
    }
}

// ============================================
// VITALS TRACKER (Feature #4)
// ============================================
function initVitalsModal() {
    switchVitalsTab('log');
    setDefaultDateTime();
}

function setDefaultDateTime() {
    const dtInput = document.getElementById('vitalsDateTime');
    if (dtInput) {
        const now = new Date();
        const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
        dtInput.value = local.toISOString().slice(0, 16);
    }
}

function switchVitalsTab(tab) {
    state.activeVitalsTab = tab;
    ['log', 'trends'].forEach(t => {
        const btn = document.getElementById(`vitalsTab-${t}`);
        const panel = document.getElementById(`vitalsPanel-${t}`);
        if (btn) btn.classList.toggle('active', t === tab);
        if (panel) panel.classList.toggle('hidden', t !== tab);
    });
    if (tab === 'trends') loadVitalsTrends(30);
}

async function submitVitals() {
    const getNumVal = id => {
        const el = document.getElementById(id);
        return el && el.value ? parseFloat(el.value) : null;
    };

    const timestamp = document.getElementById('vitalsDateTime')?.value;
    const tempUnit = document.getElementById('tempUnitToggle')?.value || 'F';
    const weightUnit = document.getElementById('weightUnitToggle')?.value || 'kg';

    const payload = {
        timestamp: timestamp ? new Date(timestamp).toISOString() : new Date().toISOString(),
        bp_systolic: getNumVal('vitalsBPSystolic'),
        bp_diastolic: getNumVal('vitalsBPDiastolic'),
        heart_rate: getNumVal('vitalsHeartRate'),
        blood_glucose: getNumVal('vitalsGlucose'),
        spo2: getNumVal('vitalsSPO2'),
        temperature: getNumVal('vitalsTemp'),
        weight: getNumVal('vitalsWeight'),
        temp_unit: tempUnit,
        weight_unit: weightUnit
    };

    // Check at least one value
    const hasData = ['bp_systolic','bp_diastolic','heart_rate','blood_glucose','spo2','temperature','weight']
        .some(k => payload[k] !== null);

    if (!hasData) {
        if (typeof showToast === 'function') showToast('Please enter at least one vital sign', 'warning');
        else alert('Please enter at least one vital sign');
        return;
    }

    const btn = document.getElementById('submitVitalsBtn');
    if (btn) { btn.disabled = true; btn.textContent = 'Saving...'; }

    try {
        const response = await apiCall('/api/vitals', 'POST', payload);
        displayVitalsAlerts(response.alerts || []);
        clearVitalsForm();
        if (typeof showToast === 'function') showToast('✅ Vitals saved!', 'success');
    } catch (error) {
        console.error('Save vitals failed:', error);
        if (typeof showToast === 'function') showToast('Failed to save vitals', 'error');
    } finally {
        if (btn) { btn.disabled = false; btn.textContent = 'Save Vitals'; }
    }
}

function displayVitalsAlerts(alerts) {
    const container = document.getElementById('vitalsAlerts');
    if (!container || !alerts.length) return;
    container.classList.remove('hidden');
    container.innerHTML = alerts.map(alert => {
        const cls = alert.status === 'normal' ? 'vitals-alert-normal'
            : alert.status === 'warning' ? 'vitals-alert-warning'
            : 'vitals-alert-danger';
        const icon = alert.status === 'normal' ? '✅'
            : alert.status === 'warning' ? '⚠️' : '🔴';
        return `<div class="${cls}">${icon} ${sanitize(alert.message)}</div>`;
    }).join('');
}

function clearVitalsForm() {
    ['vitalsBPSystolic','vitalsBPDiastolic','vitalsHeartRate','vitalsGlucose','vitalsSPO2','vitalsTemp','vitalsWeight']
        .forEach(id => {
            const el = document.getElementById(id);
            if (el) el.value = '';
        });
    setDefaultDateTime();
}

async function loadVitalsTrends(days) {
    const container = document.getElementById('vitalsChartsContainer');
    if (!container) return;

    // Update active button
    document.querySelectorAll('.vitals-days-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.days) === days);
    });

    try {
        const data = await apiCall(`/api/vitals?days=${days}`);
        renderVitalsCharts(data.readings || [], data.stats || {});
        renderVitalsStats(data.stats || {});
    } catch (error) {
        console.error('Failed to load vitals trends:', error);
    }
}

function renderVitalsCharts(readings, stats) {
    const metrics = [
        { key: 'heart_rate', label: 'Heart Rate', unit: 'bpm', color: '#ef4444', min: 60, max: 100 },
        { key: 'bp_systolic', label: 'BP Systolic', unit: 'mmHg', color: '#3b82f6', min: 90, max: 120 },
        { key: 'bp_diastolic', label: 'BP Diastolic', unit: 'mmHg', color: '#8b5cf6', min: 60, max: 80 },
        { key: 'spo2', label: 'SpO2', unit: '%', color: '#22c55e', min: 95, max: 100 },
        { key: 'blood_glucose', label: 'Blood Glucose', unit: 'mg/dL', color: '#f59e0b', min: 70, max: 100 },
        { key: 'temperature', label: 'Temperature', unit: '°F', color: '#f97316', min: 97, max: 99 }
    ];

    const container = document.getElementById('vitalsChartsContainer');
    if (!container) return;
    container.innerHTML = '';

    metrics.forEach(metric => {
        const metricReadings = readings.filter(r => r[metric.key] != null);
        if (metricReadings.length === 0) return;

        const chartId = `chart-${metric.key}`;
        const div = document.createElement('div');
        div.className = 'chart-container mb-6';
        div.innerHTML = `
            <h4 class="font-medium mb-3">${metric.label}</h4>
            <canvas id="${chartId}"></canvas>`;
        container.appendChild(div);

        const labels = metricReadings.map(r =>
            new Date(r.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
        ).reverse();
        const values = metricReadings.map(r => r[metric.key]).reverse();

        if (state.vitalsCharts[metric.key]) {
            state.vitalsCharts[metric.key].destroy();
        }

        const ctx = document.getElementById(chartId);
        if (!ctx || typeof Chart === 'undefined') return;

        state.vitalsCharts[metric.key] = new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label: metric.label,
                        data: values,
                        borderColor: metric.color,
                        backgroundColor: metric.color + '20',
                        tension: 0.3,
                        fill: true,
                        pointBackgroundColor: metric.color,
                        pointRadius: 4
                    },
                    // Normal range band — max line
                    {
                        label: 'Normal Max',
                        data: Array(labels.length).fill(metric.max),
                        borderColor: 'rgba(34,197,94,0.4)',
                        borderDash: [5, 5],
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: false
                    },
                    // Normal range band — min line
                    {
                        label: 'Normal Min',
                        data: Array(labels.length).fill(metric.min),
                        borderColor: 'rgba(34,197,94,0.4)',
                        borderDash: [5, 5],
                        borderWidth: 1,
                        pointRadius: 0,
                        fill: '-1',
                        backgroundColor: 'rgba(34,197,94,0.05)'
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => `${ctx.parsed.y} ${metric.unit}`
                        }
                    }
                },
                scales: {
                    x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: {
                        ticks: { color: '#94a3b8' },
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    }
                }
            }
        });
    });
}

function renderVitalsStats(stats) {
    const container = document.getElementById('vitalsStatsContainer');
    if (!container) return;

    const labels = {
        heart_rate: 'Heart Rate',
        bp_systolic: 'BP Systolic',
        bp_diastolic: 'BP Diastolic',
        spo2: 'SpO2',
        blood_glucose: 'Blood Glucose',
        temperature: 'Temperature',
        weight: 'Weight'
    };
    const units = {
        heart_rate: 'bpm', bp_systolic: 'mmHg', bp_diastolic: 'mmHg',
        spo2: '%', blood_glucose: 'mg/dL', temperature: '°F', weight: 'kg'
    };

    const activeMetrics = Object.entries(stats).filter(([k, v]) => v.count > 0);
    if (activeMetrics.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-sm text-center py-4">No vitals data for this period.</p>';
        return;
    }

    container.innerHTML = `
        <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
            ${activeMetrics.map(([key, s]) => `
                <div class="stat-card text-left">
                    <div class="text-xs text-gray-500 mb-2">${labels[key] || key}</div>
                    <div class="flex gap-3 text-sm">
                        <div><span class="text-gray-500">Min</span><br><span class="font-medium text-blue-400">${s.min ?? '--'}</span></div>
                        <div><span class="text-gray-500">Avg</span><br><span class="font-medium text-emerald-400">${s.avg ?? '--'}</span></div>
                        <div><span class="text-gray-500">Max</span><br><span class="font-medium text-amber-400">${s.max ?? '--'}</span></div>
                    </div>
                    <div class="text-xs text-gray-600 mt-1">${units[key] || ''} · ${s.count} readings</div>
                </div>`).join('')}
        </div>`;
}

function exportVitalsCSV() {
    apiCall('/api/vitals?days=365').then(data => {
        const readings = data.readings || [];
        if (!readings.length) {
            if (typeof showToast === 'function') showToast('No vitals data to export', 'warning');
            return;
        }

        const headers = ['Timestamp','BP Systolic','BP Diastolic','Heart Rate','Blood Glucose','SpO2','Temperature','Weight'];
        const rows = readings.map(r => [
            r.timestamp, r.bp_systolic ?? '', r.bp_diastolic ?? '',
            r.heart_rate ?? '', r.blood_glucose ?? '',
            r.spo2 ?? '', r.temperature ?? '', r.weight ?? ''
        ]);

        const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `vitals-${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);

        if (typeof showToast === 'function') showToast('📊 Vitals exported!', 'success');
    }).catch(() => {
        if (typeof showToast === 'function') showToast('Export failed', 'error');
    });
}

// ============================================
// PAGE LOAD
// ============================================
window.onload = function () {
    themeManager.init();
    if (localStorage.getItem('disclaimer_accepted') === 'true' ||
        localStorage.getItem('disclaimerAccepted') === 'true') {
        const modal = document.getElementById('disclaimerModal');
        modal.classList.add('hidden');
        modal.style.display = 'none';
        document.getElementById('mainApp').classList.remove('hidden');
        initApp();
    }
};

document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
        document.querySelectorAll('.modal.visible').forEach(modal => {
            if (modal.id !== 'disclaimerModal') {
                modal.classList.add('hidden');
                modal.classList.remove('visible');
                document.body.style.overflow = '';
            }
        });
    }
});

document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', function (e) {
        if (e.target === this && this.id !== 'disclaimerModal') {
            closeModal(this.id);
        }
    });
});