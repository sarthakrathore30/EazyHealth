/**
 * HealthCare AI - Main JavaScript
 * Handles UI interactions and API calls
 * Phase 1: Added Theme Manager, Disclaimer Flow, Health Tips, Prediction Feedback
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
    currentPredictionId: null
};

// Initialize the application
async function initApp() {
    console.log('Initializing HealthCare AI...');
    
    // Load initial data
    await Promise.all([
        loadModelMetrics(),
        loadSymptoms(),
        loadDiseases()
    ]);
    
    // Update UI
    updateStats();
    
    // Load daily health tip (Feature #19)
    loadHealthTip();
    
    // Start onboarding if first time (Feature #17)
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

// API Helper Functions
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    if (data) {
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

// Load model metrics from backend
async function loadModelMetrics() {
    try {
        const metrics = await apiCall('/api/model/metrics');
        state.modelMetrics = metrics;
        
        // Update model status
        const statusEl = document.getElementById('modelStatus');
        if (metrics.is_trained) {
            statusEl.innerHTML = `
                <span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                <span class="text-gray-400">ONLINE</span>
            `;
        }
        // Update accuracy display
        document.getElementById('accuracyRate').textContent = `${metrics.cross_val_mean}%`;
        
    } catch (error) {
        console.error('Failed to load model metrics:', error);
        document.getElementById('modelStatus').innerHTML = `
            <span class="w-2 h-2 rounded-full bg-red-500"></span>
            <span class="text-gray-400">Model Error</span>
        `;
    }
}

// Load symptoms from backend
async function loadSymptoms() {
    try {
        const data = await apiCall('/api/symptoms');
        state.allSymptoms = data.symptoms || [];
        state.categories = data.categories || {};
    } catch (error) {
        console.error('Failed to load symptoms:', error);
    }
}

// Load diseases from backend
async function loadDiseases() {
    try {
        const data = await apiCall('/api/diseases');
        state.diseases = data.diseases || {};
    } catch (error) {
        console.error('Failed to load diseases:', error);
    }
}

// Update statistics display
function updateStats() {
    document.getElementById('diseaseCount').textContent = Object.keys(state.diseases).length || '--';
    document.getElementById('symptomCount').textContent = state.allSymptoms.length || '--';
    document.getElementById('predictionCount').textContent = state.predictionCount;
}

// Sanitize text for display (XSS prevention)
function sanitize(str) {
    if (typeof str !== 'string') return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================
// DISCLAIMER FLOW (Feature #21 - Enhanced)
// ============================================
function onDisclaimerCheckboxChange() {
    const checkbox = document.getElementById('disclaimerCheckbox');
    const btn = document.getElementById('disclaimerAcceptBtn');
    if (checkbox.checked) {
        btn.disabled = false;
    } else {
        btn.disabled = true;
    }
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
        // Also set old key for backward compatibility
        localStorage.setItem('disclaimerAccepted', 'true');
        initApp();
    }, 300);
}

function showDisclaimer() {
    const modal = document.getElementById('disclaimerModal');
    modal.classList.remove('hidden', 'animate-fade-out');
    modal.style.display = 'flex';
}

// Emergency functions
function showEmergency(message) {
    const banner = document.getElementById('emergencyBanner');
    document.getElementById('emergencyMessage').textContent = message;
    banner.classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function hideEmergency() {
    document.getElementById('emergencyBanner').classList.add('hidden');
}

// Modal functions
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.remove('hidden');
    modal.classList.add('visible');
    document.body.style.overflow = 'hidden';
    
    if (modalId === 'symptomChecker') {
        initSymptomChecker();
    } else if (modalId === 'diseaseDb') {
        initDiseaseDatabase();
    } else if (modalId === 'records') {
        loadHealthRecords();
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    modal.classList.add('hidden');
    modal.classList.remove('visible');
    document.body.style.overflow = '';
}

// Initialize symptom checker
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
        `).join('')}
    `;
    
    renderSymptomTags();
    updateSelectedDisplay();
}

// Filter symptoms by category
function filterByCategory(category) {
    state.activeCategory = category;
    initSymptomChecker();
}

// Render symptom tags
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
        </button>
    `).join('');
}

// Toggle symptom selection
function toggleSymptom(symptom) {
    if (state.selectedSymptoms.has(symptom)) {
        state.selectedSymptoms.delete(symptom);
    } else {
        state.selectedSymptoms.add(symptom);
    }
    
    renderSymptomTags();
    updateSelectedDisplay();
}

// Update selected symptoms display
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
            </span>
        `).join('');
    }
}

// Generate a prediction ID
function generatePredictionId() {
    return 'pred_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// Analyze symptoms with backend ML model
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
    
    // Generate prediction ID for feedback
    state.currentPredictionId = generatePredictionId();
    
    try {
        const response = await apiCall('/api/predict', 'POST', {
            symptoms,
            duration,
            age
        });
        
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

// Display analysis results
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
                <p class="text-gray-400">Your symptoms don't strongly match any conditions in our database. Please consult a healthcare professional for proper evaluation.</p>
            </div>
        `;
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
            </div>
        `;
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
                    <span class="severity-badge severity-${pred.severity}">
                        ${pred.severity.toUpperCase()}
                    </span>
                    <span class="px-2 py-1 rounded-full text-xs bg-white/10 border border-white/20">
                        Urgency: ${pred.urgency}/5
                    </span>
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
            </div>
        `).join('')}
        
        <div class="bg-red-900/20 border border-red-500/30 rounded-xl p-4 mt-6">
            <p class="text-sm text-red-300 flex items-start gap-2">
                <span>⚠</span>
                <span><strong>Reminder:</strong> This is not a medical diagnosis. Please consult a qualified healthcare provider for proper evaluation and treatment.</span>
            </p>
        </div>
        
        <!-- Prediction Feedback Section (Feature #18) -->
        <div id="feedbackSection" class="feedback-card">
            <h4 class="text-lg font-semibold mb-4 flex items-center gap-2">
                <span>📝</span> Was this prediction helpful?
            </h4>
            
            <!-- Star Rating -->
            <div class="mb-4">
                <p class="text-sm text-gray-400 mb-2">Rate this prediction:</p>
                <div class="star-rating" id="starRating">
                    ${[1,2,3,4,5].map(i => `<span class="star" data-rating="${i}" onclick="setFeedbackRating(${i})" onmouseenter="hoverStars(${i})" onmouseleave="unhoverStars()">☆</span>`).join('')}
                </div>
            </div>
            
            <!-- Accuracy Toggle -->
            <div class="mb-4">
                <p class="text-sm text-gray-400 mb-2">Was this accurate for you?</p>
                <div class="flex gap-3">
                    <button onclick="setFeedbackAccuracy(true)" id="accuracyYes" class="feedback-accuracy-btn">👍 Yes</button>
                    <button onclick="setFeedbackAccuracy(false)" id="accuracyNo" class="feedback-accuracy-btn">👎 No</button>
                </div>
            </div>
            
            <!-- Comment -->
            <div class="mb-4">
                <textarea id="feedbackComment" class="feedback-textarea" rows="2" maxlength="200" placeholder="Tell us more (optional) — max 200 characters"></textarea>
                <p class="text-xs text-gray-500 mt-1 text-right"><span id="feedbackCharCount">0</span>/200</p>
            </div>
            
            <!-- Submit -->
            <button onclick="submitFeedback()" id="feedbackSubmitBtn" class="btn-primary px-6 py-2 text-sm">
                Submit Feedback
            </button>
        </div>
    `;
    
    // Add char counter listener
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
let feedbackState = {
    rating: 0,
    wasAccurate: null,
    comment: ''
};

function setFeedbackRating(rating) {
    feedbackState.rating = rating;
    const stars = document.querySelectorAll('#starRating .star');
    stars.forEach((star, idx) => {
        if (idx < rating) {
            star.classList.add('selected');
            star.textContent = '★';
        } else {
            star.classList.remove('selected');
            star.textContent = '☆';
        }
    });
}

function hoverStars(rating) {
    const stars = document.querySelectorAll('#starRating .star');
    stars.forEach((star, idx) => {
        if (idx < rating) {
            star.classList.add('hovered');
            star.textContent = '★';
        }
    });
}

function unhoverStars() {
    const stars = document.querySelectorAll('#starRating .star');
    stars.forEach((star, idx) => {
        star.classList.remove('hovered');
        if (idx < feedbackState.rating) {
            star.textContent = '★';
        } else {
            star.textContent = '☆';
        }
    });
}

function setFeedbackAccuracy(isAccurate) {
    feedbackState.wasAccurate = isAccurate;
    const yesBtn = document.getElementById('accuracyYes');
    const noBtn = document.getElementById('accuracyNo');
    
    yesBtn.classList.remove('selected-yes');
    noBtn.classList.remove('selected-no');
    
    if (isAccurate) {
        yesBtn.classList.add('selected-yes');
    } else {
        noBtn.classList.add('selected-no');
    }
}

async function submitFeedback() {
    const comment = document.getElementById('feedbackComment')?.value || '';
    feedbackState.comment = comment;
    
    if (feedbackState.rating === 0) {
        alert('Please select a star rating');
        return;
    }
    
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
        
        // Replace feedback form with thank you
        const section = document.getElementById('feedbackSection');
        section.innerHTML = `
            <div class="feedback-thank-you">
                <div class="text-4xl mb-3">🎉</div>
                <h4 class="text-lg font-semibold mb-2">Thank you for your feedback!</h4>
                <p class="text-sm text-gray-400">Your input helps us improve our predictions.</p>
            </div>
        `;
    } catch (error) {
        console.error('Feedback submission failed:', error);
        btn.disabled = false;
        btn.textContent = 'Submit Feedback';
        alert('Failed to submit feedback. Please try again.');
    }
    
    // Reset feedback state
    feedbackState = { rating: 0, wasAccurate: null, comment: '' };
}

// ============================================
// DAILY HEALTH TIP (Feature #19)
// ============================================
async function loadHealthTip() {
    // Check if tip was dismissed today
    const lastDismissed = localStorage.getItem('healthTipDismissed');
    if (lastDismissed) {
        const dismissedTime = parseInt(lastDismissed);
        const hoursSince = (Date.now() - dismissedTime) / (1000 * 60 * 60);
        if (hoursSince < 24) {
            return; // Still within 24-hour dismiss window
        }
    }
    
    try {
        const data = await apiCall('/api/health-tip');
        if (data && data.text) {
            const tipCard = document.getElementById('healthTipCard');
            const tipIcon = document.getElementById('tipIcon');
            const tipText = document.getElementById('tipText');
            const tipDetail = document.getElementById('tipDetail');
            const tipBadge = document.getElementById('tipCategoryBadge');
            
            tipIcon.textContent = data.icon || '💡';
            tipText.textContent = data.text;
            tipDetail.textContent = data.detail || '';
            tipBadge.textContent = data.category || 'Health';
            
            tipCard.classList.remove('hidden');
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
        // Small delay for transition
        requestAnimationFrame(() => {
            detail.classList.add('expanded');
        });
        arrow.style.transform = 'rotate(180deg)';
        text.textContent = 'Show Less';
    }
}

function dismissHealthTip() {
    const tipCard = document.getElementById('healthTipCard');
    tipCard.classList.add('animate-fade-out');
    setTimeout(() => {
        tipCard.classList.add('hidden');
    }, 300);
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

// BMI Calculator
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
        const response = await apiCall('/api/bmi', 'POST', {
            weight, height, age, gender, activity
        });
        displayBMIResults(response);
    } catch (error) {
        console.error('BMI calculation failed:', error);
        alert('Failed to calculate BMI. Please check your inputs.');
    }
}

// Display BMI results
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
            
            <p class="text-xs text-gray-500 mt-4 text-center">
                ⚠️ BMI is a general indicator and may not account for muscle mass, bone density, or other factors.
            </p>
        </div>
    `;
}

// Chatbot functions
async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    const chatContainer = document.getElementById('chatMessages');
    
    chatContainer.innerHTML += `
        <div class="chat-message user">
            ${sanitize(message)}
        </div>
    `;
    
    input.value = '';
    
    chatContainer.innerHTML += `
        <div class="chat-message bot" id="loadingMsg">
            <div class="loading-spinner"></div>
        </div>
    `;
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    try {
        const response = await apiCall('/api/chat', 'POST', { message });
        document.getElementById('loadingMsg').remove();
        
        chatContainer.innerHTML += `
            <div class="chat-message bot">
                ${response.response}
            </div>
        `;
    } catch (error) {
        document.getElementById('loadingMsg').remove();
        chatContainer.innerHTML += `
            <div class="chat-message bot bg-red-900/30 border-red-500/30">
                Sorry, I encountered an error. Please try again.
            </div>
        `;
    }
    
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Disease Database
function initDiseaseDatabase() {
    const categories = [...new Set(Object.values(state.diseases).map(d => d.category))].sort();
    
    const categoryFilters = document.getElementById('diseaseCategoryFilters');
    categoryFilters.innerHTML = `
        <button onclick="filterDiseases(null, event)" class="category-btn active">All</button>
        ${categories.map(cat => `
            <button onclick="filterDiseases('${sanitize(cat)}', event)" class="category-btn">
                ${sanitize(cat)}
            </button>
        `).join('')}
    `;
    
    displayDiseases(Object.entries(state.diseases));
}

function filterDiseases(category, event) {
    const filtered = Object.entries(state.diseases).filter(([name, info]) => 
        !category || info.category === category
    );
    displayDiseases(filtered);
    
    document.querySelectorAll('#diseaseCategoryFilters button').forEach(btn => {
        btn.classList.remove('active');
    });
    if (event && event.target) {
        event.target.classList.add('active');
    }
}

function searchDiseases() {
    const query = document.getElementById('diseaseSearch').value.toLowerCase();
    const filtered = Object.entries(state.diseases).filter(([name, info]) =>
        name.toLowerCase().includes(query) ||
        info.symptoms.some(s => s.toLowerCase().includes(query)) ||
        info.category.toLowerCase().includes(query) ||
        (info.description && info.description.toLowerCase().includes(query))
    );
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
                <span class="severity-badge severity-${info.severity}">
                    ${info.severity.toUpperCase()}
                </span>
            </div>
            <p class="text-gray-300 text-sm mb-3">${sanitize(info.description || '')}</p>
            <div class="text-sm space-y-2">
                <p><strong class="text-blue-400">Symptoms:</strong> <span class="text-gray-400">${info.symptoms.slice(0, 6).map(s => sanitize(s)).join(', ')}${info.symptoms.length > 6 ? '...' : ''}</span></p>
                <p><strong class="text-emerald-400">Recommendations:</strong> <span class="text-gray-400">${sanitize(info.recommendations)}</span></p>
            </div>
        </div>
    `).join('');
}

// Health Records
async function loadHealthRecords() {
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
                    <div class="font-medium text-blue-400">${sanitize(record.disease)}</div>
                    <div class="text-xs text-gray-400 mt-1">Symptoms: ${record.symptoms.map(s => sanitize(s)).join(', ')}</div>
                </div>
            `).join('');
        }
        
        if (!data.bmi_records || data.bmi_records.length === 0) {
            bmiContainer.innerHTML = '<p class="text-gray-500 text-sm py-4">No BMI records yet.</p>';
        } else {
            bmiContainer.innerHTML = data.bmi_records.map(record => `
                <div class="record-item">
                    <div class="text-xs text-gray-500 mb-1">${record.timestamp}</div>
                    <div class="font-medium">BMI: <span class="text-emerald-400">${record.bmi}</span> <span class="text-gray-400">(${sanitize(record.category)})</span></div>
                </div>
            `).join('');
        }
        
    } catch (error) {
        console.error('Failed to load records:', error);
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

// Check if disclaimer was accepted on page load
window.onload = function() {
    // Apply theme immediately
    themeManager.init();
    
    if (localStorage.getItem('disclaimer_accepted') === 'true' || localStorage.getItem('disclaimerAccepted') === 'true') {
        const modal = document.getElementById('disclaimerModal');
        modal.classList.add('hidden');
        modal.style.display = 'none';
        document.getElementById('mainApp').classList.remove('hidden');
        initApp();
    }
};

// Close modal on escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const modals = document.querySelectorAll('.modal.visible');
        modals.forEach(modal => {
            if (modal.id !== 'disclaimerModal') {
                modal.classList.add('hidden');
                modal.classList.remove('visible');
                document.body.style.overflow = '';
            }
        });
    }
});

// Prevent body scroll when modal is open
document.querySelectorAll('.modal').forEach(modal => {
    modal.addEventListener('click', function(e) {
        if (e.target === this && this.id !== 'disclaimerModal') {
            closeModal(this.id);
        }
    });
});