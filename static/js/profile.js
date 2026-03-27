/**
 * ProfileManager — Feature #14
 * Handles user profile stored in localStorage
 */

class ProfileManager {
    constructor() {
        this.STORAGE_KEY = 'healthcare_ai_profile';
        this.MANDATORY_FIELDS = ['name', 'age', 'gender'];
    }

    /** Load profile from localStorage */
    load() {
        try {
            const raw = localStorage.getItem(this.STORAGE_KEY);
            return raw ? JSON.parse(raw) : null;
        } catch (e) {
            console.error('ProfileManager: Failed to load profile', e);
            return null;
        }
    }

    /** Save profile data to localStorage */
    save(data) {
        try {
            const existing = this.load() || {};
            const now = new Date().toISOString();
            const merged = {
                ...existing,
                ...data,
                updatedAt: now
            };
            if (!merged.createdAt) {
                merged.createdAt = now;
            }
            // Ensure arrays are arrays
            ['knownConditions', 'allergies', 'medications'].forEach(key => {
                if (!Array.isArray(merged[key])) {
                    merged[key] = [];
                }
            });
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(merged));
            return true;
        } catch (e) {
            console.error('ProfileManager: Failed to save profile', e);
            return false;
        }
    }

    /** Get a specific field value */
    get(key) {
        const profile = this.load();
        return profile ? profile[key] : null;
    }

    /** Check if mandatory fields are filled */
    isComplete() {
        const profile = this.load();
        if (!profile) return false;
        return this.MANDATORY_FIELDS.every(field => {
            const val = profile[field];
            return val !== null && val !== undefined && val !== '';
        });
    }

    /** Get completeness status: 'empty', 'partial', 'complete' */
    getStatus() {
        const profile = this.load();
        if (!profile) return 'empty';

        const allFields = [
            'name', 'age', 'gender', 'height', 'weight',
            'bloodType', 'emergencyContact'
        ];
        const filledCount = allFields.filter(f => {
            const v = profile[f];
            return v !== null && v !== undefined && v !== '';
        }).length;

        if (filledCount === 0) return 'empty';
        if (this.isComplete() && filledCount >= 5) return 'complete';
        return 'partial';
    }

    /** Clear all profile data */
    clear() {
        localStorage.removeItem(this.STORAGE_KEY);
    }

    /** Get profile data for passing to API calls */
    getApiData() {
        const profile = this.load();
        if (!profile) return {};
        return {
            age: profile.age || null,
            gender: profile.gender || null,
            height: profile.height || null,
            weight: profile.weight || null,
            knownConditions: profile.knownConditions || [],
            allergies: profile.allergies || [],
            medications: profile.medications || [],
            bloodType: profile.bloodType || null
        };
    }
}

// Global instance
window.profileManager = new ProfileManager();

// ============================================================
// PROFILE UI FUNCTIONS
// ============================================================

/** Update the navbar profile dot color */
function updateProfileIndicator() {
    const dot = document.getElementById('profileStatusDot');
    if (!dot) return;

    const status = window.profileManager.getStatus();
    dot.className = 'w-3 h-3 rounded-full border-2 border-gray-900 absolute -top-0.5 -right-0.5 transition-colors';
    if (status === 'empty') {
        dot.classList.add('bg-red-500');
    } else if (status === 'partial') {
        dot.classList.add('bg-yellow-400');
    } else {
        dot.classList.add('bg-emerald-400');
    }
}

/** Open profile modal and populate form fields */
function openProfileModal() {
    openModal('profileModal');
    populateProfileForm();
}

/** Populate form from stored profile */
function populateProfileForm() {
    const profile = window.profileManager.load();
    if (!profile) return;

    const setVal = (id, val) => {
        const el = document.getElementById(id);
        if (el && val !== undefined && val !== null) el.value = val;
    };

    setVal('profileName', profile.name);
    setVal('profileDOB', profile.dob);
    setVal('profileGender', profile.gender);
    setVal('profileHeight', profile.height);
    setVal('profileWeight', profile.weight);
    setVal('profileBloodType', profile.bloodType);
    setVal('profileEmergencyContact', profile.emergencyContact);
    setVal('profileEmergencyPhone', profile.emergencyPhone);

    // Auto-calculate age from DOB
    if (profile.dob) {
        updateAgeFromDOB();
    }

    // Render tag fields
    renderTags('conditionsTags', profile.knownConditions || [], 'knownConditions');
    renderTags('allergiesTags', profile.allergies || [], 'allergies');
    renderTags('medicationsTags', profile.medications || [], 'medications');
}

/** Render tag chips inside a container */
function renderTags(containerId, items, fieldName) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = items.map(item => `
        <span class="inline-flex items-center gap-1 px-3 py-1 rounded-full text-xs
            bg-blue-500/20 border border-blue-500/30 text-blue-300">
            ${sanitize(item)}
            <button onclick="removeTag('${containerId}','${fieldName}','${sanitize(item)}')"
                class="hover:text-red-400 transition text-sm leading-none ml-1">&times;</button>
        </span>
    `).join('');
}

/** Add a tag when Enter is pressed in a tag input */
function handleTagInput(event, containerId, fieldName) {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    const input = event.target;
    const value = input.value.trim();
    if (!value) return;

    const profile = window.profileManager.load() || {};
    const arr = Array.isArray(profile[fieldName]) ? profile[fieldName] : [];
    if (!arr.includes(value)) {
        arr.push(value);
        window.profileManager.save({ [fieldName]: arr });
    }
    input.value = '';
    renderTags(containerId, arr, fieldName);
}

/** Remove a tag */
function removeTag(containerId, fieldName, value) {
    const profile = window.profileManager.load() || {};
    const arr = Array.isArray(profile[fieldName]) ? profile[fieldName] : [];
    const updated = arr.filter(item => item !== value);
    window.profileManager.save({ [fieldName]: updated });
    renderTags(containerId, updated, fieldName);
}

/** Auto-calculate age from date of birth */
function updateAgeFromDOB() {
    const dobInput = document.getElementById('profileDOB');
    const ageDisplay = document.getElementById('profileAgeDisplay');
    if (!dobInput || !ageDisplay) return;

    const dob = new Date(dobInput.value);
    if (isNaN(dob.getTime())) {
        ageDisplay.textContent = '';
        return;
    }
    const today = new Date();
    let age = today.getFullYear() - dob.getFullYear();
    const m = today.getMonth() - dob.getMonth();
    if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) age--;
    ageDisplay.textContent = age > 0 ? `Age: ${age}` : '';
}

/** Save profile from form */
function saveProfile() {
    const getVal = id => {
        const el = document.getElementById(id);
        return el ? el.value.trim() : '';
    };

    const dob = getVal('profileDOB');
    let age = null;
    if (dob) {
        const dobDate = new Date(dob);
        const today = new Date();
        age = today.getFullYear() - dobDate.getFullYear();
        const m = today.getMonth() - dobDate.getMonth();
        if (m < 0 || (m === 0 && today.getDate() < dobDate.getDate())) age--;
    }

    const existing = window.profileManager.load() || {};
    const profileData = {
        name: getVal('profileName'),
        dob: dob,
        age: age,
        gender: getVal('profileGender'),
        height: parseFloat(getVal('profileHeight')) || null,
        weight: parseFloat(getVal('profileWeight')) || null,
        bloodType: getVal('profileBloodType'),
        emergencyContact: getVal('profileEmergencyContact'),
        emergencyPhone: getVal('profileEmergencyPhone'),
        // preserve arrays
        knownConditions: existing.knownConditions || [],
        allergies: existing.allergies || [],
        medications: existing.medications || []
    };

    const success = window.profileManager.save(profileData);
    if (success) {
        showToast('✅ Profile saved successfully!', 'success');
        updateProfileIndicator();

        // Auto-fill other forms
        autoFillFormsFromProfile();
    } else {
        showToast('❌ Failed to save profile.', 'error');
    }
}

/** Auto-fill BMI calculator and symptom checker from profile */
function autoFillFormsFromProfile() {
    const profile = window.profileManager.load();
    if (!profile) return;

    // BMI Calculator
    if (profile.weight) {
        const bmiWeight = document.getElementById('bmiWeight');
        if (bmiWeight && !bmiWeight.value) bmiWeight.value = profile.weight;
    }
    if (profile.height) {
        const bmiHeight = document.getElementById('bmiHeight');
        if (bmiHeight && !bmiHeight.value) bmiHeight.value = profile.height;
    }
    if (profile.age) {
        const bmiAge = document.getElementById('bmiAge');
        if (bmiAge && !bmiAge.value) bmiAge.value = profile.age;
    }
    if (profile.gender) {
        const bmiGender = document.getElementById('bmiGender');
        if (bmiGender && !bmiGender.value) bmiGender.value = profile.gender;
    }

    // Symptom Checker age
    if (profile.age) {
        const patientAge = document.getElementById('patientAge');
        if (patientAge && !patientAge.value) patientAge.value = profile.age;
    }
}

/** Simple toast notification */
function showToast(message, type = 'success') {
    // Remove existing toast
    const existing = document.getElementById('toastNotification');
    if (existing) existing.remove();

    const colors = {
        success: 'from-emerald-500 to-green-600',
        error: 'from-red-500 to-red-600',
        info: 'from-blue-500 to-blue-600',
        warning: 'from-amber-500 to-yellow-600'
    };

    const toast = document.createElement('div');
    toast.id = 'toastNotification';
    toast.className = `fixed bottom-6 right-6 z-[9999] px-6 py-4 rounded-2xl
        bg-gradient-to-r ${colors[type] || colors.info}
        text-white font-medium shadow-2xl animate-slide-up
        flex items-center gap-3 max-w-sm`;
    toast.innerHTML = `<span>${message}</span>`;
    document.body.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('animate-fade-out');
        setTimeout(() => toast.remove(), 400);
    }, 3000);
}

// Initialize profile indicator on load
document.addEventListener('DOMContentLoaded', () => {
    updateProfileIndicator();
    // Auto-fill forms if profile exists
    if (window.profileManager.load()) {
        setTimeout(autoFillFormsFromProfile, 500);
    }
});