/**
 * GoalManager — Feature #20: Health Goal Setter
 * Frontend-only, localStorage-based
 */

class GoalManager {
    constructor() {
        this.STORAGE_KEY = 'healthcare_ai_goals';
    }

    /** Load all goals from localStorage */
    _load() {
        try {
            const raw = localStorage.getItem(this.STORAGE_KEY);
            return raw ? JSON.parse(raw) : [];
        } catch (e) {
            console.error('GoalManager: load error', e);
            return [];
        }
    }

    /** Persist goals to localStorage */
    _save(goals) {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(goals));
            return true;
        } catch (e) {
            console.error('GoalManager: save error', e);
            return false;
        }
    }

    /** Generate a simple UUID */
    _uuid() {
        return 'goal_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    /** Get all goals */
    getGoals() {
        return this._load();
    }

    /** Get active (incomplete) goals */
    getActiveGoals() {
        return this._load().filter(g => !g.isComplete);
    }

    /** Get completed goals */
    getCompletedGoals() {
        return this._load().filter(g => g.isComplete);
    }

    /**
     * Add a new goal
     * goal: { type, title, target, unit, currentValue, startValue, deadline }
     */
    addGoal(goal) {
        const goals = this._load();
        const now = new Date().toISOString();
        const newGoal = {
            id: this._uuid(),
            type: goal.type || 'custom',
            title: goal.title || 'My Goal',
            target: parseFloat(goal.target) || 0,
            unit: goal.unit || '',
            currentValue: parseFloat(goal.currentValue) || 0,
            startValue: parseFloat(goal.currentValue) || 0,
            deadline: goal.deadline || null,
            history: [],
            isComplete: false,
            createdAt: now
        };

        // Log the starting value in history
        if (newGoal.currentValue !== 0) {
            newGoal.history.push({ date: now, value: newGoal.currentValue });
        }

        goals.push(newGoal);
        this._save(goals);
        return newGoal;
    }

    /**
     * Update progress for a goal
     * id: goal id
     * value: new current value (number)
     */
    updateProgress(id, value) {
        const goals = this._load();
        const idx = goals.findIndex(g => g.id === id);
        if (idx === -1) return null;

        const goal = goals[idx];
        const numVal = parseFloat(value);
        if (isNaN(numVal)) return null;

        goal.currentValue = numVal;
        goal.history.push({ date: new Date().toISOString(), value: numVal });

        // Check completion
        const achieved = this._checkAchieved(goal);
        if (achieved && !goal.isComplete) {
            goal.isComplete = true;
            goal.completedAt = new Date().toISOString();
        }

        goals[idx] = goal;
        this._save(goals);
        return goal;
    }

    /**
     * Delete a goal by ID
     */
    deleteGoal(id) {
        const goals = this._load();
        const filtered = goals.filter(g => g.id !== id);
        if (filtered.length === goals.length) return false;
        this._save(filtered);
        return true;
    }

    /**
     * Check if a goal is achieved based on type logic
     */
    _checkAchieved(goal) {
        switch (goal.type) {
            case 'weight':
                // achieved if currentValue <= target (losing weight)
                return goal.currentValue <= goal.target;
            case 'water':
            case 'steps':
            case 'sleep':
            case 'custom':
                return goal.currentValue >= goal.target;
            case 'medication':
                // medication is daily — check if today is logged
                return goal.currentValue >= goal.target;
            default:
                return goal.currentValue >= goal.target;
        }
    }

    /**
     * Check achievements for a goal and return { isAchieved }
     */
    checkAchievements(id) {
        const goals = this._load();
        const goal = goals.find(g => g.id === id);
        if (!goal) return { isAchieved: false };
        return { isAchieved: this._checkAchieved(goal) };
    }

    /**
     * Get progress percentage for a goal (0–100)
     */
    getProgress(goal) {
        if (!goal.target || goal.target === 0) return 0;

        let pct;
        if (goal.type === 'weight') {
            // Progress: how much of the gap from start to target have we closed
            const totalGap = goal.startValue - goal.target;
            if (totalGap <= 0) return 100;
            const closedGap = goal.startValue - goal.currentValue;
            pct = (closedGap / totalGap) * 100;
        } else {
            pct = (goal.currentValue / goal.target) * 100;
        }
        return Math.min(100, Math.max(0, Math.round(pct)));
    }

    /**
     * Get days remaining until deadline
     */
    getDaysRemaining(goal) {
        if (!goal.deadline) return null;
        const now = new Date();
        const deadline = new Date(goal.deadline);
        const diff = Math.ceil((deadline - now) / (1000 * 60 * 60 * 24));
        return diff;
    }
}

// Global instance
window.goalManager = new GoalManager();

// ============================================================
// GOALS UI
// ============================================================

// Track which goal is being logged
let _currentLogGoalId = null;

/** Open goals modal */
function openGoalsModal() {
    openModal('goalsModal');
    renderGoalsModal();
}

/** Render entire goals modal content */
function renderGoalsModal() {
    renderActiveGoals();
    renderCompletedGoals();
    renderGoalTypeSelector();
}

/** Render active goals list */
function renderActiveGoals() {
    const container = document.getElementById('activeGoalsList');
    if (!container) return;

    const goals = window.goalManager.getActiveGoals();

    if (goals.length === 0) {
        container.innerHTML = `
            <div class="text-center py-8 text-gray-500">
                <div class="text-4xl mb-3">🎯</div>
                <p>No active goals yet. Add one below!</p>
            </div>`;
        return;
    }

    container.innerHTML = goals.map(goal => {
        const pct = window.goalManager.getProgress(goal);
        const daysLeft = window.goalManager.getDaysRemaining(goal);
        const icons = getGoalIcon(goal.type);

        // Color shifts green as you near target
        let barColor = 'from-blue-500 to-blue-600';
        if (pct >= 75) barColor = 'from-emerald-500 to-green-600';
        else if (pct >= 40) barColor = 'from-amber-500 to-yellow-500';

        const daysText = daysLeft !== null
            ? (daysLeft >= 0 ? `${daysLeft} days left` : `${Math.abs(daysLeft)} days overdue`)
            : 'No deadline';

        return `
        <div class="goal-card result-card mb-4" id="goal-${goal.id}">
            <div class="flex items-start justify-between mb-3">
                <div class="flex items-center gap-3">
                    <span class="text-2xl">${icons}</span>
                    <div>
                        <h4 class="font-semibold">${sanitize(goal.title)}</h4>
                        <span class="text-xs text-gray-500 capitalize">${goal.type} goal</span>
                    </div>
                </div>
                <button onclick="deleteGoal('${goal.id}')"
                    class="text-gray-600 hover:text-red-400 transition text-lg leading-none"
                    title="Delete goal">&times;</button>
            </div>

            <!-- Progress Bar -->
            <div class="mb-2">
                <div class="flex justify-between text-sm mb-1">
                    <span class="text-gray-400">Progress</span>
                    <span class="font-medium text-emerald-400">${pct}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill bg-gradient-to-r ${barColor} transition-all duration-700"
                        style="width: ${pct}%"></div>
                </div>
            </div>

            <!-- Stats Row -->
            <div class="flex flex-wrap gap-4 text-sm my-3">
                <div>
                    <span class="text-gray-500">Current: </span>
                    <span class="font-medium">${goal.currentValue} ${sanitize(goal.unit)}</span>
                </div>
                <div>
                    <span class="text-gray-500">Target: </span>
                    <span class="font-medium">${goal.target} ${sanitize(goal.unit)}</span>
                </div>
                <div>
                    <span class="text-gray-500">${daysText}</span>
                </div>
            </div>

            <!-- Log Today Button -->
            <div id="logSection-${goal.id}">
                <button onclick="showLogInput('${goal.id}')"
                    class="text-sm px-4 py-2 rounded-lg bg-blue-500/20 border border-blue-500/30
                    text-blue-300 hover:bg-blue-500/30 transition">
                    + Log Today
                </button>
            </div>
        </div>`;
    }).join('');
}

/** Show inline log input for a goal */
function showLogInput(goalId) {
    _currentLogGoalId = goalId;
    const goal = window.goalManager.getGoals().find(g => g.id === goalId);
    if (!goal) return;

    const section = document.getElementById(`logSection-${goalId}`);
    if (!section) return;

    const isMedication = goal.type === 'medication';

    section.innerHTML = isMedication ? `
        <div class="flex gap-2 mt-2">
            <button onclick="logGoalProgress('${goalId}', 1)"
                class="btn-primary px-4 py-2 text-sm">✅ Taken Today</button>
            <button onclick="cancelLog('${goalId}')"
                class="px-4 py-2 text-sm rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition">
                Cancel</button>
        </div>` : `
        <div class="flex gap-2 mt-2 items-center">
            <input type="number" id="logInput-${goalId}"
                class="input-field w-32 text-sm py-2"
                placeholder="${goal.unit}"
                step="any">
            <button onclick="logGoalProgress('${goalId}', null)"
                class="btn-primary px-4 py-2 text-sm">Log</button>
            <button onclick="cancelLog('${goalId}')"
                class="px-4 py-2 text-sm rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 transition">
                Cancel</button>
        </div>`;
}

/** Cancel log input */
function cancelLog(goalId) {
    renderActiveGoals();
}

/** Log progress for a goal */
function logGoalProgress(goalId, overrideValue) {
    let value = overrideValue;
    if (value === null) {
        const input = document.getElementById(`logInput-${goalId}`);
        if (!input || !input.value) {
            showToast('Please enter a value', 'warning');
            return;
        }
        value = parseFloat(input.value);
        if (isNaN(value)) {
            showToast('Please enter a valid number', 'warning');
            return;
        }
    }

    const updated = window.goalManager.updateProgress(goalId, value);
    if (!updated) return;

    if (updated.isComplete) {
        triggerGoalAchieved(updated);
    } else {
        showToast('✅ Progress logged!', 'success');
    }

    renderActiveGoals();
    renderCompletedGoals();
}

/** Handle goal achievement celebration */
function triggerGoalAchieved(goal) {
    showToast(`🏆 Goal Achieved! "${goal.title}" — Well done!`, 'success');
    launchConfetti();
    renderActiveGoals();
    renderCompletedGoals();
}

/** Delete a goal */
function deleteGoal(goalId) {
    if (!confirm('Delete this goal?')) return;
    window.goalManager.deleteGoal(goalId);
    renderActiveGoals();
    renderCompletedGoals();
    showToast('Goal deleted', 'info');
}

/** Render completed goals accordion */
function renderCompletedGoals() {
    const container = document.getElementById('completedGoalsList');
    if (!container) return;

    const goals = window.goalManager.getCompletedGoals();
    if (goals.length === 0) {
        container.innerHTML = '<p class="text-gray-600 text-sm py-2">No completed goals yet.</p>';
        return;
    }

    container.innerHTML = goals.map(goal => {
        const completedDate = goal.completedAt
            ? new Date(goal.completedAt).toLocaleDateString()
            : 'Unknown';
        const startDate = new Date(goal.createdAt).toLocaleDateString();
        const icon = getGoalIcon(goal.type);

        let duration = '';
        if (goal.completedAt && goal.createdAt) {
            const days = Math.ceil(
                (new Date(goal.completedAt) - new Date(goal.createdAt)) / (1000 * 60 * 60 * 24)
            );
            duration = `${days} day${days !== 1 ? 's' : ''}`;
        }

        return `
        <div class="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 mb-3">
            <div class="flex items-center gap-3">
                <span class="text-xl">${icon}</span>
                <div class="flex-1">
                    <div class="font-medium text-emerald-300">${sanitize(goal.title)}</div>
                    <div class="text-xs text-gray-500">
                        Achieved ${completedDate}
                        ${duration ? ` · Took ${duration}` : ''}
                    </div>
                </div>
                <span class="text-emerald-400 text-lg">🏆</span>
            </div>
        </div>`;
    }).join('');
}

/** Render goal type icon grid selector */
function renderGoalTypeSelector() {
    // Already in HTML — just ensure the form is ready
}

/** Get icon for goal type */
function getGoalIcon(type) {
    const icons = {
        weight: '⚖️',
        water: '💧',
        steps: '👟',
        sleep: '😴',
        medication: '💊',
        custom: '🎯'
    };
    return icons[type] || '🎯';
}

/** Selected goal type state */
let _selectedGoalType = null;

/** Select a goal type */
function selectGoalType(type) {
    _selectedGoalType = type;

    // Update UI selection
    document.querySelectorAll('.goal-type-btn').forEach(btn => {
        btn.classList.remove('border-blue-500', 'bg-blue-500/20');
        btn.classList.add('border-white/10', 'bg-white/5');
    });
    const selected = document.getElementById(`goalType-${type}`);
    if (selected) {
        selected.classList.remove('border-white/10', 'bg-white/5');
        selected.classList.add('border-blue-500', 'bg-blue-500/20');
    }

    // Render dynamic form
    renderGoalForm(type);
}

/** Render dynamic goal form based on type */
function renderGoalForm(type) {
    const container = document.getElementById('goalFormContainer');
    if (!container) return;

    const defaults = {
        weight: { title: 'Weight Goal', unit: 'kg', targetLabel: 'Target Weight (kg)', currentLabel: 'Current Weight (kg)' },
        water: { title: 'Daily Water Intake', unit: 'glasses', targetLabel: 'Daily Target (glasses)', currentLabel: 'Current Daily Intake (glasses)', defaultTarget: 8 },
        steps: { title: 'Daily Steps', unit: 'steps', targetLabel: 'Daily Step Target', currentLabel: 'Current Daily Steps', defaultTarget: 10000 },
        sleep: { title: 'Sleep Goal', unit: 'hours/night', targetLabel: 'Target Hours per Night', currentLabel: 'Current Hours per Night', defaultTarget: 8 },
        medication: { title: 'Medication Adherence', unit: 'doses/day', targetLabel: 'Daily Doses Required', currentLabel: 'Current Adherence (doses)', defaultTarget: 1 },
        custom: { title: '', unit: '', targetLabel: 'Target Value', currentLabel: 'Starting Value' }
    };

    const cfg = defaults[type] || defaults.custom;

    container.innerHTML = `
        <div class="space-y-4 mt-4 p-4 bg-white/5 rounded-xl border border-white/10 animate-slide-up">
            ${type === 'custom' ? `
            <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">Goal Title</label>
                <input type="text" id="newGoalTitle" class="input-field" placeholder="e.g. Run 5K">
            </div>
            <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">Unit</label>
                <input type="text" id="newGoalUnit" class="input-field" placeholder="e.g. km, minutes...">
            </div>` : `
            <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">Goal Title</label>
                <input type="text" id="newGoalTitle" class="input-field" value="${cfg.title}">
            </div>`}

            <div class="grid grid-cols-2 gap-4">
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">${cfg.currentLabel}</label>
                    <input type="number" id="newGoalCurrent" class="input-field" placeholder="0" step="any">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-300 mb-2">${cfg.targetLabel}</label>
                    <input type="number" id="newGoalTarget" class="input-field"
                        placeholder="${cfg.defaultTarget || ''}" step="any"
                        value="${cfg.defaultTarget || ''}">
                </div>
            </div>

            <div>
                <label class="block text-sm font-medium text-gray-300 mb-2">Deadline (optional)</label>
                <input type="date" id="newGoalDeadline" class="input-field"
                    min="${new Date().toISOString().split('T')[0]}">
            </div>

            <button onclick="saveNewGoal()"
                class="w-full btn-primary py-3 mt-2">
                Save Goal
            </button>
        </div>`;
}

/** Save a new goal from form */
function saveNewGoal() {
    if (!_selectedGoalType) {
        showToast('Please select a goal type', 'warning');
        return;
    }

    const title = document.getElementById('newGoalTitle')?.value.trim();
    const target = parseFloat(document.getElementById('newGoalTarget')?.value);
    const current = parseFloat(document.getElementById('newGoalCurrent')?.value) || 0;
    const deadline = document.getElementById('newGoalDeadline')?.value || null;
    const unit = _selectedGoalType === 'custom'
        ? document.getElementById('newGoalUnit')?.value.trim()
        : getGoalDefaultUnit(_selectedGoalType);

    if (!title) { showToast('Please enter a goal title', 'warning'); return; }
    if (isNaN(target) || target <= 0) { showToast('Please enter a valid target value', 'warning'); return; }

    window.goalManager.addGoal({
        type: _selectedGoalType,
        title,
        target,
        currentValue: current,
        unit,
        deadline
    });

    showToast('🎯 Goal added!', 'success');

    // Reset
    _selectedGoalType = null;
    document.querySelectorAll('.goal-type-btn').forEach(btn => {
        btn.classList.remove('border-blue-500', 'bg-blue-500/20');
        btn.classList.add('border-white/10', 'bg-white/5');
    });
    const formContainer = document.getElementById('goalFormContainer');
    if (formContainer) formContainer.innerHTML = '';

    renderActiveGoals();
    renderCompletedGoals();
}

/** Get default unit for goal type */
function getGoalDefaultUnit(type) {
    const units = { weight: 'kg', water: 'glasses', steps: 'steps', sleep: 'hours', medication: 'doses', custom: '' };
    return units[type] || '';
}

/** Confetti animation using canvas-confetti CDN */
function launchConfetti() {
    if (typeof confetti !== 'undefined') {
        confetti({
            particleCount: 150,
            spread: 90,
            origin: { y: 0.6 },
            colors: ['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444']
        });
    } else {
        // Fallback: CSS confetti burst effect
        console.log('🎉 Goal achieved!');
    }
}