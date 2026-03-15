/**
 * Onboarding Tutorial Manager (Feature #17)
 * Provides a step-by-step walkthrough for first-time users
 */

class OnboardingManager {
    constructor() {
        this.currentStep = -1;
        this.overlay = null;
        this.spotlight = null;
        this.tooltip = null;
        this.welcomeModal = null;
        
        this.steps = [
            {
                type: 'welcome',
                title: 'Welcome to EazyHealth AI! 👋',
                description: 'Let us give you a quick tour of the app so you can make the most of our health tools. This will only take a minute!',
                icon: '🏥'
            },
            {
                type: 'highlight',
                target: 'featureSymptomChecker',
                title: 'AI Symptom Analyzer',
                description: 'Click here to check your symptoms. Our ML model uses Random Forest & Gradient Boosting to predict possible conditions based on what you\'re feeling.',
                position: 'bottom'
            },
            {
                type: 'highlight',
                target: 'featureBmiCalculator',
                title: 'BMI Calculator',
                description: 'Calculate your Body Mass Index with personalized health insights based on your age, gender, and activity level.',
                position: 'bottom'
            },
            {
                type: 'highlight',
                target: 'featureChatbot',
                title: 'Health Assistant Chatbot',
                description: 'Chat with our AI assistant for health information, wellness tips, and general guidance on common health topics.',
                position: 'bottom'
            },
            {
                type: 'highlight',
                target: 'featureRecords',
                title: 'Health Records',
                description: 'All your predictions and BMI calculations are saved here. Track your health history securely on your device.',
                position: 'bottom'
            },
            {
                type: 'highlight',
                target: 'emergencyBanner',
                fallbackTarget: 'featureEmergency',
                title: 'Emergency Detection',
                description: 'We automatically detect emergency symptoms and show alerts. If critical symptoms are found, you\'ll see a warning banner at the top.',
                position: 'bottom'
            },
            {
                type: 'highlight',
                target: 'disclaimerBanner',
                title: 'Important Disclaimer',
                description: 'Always remember: this tool is for educational purposes only. Our AI predictions have limitations — always consult a real healthcare professional.',
                position: 'top'
            },
            {
                type: 'final',
                title: 'You\'re All Set! 🎉',
                description: 'You\'re ready to explore EazyHealth AI! Remember, this tool is for educational use only. You can replay this tutorial anytime by clicking the "?" button in the header.',
                icon: '✅'
            }
        ];
    }
    
    start() {
        this.currentStep = 0;
        this.createOverlay();
        this.showStep(0);
    }
    
    next() {
        if (this.currentStep < this.steps.length - 1) {
            this.currentStep++;
            this.showStep(this.currentStep);
        } else {
            this.finish();
        }
    }
    
    prev() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.showStep(this.currentStep);
        }
    }
    
    skip() {
        this.finish();
    }
    
    finish() {
        localStorage.setItem('onboarding_complete', 'true');
        this.cleanup();
    }
    
    createOverlay() {
        // Create overlay
        this.overlay = document.createElement('div');
        this.overlay.className = 'onboarding-overlay';
        this.overlay.id = 'onboardingOverlay';
        document.body.appendChild(this.overlay);
        
        // Create spotlight
        this.spotlight = document.createElement('div');
        this.spotlight.className = 'onboarding-spotlight';
        this.spotlight.id = 'onboardingSpotlight';
        this.spotlight.style.display = 'none';
        document.body.appendChild(this.spotlight);
        
        // Create tooltip container
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'onboarding-tooltip';
        this.tooltip.id = 'onboardingTooltip';
        this.tooltip.style.display = 'none';
        document.body.appendChild(this.tooltip);
    }
    
    cleanup() {
        if (this.overlay) this.overlay.remove();
        if (this.spotlight) this.spotlight.remove();
        if (this.tooltip) this.tooltip.remove();
        if (this.welcomeModal) this.welcomeModal.remove();
        this.overlay = null;
        this.spotlight = null;
        this.tooltip = null;
        this.welcomeModal = null;
    }
    
    showStep(stepIndex) {
        const step = this.steps[stepIndex];
        
        // Clean up previous
        if (this.welcomeModal) {
            this.welcomeModal.remove();
            this.welcomeModal = null;
        }
        
        if (step.type === 'welcome' || step.type === 'final') {
            this.showWelcomeModal(step, stepIndex);
        } else {
            this.showHighlight(step, stepIndex);
        }
    }
    
    showWelcomeModal(step, stepIndex) {
        // Hide spotlight and tooltip
        if (this.spotlight) this.spotlight.style.display = 'none';
        if (this.tooltip) this.tooltip.style.display = 'none';
        
        // Show overlay
        if (this.overlay) this.overlay.style.display = 'block';
        
        this.welcomeModal = document.createElement('div');
        this.welcomeModal.className = 'onboarding-welcome-modal';
        
        const isFirst = step.type === 'welcome';
        const isFinal = step.type === 'final';
        
        this.welcomeModal.innerHTML = `
            <div class="onboarding-welcome-content">
                <div class="text-5xl mb-4">${step.icon}</div>
                ${this.renderProgressDots(stepIndex)}
                <h2 class="text-2xl font-bold mb-3" style="color: var(--text-primary)">${step.title}</h2>
                <p class="text-sm mb-6" style="color: var(--text-secondary); line-height: 1.6">${step.description}</p>
                <div class="flex gap-3 justify-center">
                    ${isFirst ? `
                        <button class="onboarding-btn onboarding-btn-skip" id="onboardingSkip">Skip Tour</button>
                        <button class="onboarding-btn onboarding-btn-next" id="onboardingNext">Start Tour →</button>
                    ` : ''}
                    ${isFinal ? `
                        <button class="onboarding-btn onboarding-btn-next" id="onboardingFinish">Get Started! 🚀</button>
                    ` : ''}
                </div>
            </div>
        `;
        
        document.body.appendChild(this.welcomeModal);
        
        // Bind events
        const skipBtn = document.getElementById('onboardingSkip');
        const nextBtn = document.getElementById('onboardingNext');
        const finishBtn = document.getElementById('onboardingFinish');
        
        if (skipBtn) skipBtn.addEventListener('click', () => this.skip());
        if (nextBtn) nextBtn.addEventListener('click', () => this.next());
        if (finishBtn) finishBtn.addEventListener('click', () => this.finish());
    }
    
    showHighlight(step, stepIndex) {
        let targetEl = document.getElementById(step.target);
        
        // Use fallback target if primary doesn't exist or is hidden
        if ((!targetEl || targetEl.offsetParent === null || targetEl.classList.contains('hidden')) && step.fallbackTarget) {
            targetEl = document.getElementById(step.fallbackTarget);
        }
        
        if (!targetEl || targetEl.offsetParent === null) {
            // Element not found, skip to next
            this.next();
            return;
        }
        
        // Scroll element into view
        targetEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
        // Slight delay for scroll to finish
        setTimeout(() => {
            const rect = targetEl.getBoundingClientRect();
            const padding = 8;
            
            // Position spotlight
            if (this.spotlight) {
                this.spotlight.style.display = 'block';
                this.spotlight.style.top = (rect.top - padding) + 'px';
                this.spotlight.style.left = (rect.left - padding) + 'px';
                this.spotlight.style.width = (rect.width + padding * 2) + 'px';
                this.spotlight.style.height = (rect.height + padding * 2) + 'px';
            }
            
            // Show overlay
            if (this.overlay) this.overlay.style.display = 'block';
            
            // Show tooltip
            this.showTooltip(step, stepIndex, rect);
        }, 400);
    }
    
    showTooltip(step, stepIndex, targetRect) {
        if (!this.tooltip) return;
        
        const totalSteps = this.steps.length;
        
        this.tooltip.style.display = 'block';
        this.tooltip.innerHTML = `
            <div class="onboarding-tooltip-content">
                ${this.renderProgressDots(stepIndex)}
                <div class="onboarding-step-counter">Step ${stepIndex} of ${totalSteps - 1}</div>
                <div class="onboarding-title">${step.title}</div>
                <div class="onboarding-description">${step.description}</div>
                <div class="onboarding-buttons">
                    <button class="onboarding-btn onboarding-btn-skip" id="onboardingSkip">Skip</button>
                    <div class="flex gap-2">
                        ${stepIndex > 1 ? `<button class="onboarding-btn onboarding-btn-prev" id="onboardingPrev">← Prev</button>` : ''}
                        <button class="onboarding-btn onboarding-btn-next" id="onboardingNext">${stepIndex === totalSteps - 2 ? 'Finish' : 'Next →'}</button>
                    </div>
                </div>
            </div>
        `;
        
        // Position tooltip
        this.positionTooltip(step.position || 'bottom', targetRect);
        
        // Bind events
        const skipBtn = document.getElementById('onboardingSkip');
        const prevBtn = document.getElementById('onboardingPrev');
        const nextBtn = document.getElementById('onboardingNext');
        
        if (skipBtn) skipBtn.addEventListener('click', () => this.skip());
        if (prevBtn) prevBtn.addEventListener('click', () => this.prev());
        if (nextBtn) nextBtn.addEventListener('click', () => this.next());
    }
    
    positionTooltip(position, targetRect) {
        if (!this.tooltip) return;
        
        const tooltipRect = this.tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const gap = 16;
        
        let top, left;
        
        if (position === 'bottom') {
            top = targetRect.bottom + gap;
            left = targetRect.left + (targetRect.width / 2) - (Math.min(360, viewportWidth * 0.9) / 2);
        } else if (position === 'top') {
            top = targetRect.top - gap - 220; // approximate tooltip height
            left = targetRect.left + (targetRect.width / 2) - (Math.min(360, viewportWidth * 0.9) / 2);
        }
        
        // Keep within viewport
        left = Math.max(10, Math.min(left, viewportWidth - Math.min(370, viewportWidth * 0.9 + 10)));
        top = Math.max(10, Math.min(top, viewportHeight - 250));
        
        this.tooltip.style.top = top + 'px';
        this.tooltip.style.left = left + 'px';
    }
    
    renderProgressDots(currentStep) {
        const totalSteps = this.steps.length;
        let dots = '<div class="onboarding-progress">';
        for (let i = 0; i < totalSteps; i++) {
            let cls = 'onboarding-dot';
            if (i === currentStep) cls += ' active';
            else if (i < currentStep) cls += ' completed';
            dots += `<div class="${cls}"></div>`;
        }
        dots += '</div>';
        return dots;
    }
}