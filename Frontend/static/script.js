const socket = io();

const signDisplay = document.getElementById('current-sign');
const sentenceDisplay = document.getElementById('sentence-display');

// Use built-in synthesis for TTS to avoid server lag
const synth = window.speechSynthesis;

socket.on('connect', () => {
    console.log("Connected to server");
});

socket.on('update_status', (data) => {
    const { sign, sentence } = data;

    // Update UI
    if (sign) {
        signDisplay.innerText = sign;
        signDisplay.style.color = '#10b981'; // Active color
    } else {
        signDisplay.innerText = "Listening...";
        signDisplay.style.color = '#94a3b8'; // Idle color
    }

    sentenceDisplay.innerText = sentence;
});

function sendCommand(action) {
    socket.emit('command', { action: action });
}

function speakText() {
    const text = sentenceDisplay.innerText;
    if (text) {
        const utterance = new SpeechSynthesisUtterance(text);
        synth.speak(utterance);
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Backspace') sendCommand('backspace');
    if (e.key === 'c') sendCommand('clear');
    if (e.key === 'Enter') speakText();
});

// ==============================
// SPEECH TO TEXT (Web Speech API)
// ==============================
const speechDisplay = document.getElementById('speech-display');
const micBtn = document.getElementById('mic-btn');

let recognition;
let isListening = false;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false; // Stop after one sentence
    recognition.interimResults = false;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
        isListening = true;
        micBtn.innerText = "Listening...";
        micBtn.classList.add('active');
        speechDisplay.innerText = "...";
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        const lang = document.getElementById('lang-select').value;

        if (lang === 'hi') {
            // Request translation from backend
            socket.emit('translate_now', { text: transcript, target: 'hi' });
        } else {
            speechDisplay.innerText = transcript;
        }
    };

    // Add listener for STT translation
    socket.on('stt_translation', (data) => {
        speechDisplay.innerText = data.translated;
    });

    recognition.onerror = (event) => {
        console.error("Speech Error:", event.error);
        stopListening();

        let errorMsg = "Error: " + event.error;
        if (event.error === 'network') {
            errorMsg = "Error: Network issue or Browser limitation.";
            // Brave detection logic
            if (navigator.brave !== undefined || navigator.userAgent.includes("Brave")) {
                errorMsg = "Error: Brave browser disables this feature. Please use Chrome or Edge.";
            } else {
                errorMsg = "Error: Network issue. Internet is required.";
            }
        } else if (event.error === 'not-allowed') {
            errorMsg = "Error: Microphone access denied.";
        } else if (event.error === 'no-speech') {
            errorMsg = "No speech detected. Try again.";
        }

        if (!navigator.onLine) {
            errorMsg += " (Offline)";
        }
        speechDisplay.innerText = errorMsg;
    };

    recognition.onend = () => {
        stopListening();
    };
} else {
    speechDisplay.innerText = "Browser not supported.";
    micBtn.disabled = true;
}

function toggleSpeech() {
    if (!recognition) return;

    if (isListening) {
        recognition.stop();
    } else {
        recognition.start();
    }
}

function copyToClipboard() {
    const text = sentenceDisplay.innerText;
    if (text) {
        navigator.clipboard.writeText(text).then(() => {
            const btn = document.querySelector('.btn-icon');
            btn.innerText = "✅";
            setTimeout(() => btn.innerText = "📋", 2000);
        });
    }
}

function copySpeechToClipboard() {
    const text = speechDisplay.innerText;
    if (text && text !== "...") {
        navigator.clipboard.writeText(text).then(() => {
            const btn = document.querySelector('.speech-header .btn-icon');
            btn.innerText = "✅";
            setTimeout(() => btn.innerText = "📋", 2000);
        });
    }
}

function changeLanguage() {
    const lang = document.getElementById('lang-select').value;
    socket.emit('set_language', { language: lang });
}

function stopListening() {
    isListening = false;
    micBtn.innerText = "Tap to Speak";
    micBtn.classList.remove('active');
}

// Initialize button text and language
if (micBtn) micBtn.innerText = "Tap to Speak";
changeLanguage(); // Sync initial language
