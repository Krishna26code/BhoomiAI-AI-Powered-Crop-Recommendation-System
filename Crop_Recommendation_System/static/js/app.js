// ===================== Language toggle =====================
function setLang(lang) {
  document.documentElement.setAttribute('data-lang', lang);
  document.getElementById('langEn').classList.toggle('active', lang === 'en');
  document.getElementById('langHi').classList.toggle('active', lang === 'hi');
  localStorage.setItem('bhoomi_lang', lang);
}
(function initLang() {
  const saved = localStorage.getItem('bhoomi_lang') || 'en';
  setLang(saved);
})();

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2200);
}

// ===================== Prediction form =====================
const predictForm = document.getElementById('predictForm');
const resultBox = document.getElementById('resultBox');
let lastResult = null;

predictForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('predictBtn');
  btn.disabled = true;
  const fd = new FormData(predictForm);
  const payload = {};
  for (const [k, v] of fd.entries()) payload[k] = parseFloat(v);

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Prediction failed');
    renderResult(data);
    lastResult = data;
    saveToHistory(data);
    loadAnalytics();
  } catch (err) {
    toast(err.message);
  } finally {
    btn.disabled = false;
  }
});

function renderResult(data) {
  document.getElementById('resEmoji').textContent = data.emoji;
  document.getElementById('resCrop').textContent = data.crop;
  document.getElementById('resConf').textContent = data.confidence;

  // gauge: circumference = 2*pi*42 ≈ 264
  const circumference = 264;
  const offset = circumference - (circumference * data.confidence / 100);
  const arc = document.getElementById('gaugeArc');
  arc.setAttribute('stroke-dasharray', circumference);
  arc.setAttribute('stroke-dashoffset', offset);
  document.getElementById('gaugeText').textContent = Math.round(data.confidence) + '%';

  const top3Box = document.getElementById('top3Box');
  top3Box.innerHTML = data.top3.map(t => `<span class="top3-chip">${t.crop} · ${t.confidence}%</span>`).join('');

  const cal = data.calendar || {};
  const calMini = document.getElementById('calMini');
  if (cal.season) {
    calMini.innerHTML = `<b>${data.crop}</b> — ${cal.season}<br>
      Sow: ${cal.sow} &nbsp;|&nbsp; Harvest: ${cal.harvest}`;
  } else {
    calMini.innerHTML = '';
  }

  resultBox.classList.add('show');
  resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ===================== PDF report =====================
document.getElementById('downloadPdfBtn').addEventListener('click', async () => {
  if (!lastResult) return;
  const res = await fetch('/api/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(lastResult),
  });
  if (!res.ok) { toast('Could not generate PDF'); return; }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `crop_report_${lastResult.crop}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
});

// ===================== History (localStorage, last 10) =====================
const HISTORY_KEY = 'bhoomi_history';

function saveToHistory(data) {
  const list = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  list.unshift({
    crop: data.crop, emoji: data.emoji, confidence: data.confidence,
    time: new Date().toLocaleString(),
  });
  localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 10)));
  renderHistory();
}

function renderHistory() {
  const list = JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  const el = document.getElementById('historyList');
  if (!list.length) {
    el.innerHTML = `<li class="empty-note"><span data-en>No predictions yet.</span><span data-hi>Abhi tak koi prediction nahi.</span></li>`;
    return;
  }
  el.innerHTML = list.map(item => `
    <li>
      <span class="history-crop">${item.emoji} ${item.crop}</span>
      <span class="history-time">${item.confidence}% · ${item.time}</span>
    </li>`).join('');
}
renderHistory();

// ===================== Analytics dashboard =====================
let analyticsChart = null;
async function loadAnalytics() {
  try {
    const res = await fetch('/api/analytics');
    const data = await res.json();
    document.getElementById('statTotal').textContent = data.total_predictions;
    document.getElementById('statTop').textContent = data.by_crop[0] ? data.by_crop[0].crop : '—';

    const labels = data.by_crop.map(c => c.crop);
    const counts = data.by_crop.map(c => c.count);

    const ctx = document.getElementById('analyticsChart');
    if (analyticsChart) analyticsChart.destroy();
    analyticsChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Times recommended',
          data: counts,
          backgroundColor: '#1F6F43',
          borderRadius: 6,
        }],
      },
      options: {
        plugins: { legend: { display: false } },
        scales: { x: { ticks: { autoSkip: false, maxRotation: 60, minRotation: 45 } } },
      },
    });
  } catch (err) {
    console.warn('Analytics load failed', err);
  }
}
loadAnalytics();

// ===================== Weather widget =====================
async function fetchWeather(params) {
  const box = document.getElementById('weatherBox');
  box.innerHTML = '<span class="empty-note">Loading…</span>';
  try {
    const qs = new URLSearchParams(params).toString();
    const res = await fetch(`/api/weather?${qs}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Weather unavailable');
    box.innerHTML = `
      <img class="weather-icon" src="https://openweathermap.org/img/wn/${data.icon}@2x.png" alt="">
      <div>
        <div class="weather-temp">${Math.round(data.temperature)}°C</div>
        <div class="weather-meta">${data.city} · ${data.description}<br>Humidity: ${data.humidity}% · Wind: ${data.wind_speed} m/s</div>
      </div>`;
    window._lastWeather = data;
  } catch (err) {
    box.innerHTML = `<span class="empty-note">${err.message}</span>`;
  }
}

document.getElementById('weatherSearchBtn').addEventListener('click', () => {
  const city = document.getElementById('weatherCity').value.trim();
  if (city) fetchWeather({ city });
});
document.getElementById('weatherCity').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') document.getElementById('weatherSearchBtn').click();
});

document.getElementById('useWeatherBtn').addEventListener('click', () => {
  if (!window._lastWeather) { toast('Fetch weather first →'); return; }
  const w = window._lastWeather;
  predictForm.temperature.value = w.temperature;
  predictForm.humidity.value = w.humidity;
  toast('Weather values filled in ✓');
});

if (navigator.geolocation) {
  navigator.geolocation.getCurrentPosition(
    (pos) => fetchWeather({ lat: pos.coords.latitude, lon: pos.coords.longitude }),
    () => { /* silently ignore; user can search manually */ }
  );
}

// ===================== Crop calendar grid =====================
async function loadCalendar() {
  try {
    const res = await fetch(window.CROP_CALENDAR_URL);
    const data = await res.json();
    const grid = document.getElementById('calendarGrid');
    grid.innerHTML = Object.entries(data).map(([crop, info]) => `
      <div class="calendar-card">
        <div class="cc-head">${info.emoji} ${crop}</div>
        <dl>
          <dt>Season</dt><dd>${info.season}</dd>
          <dt>Sow</dt><dd>${info.sow}</dd>
          <dt>Harvest</dt><dd>${info.harvest}</dd>
        </dl>
      </div>`).join('');
  } catch (err) {
    console.warn('Calendar load failed', err);
  }
}
loadCalendar();

// ===================== Chatbot =====================
const chatFab = document.getElementById('chatFab');
const chatPanel = document.getElementById('chatPanel');
const chatBody = document.getElementById('chatBody');
const chatInput = document.getElementById('chatInput');

chatFab.addEventListener('click', () => {
  chatPanel.classList.toggle('show');
  if (chatPanel.classList.contains('show') && !chatBody.dataset.greeted) {
    addChatMsg('bot', 'Namaste! 🌱 Main aapka Farming Assistant hoon. Kuch bhi poochiye — N, P, K, soil pH, fertilizer, season...');
    chatBody.dataset.greeted = '1';
  }
});

function addChatMsg(who, text) {
  const div = document.createElement('div');
  div.className = `chat-msg ${who}`;
  div.textContent = text;
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
}

async function sendChat() {
  const msg = chatInput.value.trim();
  if (!msg) return;
  addChatMsg('user', msg);
  chatInput.value = '';
  try {
    const res = await fetch('/api/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg }),
    });
    const data = await res.json();
    addChatMsg('bot', data.reply);
  } catch {
    addChatMsg('bot', 'Connection issue — try again.');
  }
}
document.getElementById('chatSend').addEventListener('click', sendChat);
chatInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendChat(); });

// ===================== PWA install =====================
let deferredPrompt;
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  document.getElementById('installBtn').style.display = 'flex';
});
document.getElementById('installBtn').addEventListener('click', async () => {
  if (!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  document.getElementById('installBtn').style.display = 'none';
});

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/static/service-worker.js').catch(() => {});
  });
}
