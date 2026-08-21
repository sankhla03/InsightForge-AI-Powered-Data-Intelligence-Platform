/**
 * insightforge.js
 * ===============
 * Shared JavaScript utilities for InsightForge.
 *
 * Provides:
 *   1. Dark mode toggle (localStorage)
 *   2. Toast notification system
 *   3. Skeleton loaders
 *   4. Table search, sort, and pagination
 *   5. AutoML progress bar polling
 *   6. Form validation helpers
 *   7. Django messages → toast conversion
 *   8. Mobile sidebar toggle
 *   9. File drop zone interactions
 */

/* ============================================================
   1. DARK MODE
   ============================================================ */
const InsightForge = (function () {

  // === Dark Mode ===
  function initDarkMode() {
    const stored = localStorage.getItem('if-theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = stored || (prefersDark ? 'dark' : 'light');
    document.documentElement.setAttribute('data-theme', theme);

    document.querySelectorAll('.if-dark-toggle').forEach(btn => {
      btn.textContent = theme === 'dark' ? '☀️' : '🌙';
      btn.setAttribute('title', theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode');
      btn.addEventListener('click', toggleDark);
    });
  }

  function toggleDark() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('if-theme', next);
    document.querySelectorAll('.if-dark-toggle').forEach(btn => {
      btn.textContent = next === 'dark' ? '☀️' : '🌙';
    });
    toast({ type: 'info', title: next === 'dark' ? '🌙 Dark Mode' : '☀️ Light Mode', body: 'Theme updated!' });
  }


  /* ============================================================
     2. TOAST NOTIFICATIONS
     ============================================================ */
  function ensureToastContainer() {
    let container = document.getElementById('if-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'if-toast-container';
      container.className = 'if-toast-container';
      document.body.appendChild(container);
    }
    return container;
  }

  const TOAST_ICONS = {
    success: '✅',
    error:   '❌',
    warning: '⚠️',
    info:    'ℹ️',
  };

  /**
   * Show a toast notification.
   * @param {Object} opts - { type, title, body, duration }
   */
  function toast({ type = 'info', title = '', body = '', duration = 4000 } = {}) {
    const container = ensureToastContainer();
    const el = document.createElement('div');
    el.className = `if-toast if-toast-${type}`;
    el.innerHTML = `
      <span class="if-toast-icon">${TOAST_ICONS[type] || 'ℹ️'}</span>
      <div class="if-toast-content">
        ${title ? `<div class="if-toast-title">${title}</div>` : ''}
        ${body  ? `<div class="if-toast-body">${body}</div>` : ''}
      </div>
      <button class="if-toast-close" aria-label="Close">✕</button>
    `;

    // Close button
    el.querySelector('.if-toast-close').addEventListener('click', () => dismissToast(el));

    container.appendChild(el);

    // Auto-dismiss
    if (duration > 0) {
      setTimeout(() => dismissToast(el), duration);
    }

    return el;
  }

  function dismissToast(el) {
    el.classList.add('exiting');
    el.addEventListener('animationend', () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 400);
  }

  /** Convert Django messages (already rendered as .if-alert elements) into toasts */
  function convertDjangoMessages() {
    document.querySelectorAll('.if-django-message').forEach(msg => {
      const type = msg.dataset.type || 'info';
      const text = msg.textContent.trim();
      if (text) {
        const mapped = { 'success': 'success', 'error': 'error', 'warning': 'warning', 'info': 'info', 'debug': 'info' };
        toast({ type: mapped[type] || 'info', title: text, duration: 5000 });
      }
      msg.remove();
    });
  }


  /* ============================================================
     3. TABLE ENHANCEMENTS (search, sort, global scroll)
     ============================================================ */

  /**
   * Enhance a table with search and sort without pagination (uses global scrolling container).
   * @param {string} tableId - ID of the <table> element
   * @param {Object} opts - { searchable: true, sortable: true }
   */
  function enhanceTable(tableId, opts = {}) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const { searchable = true, sortable = true } = opts;
    const tbody = table.querySelector('tbody');
    if (!tbody) return;
    const allRows = () => Array.from(tbody.querySelectorAll('tr'));

    let filteredRows = allRows();
    let sortCol = -1;
    let sortAsc = true;

    const paginationEl = document.getElementById(`${tableId}-pagination`);
    if (paginationEl) paginationEl.style.display = 'none';

    const pageInfoEl = document.getElementById(`${tableId}-pageinfo`);
    if (pageInfoEl) pageInfoEl.style.display = 'none';

    // ── Search ──────────────────────────────────────────────────────────────
    if (searchable) {
      const searchInput = document.getElementById(`${tableId}-search`);
      if (searchInput) {
        searchInput.addEventListener('input', () => {
          const q = searchInput.value.toLowerCase();
          allRows().forEach(row => {
            const matches = row.textContent.toLowerCase().includes(q);
            row.style.display = matches ? '' : 'none';
          });
        });
      }
    }

    // ── Sort ─────────────────────────────────────────────────────────────────
    if (sortable) {
      table.querySelectorAll('th[data-sortable]').forEach((th, idx) => {
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => {
          if (sortCol === idx) {
            sortAsc = !sortAsc;
          } else {
            sortCol = idx;
            sortAsc = true;
          }
          table.querySelectorAll('th').forEach(t => t.classList.remove('sorted-asc', 'sorted-desc'));
          th.classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');

          const rows = allRows();
          rows.sort((a, b) => {
            const aText = (a.cells[idx]?.textContent || '').trim();
            const bText = (b.cells[idx]?.textContent || '').trim();
            const aNum = parseFloat(aText);
            const bNum = parseFloat(bText);
            if (!isNaN(aNum) && !isNaN(bNum)) {
              return sortAsc ? aNum - bNum : bNum - aNum;
            }
            return sortAsc ? aText.localeCompare(bText) : bText.localeCompare(aText);
          });

          rows.forEach(r => tbody.appendChild(r));
        });
      });
    }

    // Ensure all rows are visible
    allRows().forEach(r => r.style.display = '');
  }


  /* ============================================================
     4. AUTOML PROGRESS POLLING
     ============================================================ */

  /**
   * Poll the AutoML status endpoint and update the UI.
   * @param {string} statusUrl - URL of the automl_status_view endpoint
   * @param {Function} onComplete - Callback when AutoML is done
   * @param {Function} onError - Callback when AutoML errors
   */
  function pollAutoMLProgress(statusUrl, onComplete, onError) {
    let interval;
    let pollCount = 0;
    const MAX_POLLS = 600;  // 10 minutes at 1s interval

    function updateUI(data) {
      // Update progress bar
      const fill = document.getElementById('automl-progress-fill');
      const label = document.getElementById('automl-progress-label');
      if (fill) fill.style.width = (data.progress || 0) + '%';
      if (label) label.textContent = (data.progress || 0) + '%';

      // Update stage list
      if (data.stages && data.stage) {
        data.stages.forEach(stage => {
          const el = document.getElementById(`stage-${stage.key}`);
          if (!el) return;
          el.className = 'if-automl-stage';
          const statusEl = el.querySelector('.if-automl-stage-status');

          if (stage.key === data.stage) {
            el.classList.add('active');
            if (statusEl) statusEl.textContent = 'Running...';
          } else if (isStageCompleted(stage.key, data.stage, data.stages)) {
            el.classList.add('done');
            if (statusEl) statusEl.textContent = '✓ Done';
          } else {
            el.classList.add('pending');
            if (statusEl) statusEl.textContent = 'Pending';
          }
        });
      }

      // Update message
      const msgEl = document.getElementById('automl-message');
      if (msgEl && data.message) msgEl.textContent = data.message;
    }

    function isStageCompleted(stageKey, currentStage, stages) {
      const keys = stages.map(s => s.key);
      const stageIdx = keys.indexOf(stageKey);
      const currentIdx = keys.indexOf(currentStage);
      return stageIdx < currentIdx || currentStage === 'complete';
    }

    interval = setInterval(async () => {
      pollCount++;
      if (pollCount > MAX_POLLS) {
        clearInterval(interval);
        if (onError) onError('AutoML timed out.');
        return;
      }

      try {
        const res = await fetch(statusUrl);
        const data = await res.json();
        updateUI(data);

        if (data.is_complete) {
          clearInterval(interval);
          const progressFill = document.getElementById('automl-progress-fill');
          if (progressFill) progressFill.style.width = '100%';
          if (onComplete) onComplete(data);
        } else if (data.has_error) {
          clearInterval(interval);
          if (onError) onError(data.error_message);
        }
      } catch (err) {
        console.warn('AutoML status poll failed:', err);
      }
    }, 1000);

    return { stop: () => clearInterval(interval) };
  }


  /* ============================================================
     5. FORM VALIDATION
     ============================================================ */

  function validateForm(formId, rules) {
    const form = document.getElementById(formId);
    if (!form) return false;

    let valid = true;

    Object.entries(rules).forEach(([fieldName, rule]) => {
      const field = form.querySelector(`[name="${fieldName}"]`);
      if (!field) return;

      const val = field.value.trim();
      const errEl = form.querySelector(`#error-${fieldName}`);

      let error = '';
      if (rule.required && !val) {
        error = rule.requiredMsg || 'This field is required.';
      } else if (rule.minLength && val.length < rule.minLength) {
        error = rule.minLengthMsg || `Minimum ${rule.minLength} characters required.`;
      } else if (rule.pattern && !rule.pattern.test(val)) {
        error = rule.patternMsg || 'Invalid format.';
      }

      if (error) {
        valid = false;
        field.classList.add('error');
        if (errEl) { errEl.textContent = error; errEl.style.display = 'flex'; }
      } else {
        field.classList.remove('error');
        field.classList.add('success');
        if (errEl) errEl.style.display = 'none';
      }
    });

    return valid;
  }


  /* ============================================================
     6. FILE DROP ZONE
     ============================================================ */

  function initFileDrop(dropId) {
    const zone = document.getElementById(dropId);
    if (!zone) return;

    zone.addEventListener('dragover', e => {
      e.preventDefault();
      zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', e => {
      e.preventDefault();
      zone.classList.remove('dragover');
      const fileInput = zone.querySelector('input[type="file"]');
      if (fileInput && e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        const nameEl = zone.querySelector('.if-file-drop-text');
        if (nameEl) nameEl.textContent = `📎 ${e.dataTransfer.files[0].name}`;
        zone.classList.add('has-file');
      }
    });

    // Update text on file input change
    const fileInput = zone.querySelector('input[type="file"]');
    if (fileInput) {
      fileInput.addEventListener('change', () => {
        const nameEl = zone.querySelector('.if-file-drop-text');
        if (nameEl && fileInput.files.length) {
          nameEl.textContent = `📎 ${fileInput.files[0].name}`;
          zone.classList.add('has-file');
        }
      });
    }
  }


  /* ============================================================
     7. MOBILE SIDEBAR
     ============================================================ */

  function initMobileSidebar() {
    const sidebar = document.querySelector('.if-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const toggleBtns = document.querySelectorAll('.sidebar-toggle');

    if (!sidebar) return;

    function openSidebar() {
      sidebar.classList.add('open');
      if (overlay) overlay.style.display = 'block';
    }
    function closeSidebar() {
      sidebar.classList.remove('open');
      if (overlay) overlay.style.display = 'none';
    }

    toggleBtns.forEach(btn => btn.addEventListener('click', () => {
      sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
    }));
    if (overlay) overlay.addEventListener('click', closeSidebar);
  }


  /* ============================================================
     8. MARK ACTIVE NAV ITEM
     ============================================================ */

  function markActiveNav() {
    const path = window.location.pathname;
    document.querySelectorAll('.if-nav-item').forEach(item => {
      const href = item.getAttribute('href') || '';
      if (href && path.startsWith(href) && href !== '/') {
        item.classList.add('active');
      }
    });
  }


  /* ============================================================
     9. PROGRESS BAR HELPER
     ============================================================ */

  function setProgress(fillId, pct, labelId) {
    const fill = document.getElementById(fillId);
    if (fill) fill.style.width = Math.max(0, Math.min(100, pct)) + '%';
    const label = document.getElementById(labelId);
    if (label) label.textContent = Math.round(pct) + '%';
  }


  function autoWrapTables() {
    const containerClasses = [
      'if-table-wrap',
      'insightforge-table-container',
      'table-responsive',
      'table-container',
      'table-wrap'
    ];

    document.querySelectorAll('table').forEach(table => {
      const parent = table.parentElement;
      if (!parent) return;
      const isAlreadyWrapped = containerClasses.some(cls => parent.classList.contains(cls));
      if (!isAlreadyWrapped) {
        const wrapper = document.createElement('div');
        wrapper.className = 'insightforge-table-container';
        parent.insertBefore(wrapper, table);
        wrapper.appendChild(table);
      }
    });
  }

  /* ============================================================
     INIT — Called automatically on DOMContentLoaded
     ============================================================ */

  function init() {
    initDarkMode();
    initMobileSidebar();
    markActiveNav();
    convertDjangoMessages();
    autoWrapTables();
    if (window.lucide) {
      window.lucide.createIcons();
    }
  }

  document.addEventListener('DOMContentLoaded', init);


  /* ============================================================
     PUBLIC API
     ============================================================ */

  return {
    toast,
    enhanceTable,
    pollAutoMLProgress,
    validateForm,
    initFileDrop,
    toggleDark,
    setProgress,
    markActiveNav,
  };

})();

// Convenience alias
window.IF = InsightForge;
