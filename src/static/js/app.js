/* Bridge Inspector frontend JS (no Azure / no maps)
 *
 * This file intentionally avoids any map SDK usage and only uses local Flask endpoints.
 */

// ========== UTIL ==========

function $(id) {
  return document.getElementById(id);
}

function safeText(v) {
  if (v === null || v === undefined) return 'N/A';
  return String(v);
}

function showError(targetId, msg) {
  const el = $(targetId);
  if (!el) return;
  el.innerHTML = `<div class="error">${safeText(msg)}</div>`;
}

function showInfo(targetId, msg) {
  const el = $(targetId);
  if (!el) return;
  el.innerHTML = `<div class="info">${safeText(msg)}</div>`;
}

function showSuccess(targetId, msg) {
  const el = $(targetId);
  if (!el) return;
  el.innerHTML = `<div class="success">${safeText(msg)}</div>`;
}

function renderMarkdown(md) {
  if (typeof marked !== 'undefined') {
    return marked.parse(md || '');
  }
  return (md || '').replaceAll('\n', '<br/>');
}

// ========== NAV ==========

function showPage(pageName) {
  const pages = document.querySelectorAll('.page');
  pages.forEach((page) => page.classList.remove('active'));

  const buttons = document.querySelectorAll('.menu-btn');
  buttons.forEach((btn) => btn.classList.remove('active'));

  const page = $(pageName);
  if (page) page.classList.add('active');

  const evt = window.event;
  if (evt && evt.target) evt.target.classList.add('active');

  if (pageName === 'ai-assistant') {
    checkAgentStatus();
    const resultDiv = $('agentResult');
    if (resultDiv) resultDiv.innerHTML = '';
  }
}

// ========== AGENT ==========

async function checkAgentStatus() {
  const statusText = $('statusText');
  if (!statusText) return;
  statusText.textContent = 'Checking agent status...';

  try {
    const resp = await fetch('/api/agent/status');
    const data = await resp.json();
    statusText.textContent = data.status === 'ready' ? 'Agent ready' : 'Agent not ready';
  } catch {
    statusText.textContent = 'Agent status unavailable';
  }
}

function setQuery(q) {
  const input = $('agentQuery');
  if (input) input.value = q;
}

async function queryAgent() {
  const input = $('agentQuery');
  const resultDiv = $('agentResult');
  const btn = $('queryButton');
  if (!input || !resultDiv || !btn) return;

  const query = (input.value || '').trim();
  if (!query) {
    resultDiv.innerHTML = `<div class="error">Please enter a query.</div>`;
    resultDiv.classList.add('visible');
    return;
  }

  // Check if this is a navigation query for statistics or overview
  const lowerQuery = query.toLowerCase();
  if (lowerQuery.includes('statistic')) {
    // Navigate to statistics page
    showPage('statistics');
    
    // Auto-click the appropriate statistics button
    if (lowerQuery.includes('county')) {
      setTimeout(() => getCountyStats(), 500);
    } else if (lowerQuery.includes('span')) {
      setTimeout(() => getSpanStats(), 500);
    }
    
    // Clear the query and show message
    input.value = '';
    resultDiv.innerHTML = '<div class="success">✓ Navigated to Statistics page.</div>';
    resultDiv.classList.add('visible');
    return;
  }
  
  // Check if this is a total count query
  if ((lowerQuery.includes('how many') || lowerQuery.includes('total') || lowerQuery.includes('count')) && 
      lowerQuery.includes('bridge') && 
      !lowerQuery.includes('county') && 
      !lowerQuery.includes('span') && 
      !lowerQuery.includes('carries') && 
      !lowerQuery.includes('carrying') && 
      !lowerQuery.includes('cross') &&
      !lowerQuery.includes('in ')) {
    // Navigate to overview page and show count
    showPage('overview');
    
    // Auto-click the count button
    setTimeout(() => getBridgeCount(), 500);
    
    // Clear the query and show message
    input.value = '';
    resultDiv.innerHTML = '<div class="success">✓ Showing total bridge count on Overview page.</div>';
    resultDiv.classList.add('visible');
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Working...';
  resultDiv.innerHTML = `<div class="info">Processing...</div>`;
  resultDiv.classList.add('visible');

  try {
    const resp = await fetch('/api/agent/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    const data = await resp.json();

    // Debug logging
    console.log('Agent response:', data);

    if (!resp.ok || !data.success) {
      resultDiv.innerHTML = `<div class="error">${safeText(data.error || 'Agent error')}</div>`;
      resultDiv.classList.add('visible');
      return;
    }

    const formatted = formatAgentResults(data);
    console.log('Formatted HTML length:', formatted.length);
    resultDiv.innerHTML = formatted;
    resultDiv.classList.add('visible');
  } catch (e) {
    console.error('Error in queryAgent:', e);
    resultDiv.innerHTML = `<div class="error">${safeText(e.message || e)}</div>`;
    resultDiv.classList.add('visible');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Ask';
  }
}

function formatAgentResults(data) {
  const { result_type, data: resultData, total_results, endpoint } = data;

  // Debug logging
  console.log('formatAgentResults called with:', {
    result_type,
    resultData,
    total_results,
    endpoint,
    isArray: Array.isArray(resultData),
    length: Array.isArray(resultData) ? resultData.length : 'not array'
  });

  if (!resultData || (Array.isArray(resultData) && resultData.length === 0)) {
    return '<div class="info">No results found.</div>';
  }

  let html = '<div class="agent-response">';
  html += `<div class="agent-response-header">`;
  html += `<span class="agent-response-title">Results</span>`;
  html += `<span class="agent-response-endpoint">API: ${safeText(endpoint)}</span>`;
  html += '</div>';
  html += '<div class="copyable-table-container">';
  html += '<button class="copy-table-btn" onclick="copyTableToClipboard(this)">📋 Copy as CSV</button>';
  html += '<div class="table-wrapper"><table class="copyable-table">';

  switch (result_type) {
    case "count":
      html += "<thead><tr><th>Metric</th><th>Value</th></tr></thead>";
      html += "<tbody>";
      html += `<tr><td>Total Bridges</td><td>${resultData.total_bridges}</td></tr>`;
      html += "</tbody>";
      break;

    case "group_by_county":
      html +=
        "<thead><tr><th>County</th><th>Count</th><th>Avg Spans</th></tr></thead>";
      html += "<tbody>";
      resultData.forEach((row) => {
        html += `<tr><td>${safeText(row.county)}</td><td>${
          row.count
        }</td><td>${parseFloat(row.avg_spans).toFixed(2)}</td></tr>`;
      });
      html += "</tbody>";
      break;

    case "group_by_spans":
      html += "<thead><tr><th>Spans</th><th>Count</th></tr></thead>";
      html += "<tbody>";
      resultData.forEach((row) => {
        html += `<tr><td>${row.spans}</td><td>${row.count}</td></tr>`;
      });
      html += "</tbody>";
      break;

    case "bridges":
      console.log("Rendering bridges table with", resultData.length, "rows");
      html +=
        "<thead><tr><th>#</th><th>BIN</th><th>County</th><th>Carries</th><th>Crosses</th><th>Spans</th><th>Location</th><th>Maps</th></tr></thead>";
      html += "<tbody>";
      resultData.forEach((bridge, idx) => {
        console.log("Bridge row", idx, ":", bridge);
        const lat = bridge.latitude || "";
        const lng = bridge.longitude || "";
        const mapsLink =
          lat && lng
            ? `<a href="https://www.google.com/maps?q=${lat},${lng}" target="_blank" class="maps-link">📍 View</a>`
            : "N/A";

        html += `<tr>
            <td>${idx + 1}</td>
            <td>${bridge.bin || "N/A"}</td>
            <td>${bridge.county || "N/A"}</td>
            <td>${bridge.carried || "N/A"}</td>
            <td>${bridge.crossed || "N/A"}</td>
            <td>${bridge.spans || "N/A"}</td>
            <td>${lat && lng ? `${lat}, ${lng}` : "N/A"}</td>
            <td>${mapsLink}</td>
          </tr>`;
      });
      html += "</tbody>";
      if (total_results > resultData.length) {
        html += `<tfoot><tr><td colspan="8" style="text-align: center; font-style: italic;">Showing first ${resultData.length} of ${total_results} results</td></tr></tfoot>`;
      }
      break;

    case "inspections":
      html +=
        "<thead><tr><th>Bridge BIN</th><th>Date</th><th>Time</th><th>Weather</th><th>Notes</th></tr></thead>";
      html += "<tbody>";
      resultData.forEach((insp) => {
        const notes = (insp.notes || "").substring(0, 100);
        html += `<tr>
          <td>${safeText(insp.bin) || "N/A"}</td>
          <td>${safeText(insp.date) || "N/A"}</td>
          <td>${safeText(insp.time) || "N/A"}</td>
          <td>${safeText(insp.weather) || "N/A"}</td>
          <td>${safeText(notes)}${notes.length >= 100 ? "..." : ""}</td>
        </tr>`;
      });
      html += "</tbody>";
      break;

    default:
      html += "<thead><tr><th>Data</th></tr></thead>";
      html += "<tbody>";
      html += `<tr><td><pre>${safeText(
        JSON.stringify(resultData, null, 2)
      )}</pre></td></tr>`;
      html += "</tbody>";
  }

  html += '</table></div></div>';
  if (total_results) {
    html += `<div class="agent-response-footer">Total: ${total_results} result(s)</div>`;
  }
  html += '</div>';
  return html;
}

function copyTableToClipboard(button) {
  try {
    console.log('Copy button clicked, button:', button);
    
    // Find the table - it's in the next sibling's child
    const tableWrapper = button.nextElementSibling;
    console.log('Table wrapper:', tableWrapper);
    
    if (!tableWrapper) {
      console.error('Table wrapper not found');
      alert('Error: Could not find table wrapper');
      return;
    }
    
    const table = tableWrapper.querySelector('table');
    console.log('Table found:', table);
    
    if (!table) {
      console.error('Table not found inside wrapper');
      alert('Error: Could not find table element');
      return;
    }
    
    let csv = '';

    // Get headers
    const headers = table.querySelectorAll('thead th');
    console.log('Headers found:', headers.length);
    
    if (headers.length === 0) {
      alert('Error: No table headers found');
      return;
    }
    
    const headerRow = Array.from(headers).map(th => {
      const text = th.textContent.trim();
      // Escape quotes and wrap in quotes if contains comma
      return text.includes(',') ? `"${text.replace(/"/g, '""')}"` : text;
    }).join(',');
    csv += headerRow + '\n';

    // Get data rows
    const rows = table.querySelectorAll('tbody tr');
    console.log('Data rows found:', rows.length);
    
    rows.forEach(row => {
      const cells = row.querySelectorAll('td');
      const rowData = Array.from(cells).map(td => {
        let text = td.textContent.trim();
        // Remove "View" text from maps links, keep only coordinate or N/A
        if (text.includes('📍')) {
          text = text.replace('📍 View', '').trim();
        }
        // Escape quotes and wrap in quotes if contains comma or quote
        if (text.includes(',') || text.includes('"') || text.includes('\n')) {
          return `"${text.replace(/"/g, '""')}"`;
        }
        return text;
      }).join(',');
      csv += rowData + '\n';
    });

    console.log('CSV generated, length:', csv.length);
    console.log('First 200 chars:', csv.substring(0, 200));

    // Copy to clipboard - try modern API first, fallback to legacy method
    const copySuccess = () => {
      console.log('Successfully copied to clipboard');
      const originalText = button.textContent;
      button.textContent = '✓ Copied!';
      button.style.background = '#003A70';
      setTimeout(() => {
        button.textContent = originalText;
        button.style.background = '';
      }, 2000);
    };

    if (navigator.clipboard && window.isSecureContext) {
      // Modern async clipboard API (requires HTTPS or localhost)
      navigator.clipboard.writeText(csv).then(copySuccess).catch(err => {
        console.error('Clipboard API failed, trying fallback:', err);
        copyWithFallback(csv, copySuccess);
      });
    } else {
      // Fallback for HTTP or older browsers
      console.log('Using fallback copy method');
      copyWithFallback(csv, copySuccess);
    }
  } catch (err) {
    console.error('Error in copyTableToClipboard:', err);
    alert('An error occurred while copying: ' + err.message);
  }
}

function copyWithFallback(text, successCallback) {
  try {
    // Create a temporary textarea
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    textarea.style.left = '-999999px';
    document.body.appendChild(textarea);
    
    // Select and copy
    textarea.select();
    textarea.setSelectionRange(0, text.length);
    
    const successful = document.execCommand('copy');
    document.body.removeChild(textarea);
    
    if (successful) {
      successCallback();
    } else {
      alert('Failed to copy. Please manually select and copy the table.');
    }
  } catch (err) {
    console.error('Fallback copy failed:', err);
    alert('Copy failed: ' + err.message);
  }
}

document.addEventListener('DOMContentLoaded', function () {
  const agentInput = $('agentQuery');
  if (agentInput) {
    agentInput.addEventListener('keypress', function (e) {
      if (e.key === 'Enter') queryAgent();
    });
  }
});

// ========== OVERVIEW ==========

async function getBridgeCount() {
  try {
    const resp = await fetch('/api/bridges/count');
    const data = await resp.json();
    const result = $('countResult');
    if (!result) return;
    result.innerHTML = `
      <div class="stat-box">
        <div class="stat-number">${safeText(data.total_bridges)}</div>
        <div class="stat-label">Total Bridges</div>
      </div>
    `;
  } catch (e) {
    showError('countResult', e.message || e);
  }
}

// ========== SEARCH ==========

let lastSearchResults = [];

async function searchBridges() {
  const params = new URLSearchParams();
  const bin = $('searchBin')?.value?.trim();
  const county = $('searchCounty')?.value?.trim();
  const carried = $('searchCarried')?.value?.trim();
  const crossed = $('searchCrossed')?.value?.trim();
  const minSpans = $('searchMinSpans')?.value?.trim();
  const maxSpans = $('searchMaxSpans')?.value?.trim();

  if (bin) params.append('bin', bin);
  if (county) params.append('county', county);
  if (carried) params.append('carried', carried);
  if (crossed) params.append('crossed', crossed);
  if (minSpans) params.append('min_spans', minSpans);
  if (maxSpans) params.append('max_spans', maxSpans);

  showInfo('searchResult', 'Searching...');
  const resultsContainer = $('bridgeListOnMap');
  if (resultsContainer) resultsContainer.innerHTML = '';

  try {
    const resp = await fetch(`/api/bridges/search?${params.toString()}`);
    const data = await resp.json();

    if (!resp.ok) {
      showError('searchResult', data.error || `Search failed (${resp.status})`);
      return;
    }

    lastSearchResults = data.results || [];
    if (lastSearchResults.length === 0) {
      showInfo('searchResult', 'No bridges found matching your criteria.');
      return;
    }

    showSuccess('searchResult', `Found ${safeText(data.count)} bridge(s).`);
    renderBridgeTable(lastSearchResults);
  } catch (e) {
    showError('searchResult', e.message || e);
  }
}

function clearSearch() {
  if ($('searchBin')) $('searchBin').value = '';
  if ($('searchCounty')) $('searchCounty').value = '';
  if ($('searchCarried')) $('searchCarried').value = '';
  if ($('searchCrossed')) $('searchCrossed').value = '';
  if ($('searchMinSpans')) $('searchMinSpans').value = '';
  if ($('searchMaxSpans')) $('searchMaxSpans').value = '';
  if ($('searchResult')) $('searchResult').innerHTML = '';
  if ($('bridgeListOnMap')) $('bridgeListOnMap').innerHTML = '';
  lastSearchResults = [];
}

function renderBridgeTable(bridges) {
  const container = $('bridgeListOnMap');
  if (!container) return;
  if (!bridges || bridges.length === 0) {
    container.innerHTML = '';
    return;
  }

  const maxRows = 200;
  const rows = bridges.slice(0, maxRows);

  let html = `<div class="results">
    <div style="margin-bottom: 10px;"><strong>Showing ${rows.length}</strong> of ${bridges.length}</div>
    <table>
      <tr>
        <th>BIN</th><th>County</th><th>Carries</th><th>Crosses</th><th>Spans</th><th>Location</th><th>Map</th>
      </tr>`;

  for (const b of rows) {
    const lat = b.latitude || '';
    const lng = b.longitude || '';
    const mapsLink = lat && lng 
      ? `<a href="https://www.google.com/maps?q=${lat},${lng}" target="_blank" class="maps-link">📍 View</a>`
      : 'N/A';
    
    html += `<tr>
      <td>${safeText(b.bin)}</td>
      <td>${safeText(b.county)}</td>
      <td>${safeText(b.carried)}</td>
      <td>${safeText(b.crossed)}</td>
      <td>${safeText(b.spans)}</td>
      <td>${lat && lng ? `${lat}, ${lng}` : 'N/A'}</td>
      <td>${mapsLink}</td>
    </tr>`;
  }

  html += `</table></div>`;
  container.innerHTML = html;
}

// ========== STATS ==========

async function getCountyStats() {
  try {
    const resp = await fetch('/api/bridges/group-by-county');
    const data = await resp.json();
    if (!resp.ok) {
      showError('countyStatsResult', data.error || `Error (${resp.status})`);
      return;
    }
    let html = '<table><tr><th>County</th><th>Count</th><th>Avg Spans</th></tr>';
    for (const row of data) {
      const avg = row.avg_spans ? Number(row.avg_spans) : 0;
      html += `<tr><td>${safeText(row.county)}</td><td>${safeText(row.count)}</td><td>${avg.toFixed(2)}</td></tr>`;
    }
    html += '</table>';
    $('countyStatsResult').innerHTML = html;
  } catch (e) {
    showError('countyStatsResult', e.message || e);
  }
}

async function getSpanStats() {
  try {
    const resp = await fetch('/api/bridges/group-by-spans');
    const data = await resp.json();
    if (!resp.ok) {
      showError('spanStatsResult', data.error || `Error (${resp.status})`);
      return;
    }
    let html = '<table><tr><th>Spans</th><th>Count</th></tr>';
    for (const row of data) {
      html += `<tr><td>${safeText(row.spans)}</td><td>${safeText(row.count)}</td></tr>`;
    }
    html += '</table>';
    $('spanStatsResult').innerHTML = html;
  } catch (e) {
    showError('spanStatsResult', e.message || e);
  }
}

// ========== INSPECTIONS ==========

let uploadedPhotos = []; // { url, blobName }

function previewMarkdown() {
  const raw = $('inspectionNotes')?.value || '';
  const preview = $('markdownPreview');
  if (!preview) return;
  preview.innerHTML = renderMarkdown(raw);
}

function handleMarkdownDragOver(e) {
  e.preventDefault();
}

function handleMarkdownDrop(e) {
  e.preventDefault();
  const text = e.dataTransfer?.getData('text/plain');
  if (!text) return;
  const textarea = $('inspectionNotes');
  if (!textarea) return;
  textarea.value = (textarea.value || '') + text;
  previewMarkdown();
}

function handlePhotoDragOver(e) {
  e.preventDefault();
  $('photoDropZone')?.classList?.add('drag-over');
}

function handlePhotoDragLeave(e) {
  e.preventDefault();
  $('photoDropZone')?.classList?.remove('drag-over');
}

function handlePhotoDrop(e) {
  e.preventDefault();
  $('photoDropZone')?.classList?.remove('drag-over');
  const files = Array.from(e.dataTransfer?.files || []);
  uploadPhotos(files);
}

function handlePhotoFileSelect(e) {
  const files = Array.from(e.target.files || []);
  uploadPhotos(files);
  e.target.value = '';
}

async function uploadPhotos(files) {
  if (!files || files.length === 0) return;
  showInfo('inspectionResult', `Uploading ${files.length} photo(s)...`);

  for (const file of files) {
    const fd = new FormData();
    fd.append('photo', file);
    try {
      const resp = await fetch('/api/inspections/upload-photo', { method: 'POST', body: fd });
      const data = await resp.json();
      if (!resp.ok || !data.success) {
        showError('inspectionResult', data.error || `Upload failed (${resp.status})`);
        continue;
      }
      uploadedPhotos.push({ url: data.url, blobName: data.blobName });
      renderPhotoGallery();
    } catch (e) {
      showError('inspectionResult', e.message || e);
    }
  }

  showSuccess('inspectionResult', 'Photo upload complete.');
}

function renderPhotoGallery() {
  const gallery = $('photoGallery');
  if (!gallery) return;
  if (uploadedPhotos.length === 0) {
    gallery.innerHTML = '';
    return;
  }
  gallery.innerHTML = uploadedPhotos
    .map(
      (p, idx) => `
        <div class="photo-item">
          <img src="${p.url}" alt="photo-${idx}" />
        </div>
      `
    )
    .join('');
}

async function logInspection() {
  const bin = $('inspectionBin')?.value?.trim();
  const inspection_date = $('inspectionDate')?.value;
  const inspection_time = $('inspectionTime')?.value;
  const weather = $('inspectionWeather')?.value || '';
  const notes = $('inspectionNotes')?.value || '';
  const photos = JSON.stringify(uploadedPhotos.map((p) => p.url));

  if (!bin || !inspection_date || !inspection_time) {
    showError('inspectionResult', 'BIN, date, and time are required.');
    return;
  }

  try {
    const resp = await fetch('/api/inspections/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bin, inspection_date, inspection_time, weather, notes, photos }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      showError('inspectionResult', data.error || `Error (${resp.status})`);
      return;
    }
    showSuccess('inspectionResult', data.message || 'Inspection logged.');
    uploadedPhotos = [];
    renderPhotoGallery();
  } catch (e) {
    showError('inspectionResult', e.message || e);
  }
}

async function getInspectionHistory() {
  const bin = $('viewHistoryBin')?.value?.trim();
  if (!bin) {
    showError('inspectionHistory', 'Please enter a BIN.');
    return;
  }
  showInfo('inspectionHistory', 'Loading...');
  try {
    const resp = await fetch(`/api/inspections/${encodeURIComponent(bin)}`);
    const data = await resp.json();
    if (!resp.ok) {
      showError('inspectionHistory', data.error || `Error (${resp.status})`);
      return;
    }
    const inspections = data.inspections || [];
    if (inspections.length === 0) {
      showInfo('inspectionHistory', 'No inspections found.');
      return;
    }
    let html = '<table><tr><th>Date</th><th>Time</th><th>Weather</th><th>Notes</th><th>Photos</th></tr>';
    for (const ins of inspections) {
      let photosHtml = '';
      try {
        const urls = JSON.parse(ins.photos || '[]');
        photosHtml = (urls || []).map((u) => `<a href="${u}" target="_blank">view</a>`).join(' | ');
      } catch {
        photosHtml = '';
      }
      html += `<tr>
        <td>${safeText(ins.inspection_date)}</td>
        <td>${safeText(ins.inspection_time)}</td>
        <td>${safeText(ins.weather)}</td>
        <td>${renderMarkdown(ins.notes || '')}</td>
        <td>${photosHtml}</td>
      </tr>`;
    }
    html += '</table>';
    $('inspectionHistory').innerHTML = html;
  } catch (e) {
    showError('inspectionHistory', e.message || e);
  }
}

async function loadBridgesList(page = 1) {
  const from_date = $('filterFromDate')?.value;
  const to_date = $('filterToDate')?.value;
  const limit = $('filterLimit')?.value;

  let url = `/api/inspections/bridges?page=${page}&per_page=20`;
  if (from_date) url += `&from_date=${encodeURIComponent(from_date)}`;
  if (to_date) url += `&to_date=${encodeURIComponent(to_date)}`;
  if (limit) url += `&limit=${encodeURIComponent(limit)}`;

  showInfo('bridgesListResult', 'Loading...');

  try {
    const resp = await fetch(url);
    const data = await resp.json();
    if (!resp.ok) {
      showError('bridgesListResult', data.error || `Error (${resp.status})`);
      return;
    }

    const bridges = data.bridges || [];
    if (bridges.length === 0) {
      showInfo('bridgesListResult', 'No inspection logs found.');
      return;
    }

    let html = '<table><tr><th>BIN</th><th>Inspection Count</th><th>Latest Inspection</th></tr>';
    for (const b of bridges) {
      html += `<tr>
        <td>${safeText(b.bin)}</td>
        <td>${safeText(b.inspection_count)}</td>
        <td>${safeText(b.latest_inspection_date)}</td>
      </tr>`;
    }
    html += '</table>';
    $('bridgesListResult').innerHTML = html;
  } catch (e) {
    showError('bridgesListResult', e.message || e);
  }
}

function clearInspectionFilters() {
  if ($('filterFromDate')) $('filterFromDate').value = '';
  if ($('filterToDate')) $('filterToDate').value = '';
  if ($('filterLimit')) $('filterLimit').value = '';
  $('bridgesListResult').innerHTML = '';
}

async function deleteByBridge() {
  const bin = $('deleteByBridgeBin')?.value?.trim();
  if (!bin) {
    showError('deleteByBridgeResult', 'Please enter a BIN.');
    return;
  }
  try {
    const resp = await fetch(`/api/inspections/delete-by-bridge/${encodeURIComponent(bin)}`, { method: 'DELETE' });
    const data = await resp.json();
    if (!resp.ok) {
      showError('deleteByBridgeResult', data.error || `Error (${resp.status})`);
      return;
    }
    showSuccess('deleteByBridgeResult', data.message || 'Deleted.');
  } catch (e) {
    showError('deleteByBridgeResult', e.message || e);
  }
}

async function deleteAllInspections() {
  if (!confirm('Delete ALL inspections? This cannot be undone.')) return;
  try {
    const resp = await fetch('/api/inspections/delete-all', { method: 'DELETE' });
    const data = await resp.json();
    if (!resp.ok) {
      showError('deleteAllResult', data.error || `Error (${resp.status})`);
      return;
    }
    showSuccess('deleteAllResult', data.message || 'Deleted.');
  } catch (e) {
    showError('deleteAllResult', e.message || e);
  }
}
