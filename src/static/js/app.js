/* Bridge app frontend JS
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

  const page = document.getElementById(pageName);
  if (page) page.classList.add('active');

  const evt = window.event;
  if (evt && evt.target && evt.target.classList.contains('menu-btn')) {
      evt.target.classList.add('active');
  }

  // === MAP MOVING LOGIC ===
  const mapElement = document.getElementById('bridgeMap');

  if (pageName === 'ai-assistant') {
      const aiSlot = document.getElementById('ai-map-slot');
      if (mapElement && aiSlot) {
          aiSlot.appendChild(mapElement);
      }

      checkAgentStatus();
      const resultDiv = document.getElementById('agentResult');
      if (resultDiv) resultDiv.innerHTML = '';
  }

  else if (pageName === 'search') {
      const searchSlot = document.getElementById('search-map-slot');
      if (mapElement && searchSlot) {
          searchSlot.appendChild(mapElement);
      }
  }

  if ((pageName === 'ai-assistant' || pageName === 'search')) {
      setTimeout(() => {
          if (!mapInstance) {
              if (typeof initMap === 'function') initMap();
          } else {
              mapInstance.invalidateSize();
          }
      }, 100);
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

async function queryAgent(page = 1, savedQuery = null) {
  const input = $("agentQuery");
  const resultDiv = $("agentResult");
  const btn = $("queryButton");
  if (!input || !resultDiv || !btn) return;

  // Use savedQuery if provided (from pagination click), otherwise read input
  const query = savedQuery || (input.value || "").trim();

  if (!query) {
    resultDiv.innerHTML = `<div class="error">Please enter a question!</div>`;
    resultDiv.classList.add("visible");
    return;
  }

  // Check shortcuts (statistics/count) only on the first page or fresh query
  if (!savedQuery) {
    const lowerQuery = query.toLowerCase();

    // 1. Navigation shortcuts
    if (lowerQuery.includes("statistic")) {
      showPage("statistics");
      if (lowerQuery.includes("county"))
        setTimeout(() => getCountyStats(), 500);
      else if (lowerQuery.includes("span"))
        setTimeout(() => getSpanStats(), 500);
      input.value = "";
      resultDiv.innerHTML =
        '<div class="success">✓ Navigated to Statistics page.</div>';
      resultDiv.classList.add("visible");
      return;
    }

    // 2. Simple Count shortcut (local fetch)
    const isGeneralCountQuery =
      (lowerQuery.includes("how many") ||
        lowerQuery.includes("total") ||
        lowerQuery.includes("count")) &&
      lowerQuery.includes("bridge") &&
      !lowerQuery.includes("county") &&
      !lowerQuery.includes("span") &&
      !lowerQuery.includes("carries") &&
      !lowerQuery.includes("carrying") &&
      !lowerQuery.includes("cross") &&
      !(/\bin\s+\w+/i.test(lowerQuery) && !lowerQuery.includes("in the"));

    if (isGeneralCountQuery) {
      try {
        const resp = await fetch("/api/bridges/count");
        const data = await resp.json();
        resultDiv.innerHTML = `<div class="success"><strong>${data.total_bridges.toLocaleString()}</strong> bridges</div>`;
        resultDiv.classList.add("visible");
        input.value = "";
      } catch (e) {
        resultDiv.innerHTML = `<div class="error">Error: ${e.message}</div>`;
        resultDiv.classList.add("visible");
      }
      return;
    }
  }

  // Standard AI Query
  btn.disabled = true;
  btn.textContent = "Working...";

  // Only show "Processing" on first load, not pagination (optional, but keeps UI cleaner)
  if (page === 1) {
    resultDiv.innerHTML = `<div class="info">Processing...</div>`;
    resultDiv.classList.add("visible");
  }

  try {
    // Pass 'page' in the request body
    const resp = await fetch("/api/agent/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, page }),
    });
    const data = await resp.json();

    console.log("Agent response:", data);

    if (!resp.ok || !data.success) {
      resultDiv.innerHTML = `<div class="error">${safeText(
        data.error || "Agent error"
      )}</div>`;
      resultDiv.classList.add("visible");
      return;
    }

    // Pass the 'query' to formatAgentResults so it can generate pagination links
    const formatted = formatAgentResults(data, query);
    resultDiv.innerHTML = formatted;
    resultDiv.classList.add("visible");

    // If this was a manual search (not pagination), clear input
    if (!savedQuery) {
      // input.value = ''; // Optional: uncomment if you want to clear input after search
    }
  } catch (e) {
    console.error("Error in queryAgent:", e);
    resultDiv.innerHTML = `<div class="error">${safeText(
      e.message || e
    )}</div>`;
    resultDiv.classList.add("visible");
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask";
  }
}

function formatAgentResults(data, currentQuery) {
  // Destructure 'pagination' from the response
  const {
    result_type,
    data: resultData,
    total_results,
    endpoint,
    pagination,
  } = data;

  if (!resultData || (Array.isArray(resultData) && resultData.length === 0)) {
    return '<div class="info">No results found.</div>';
  }

  let html = '<div class="agent-response">';
  html += `<div class="agent-response-header">`;
  html += `<span class="agent-response-title">Results</span>`;
  html += `<span class="agent-response-endpoint">API: ${safeText(
    endpoint
  )}</span>`;
  html += "</div>";

  // --- PAGINATION LOGIC START ---
  if (pagination && pagination.total_pages > 1) {
    // Escape quotes in query for the onclick attribute
    const safeQuery = currentQuery.replace(/'/g, "\\'");

    html += `<div class="pagination-controls" style="margin-bottom: 15px;">
      <button onclick="queryAgent(${pagination.page - 1}, '${safeQuery}')"
              ${!pagination.has_prev ? "disabled" : ""}
              class="page-btn">← Previous</button>
      <span style="margin: 0 15px;">Page ${pagination.page} of ${
      pagination.total_pages
    }</span>
      <button onclick="queryAgent(${pagination.page + 1}, '${safeQuery}')"
              ${!pagination.has_next ? "disabled" : ""}
              class="page-btn">Next →</button>
    </div>`;
  }
  // --- PAGINATION LOGIC END ---

  html += '<div class="copyable-table-container">';
  html +=
    '<button class="copy-table-btn" onclick="copyTableToClipboard(this)">📋 Copy as CSV</button>';
  html += '<div class="table-wrapper"><table class="copyable-table">';

  switch (result_type) {
    case "count":
      html += "<thead><tr><th>Metric</th><th>Value</th></tr></thead>";
      html += `<tbody><tr><td>Total Bridges</td><td>${resultData.total_bridges}</td></tr></tbody>`;
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
      html +=
        "<thead><tr><th>#</th><th>BIN</th><th>County</th><th>Carries</th><th>Crosses</th><th>Spans</th><th>Location</th><th>Maps</th></tr></thead>";
      html += "<tbody>";
      resultData.forEach((bridge, idx) => {
        // Calculate correct row number based on page
        const baseIndex = pagination
          ? (pagination.page - 1) * pagination.per_page
          : 0;
        const rowNum = baseIndex + idx + 1;

        const lat = bridge.latitude || "";
        const lng = bridge.longitude || "";
        const mapsLink =
          lat && lng
            ? `<a href="https://www.google.com/maps?q=${lat},${lng}" target="_blank" class="maps-link">📍 View</a>`
            : "N/A";

        html += `<tr>
            <td>${rowNum}</td>
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
      html += `<tbody><tr><td><pre>${safeText(
        JSON.stringify(resultData, null, 2)
      )}</pre></td></tr></tbody>`;
  }

  html += "</table></div></div>";

  if (total_results) {
    html += `<div class="agent-response-footer">Total: ${total_results} result(s)</div>`;
  }

  html += "</div>";
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
  const result = $('countResult');
  if (!result) return;

  if (result.innerHTML.trim() !== '') {
    result.innerHTML = '';
    return;
  }

  try {
    const resp = await fetch('/api/bridges/count');
    const data = await resp.json();
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

async function loadExamples() {
  try {
    const response = await fetch("/api/agent/examples");
    const data = await response.json();

    const container = document.getElementById("exampleQueries");
    container.innerHTML = ""; // Clear loading message

    data.examples.forEach((example) => {
      const span = document.createElement("span");
      span.className = "example-query";
      span.textContent = example;
      span.onclick = () => setQuery(example);
      container.appendChild(span);
    });
  } catch (error) {
    console.error("Failed to load examples:", error);
    document.getElementById("exampleQueries").innerHTML =
      '<span class="error">Failed to load examples</span>';
  }
}

document.addEventListener("DOMContentLoaded", loadExamples);

// ========== SEARCH ==========

let lastSearchResults = [];

// Add at the top of the file with other global variables
let currentPage = 1;
let currentSearchParams = null;

async function searchBridges(page = 1) {
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

  params.append('page', page);
  params.append('page_size', 50);

  currentSearchParams = params.toString().replace(/&?page=\d+/, '');
  currentPage = page;

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

    const pageInfo = `Page ${data.page} of ${data.total_pages} (${data.total_count} total bridges)`;
    showSuccess('searchResult', pageInfo);
    renderBridgeTable(lastSearchResults, data);
  } catch (e) {
    showError('searchResult', e.message || e);
  }
}

function renderBridgeTable(bridges, paginationData) {
  const container = $('bridgeListOnMap');
  if (!container) return;
  if (!bridges || bridges.length === 0) {
    container.innerHTML = '';
    return;
  }

  let html = `<div class="results">
    <div style="margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
      <strong>Showing ${bridges.length} bridges</strong>`;

  if (paginationData && paginationData.total_pages > 1) {
    html += `<div class="pagination-controls">
      <button onclick="searchBridges(${paginationData.page - 1})"
              ${paginationData.page <= 1 ? 'disabled' : ''}
              class="page-btn">← Previous</button>
      <span style="margin: 0 15px;">Page ${paginationData.page} of ${paginationData.total_pages}</span>
      <button onclick="searchBridges(${paginationData.page + 1})"
              ${paginationData.page >= paginationData.total_pages ? 'disabled' : ''}
              class="page-btn">Next →</button>
    </div>`;
  }

  html += `</div>
    <table>
      <tr>
        <th>BIN</th><th>County</th><th>Carries</th><th>Crosses</th><th>Spans</th><th>Location</th><th>Map</th>
      </tr>`;

  for (const b of bridges) {
    const lat = b.latitude;
    const lng = b.longitude;

    const safeBin = (b.bin || 'N/A').replace(/'/g, "\\'");
    const safeCarried = (b.carried || 'N/A').replace(/'/g, "\\'");
    const safeCrossed = (b.crossed || 'N/A').replace(/'/g, "\\'");

    let mapActions = '<span style="color: #999;">N/A</span>';

    if (lat && lng) {
      const localBtn = `<button 
           onclick="showOnMap(${lat}, ${lng}, '${safeBin}', '${safeCarried}', '${safeCrossed}')" 
           title="Show on map above"
           style="cursor: pointer; background: none; border: none; color: #1abc9c; font-weight: bold; margin-right: 8px; text-decoration: underline;">
           Local Map Display
         </button>`;

      const googleUrl = `https://www.google.com/maps?q=${lat},${lng}`;
      const googleLink = `<a href="${googleUrl}" target="_blank" title="Open in Google Maps" style="color: #3498db; text-decoration: none; font-size: 0.9em;">
           Google Maps Display
         </a>`;

      mapActions = `${localBtn} ${googleLink}`;
    }

    html += `<tr>
      <td>${safeText(b.bin)}</td>
      <td>${safeText(b.county)}</td>
      <td>${safeText(b.carried)}</td>
      <td>${safeText(b.crossed)}</td>
      <td>${safeText(b.spans)}</td>
      <td style="font-size: 0.85em; color: #666;">${lat && lng ? `${lat.toFixed(4)}, ${lng.toFixed(4)}` : 'N/A'}</td>
      <td>${mapActions}</td>
    </tr>`;
  }

  html += `</table></div>`;
  container.innerHTML = html;
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

// ========== Map ==================

let mapInstance = null;
let currentMarker = null;

function initMap() {
    const mapContainer = document.getElementById('bridgeMap');
    if (!mapContainer) {
        console.warn("initMap: 'bridgeMap' container not found on this page.");
        return;
    }

    if (mapInstance) {
        console.log("initMap: Map already initialized.");
        mapInstance.invalidateSize();
        return;
    }

    try {
        console.log("initMap: Initializing Leaflet map...");
        mapInstance = L.map('bridgeMap', {
            attributionControl: false
        }).setView([41.2, -74.0], 9);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '© OpenStreetMap'
        }).addTo(mapInstance);

        L.control.attribution({
            prefix: false
        }).addTo(mapInstance);

    } catch (e) {
        console.error("initMap Error:", e);
    }
}

window.showOnMap = function(lat, lng, bin, carried, crossed) {
    console.log(`showOnMap: BIN=${bin}`);

    if (!mapInstance) initMap();
    if (!mapInstance) return;
    mapInstance.invalidateSize();

    try {
        mapInstance.setView([lat, lng], 17);

        if (currentMarker) {
            mapInstance.removeLayer(currentMarker);
        }

        // Create nice HTML content for the popup
        const popupContent = `
            <div style="font-size: 14px; line-height: 1.5;">
                <strong>BIN:</strong> ${bin}<br>
                <strong>Carries:</strong> ${carried}<br>
                <strong style="color: #666;">Crosses:</strong> ${crossed}
            </div>
        `;

        currentMarker = L.marker([lat, lng]).addTo(mapInstance)
            .bindPopup(popupContent)
            .openPopup();

        // Scroll on mobile
        if (window.innerWidth < 900) {
            document.getElementById('bridgeMap').scrollIntoView({ behavior: 'smooth' });
        }
    } catch (e) {
        console.error("showOnMap Error:", e);
    }
};

// Initialize map immediately
document.addEventListener('DOMContentLoaded', initMap);

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
  if (!confirm('Delete ALL inspections?')) return;
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
