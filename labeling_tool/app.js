/* 🛣️ Road Health Detection System — Web Labeling Tool Logic */

document.addEventListener('DOMContentLoaded', () => {
  // --- DOM Elements ---
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const browseBtn = document.getElementById('browse-btn');
  const sampleDataBtn = document.getElementById('sample-data-btn');
  const exportCsvBtn = document.getElementById('export-csv-btn');
  const resetZoomBtn = document.getElementById('reset-zoom-btn');
  const btnZoomX = document.getElementById('btn-zoom-x');
  const btnZoomXY = document.getElementById('btn-zoom-xy');
  
  const inputStartTime = document.getElementById('input-start-time');
  const inputEndTime = document.getElementById('input-end-time');
  const inputSelectionWidth = document.getElementById('input-selection-width');
  const valSelectionWidth = document.getElementById('val-selection-width');
  
  const btnLabelPothole = document.getElementById('btn-label-pothole');
  const btnLabelSpeedbreaker = document.getElementById('btn-label-speedbreaker');
  const btnClearSelection = document.getElementById('btn-clear-selection');
  
  const statTotalRows = document.getElementById('stat-total-rows');
  const statPotholes = document.getElementById('stat-potholes');
  const statSpeedBreakers = document.getElementById('stat-speed-breakers');
  
  const labelsTableBody = document.getElementById('labels-list-body');
  const toastNotification = document.getElementById('toast-notification');
  const toastMessage = document.getElementById('toast-message');
  const loadingOverlay = document.getElementById('loading-overlay');
  const loadingText = document.getElementById('loading-text');
  
  const chartParent = document.getElementById('chart-parent');
  const chartPlaceholder = document.getElementById('chart-placeholder');
  const sensorChartCanvas = document.getElementById('sensor-chart');

  // --- App State ---
  let rawData = [];            // Original parsed CSV rows
  let chartData = null;        // Cleaned and normalized data for the chart
  let chartInstance = null;    // Chart.js object
  let labeledRegions = [];     // Array of { id, startTime, endTime, label }
  let selectionAnchor = null;  // The clicked timestamp (seconds)
  let activeSelection = null;  // Current selection range: { startTime, endTime }
  let zoomMode = 'x';          // 'x' or 'xy'
  let selectedLabelId = null;  // Selected label in the table list
  let currentFileName = '';    // Cached file name for localStorage keying and export name
  
  // Undo/Redo stacks
  let undoStack = [];
  let redoStack = [];

  // Register Chart.js plugins safely without double-registering
  try {
    if (typeof Chart !== 'undefined') {
      let isZoomRegistered = false;
      let isAnnotationRegistered = false;

      try {
        isZoomRegistered = !!Chart.registry.getPlugin('zoom');
      } catch (e) {
        isZoomRegistered = false;
      }

      try {
        isAnnotationRegistered = !!Chart.registry.getPlugin('annotation');
      } catch (e) {
        isAnnotationRegistered = false;
      }

      if (typeof chartjsPluginZoom !== 'undefined' && !isZoomRegistered) {
        Chart.register(chartjsPluginZoom);
      }
      if (typeof chartjsPluginAnnotation !== 'undefined' && !isAnnotationRegistered) {
        Chart.register(chartjsPluginAnnotation);
      }
    }
  } catch (e) {
    console.warn('Error checking/registering Chart.js plugins:', e);
  }

  // --- Toast Notifications ---
  function showToast(message, type = 'info') {
    toastMessage.textContent = message;
    toastNotification.className = `toast show ${type}`;
    
    // Clear previous timeouts if any
    if (window.toastTimeout) clearTimeout(window.toastTimeout);
    
    window.toastTimeout = setTimeout(() => {
      toastNotification.classList.remove('show');
    }, 3000);
  }

  // --- Loader State ---
  function showLoader(text = 'Processing data...') {
    loadingText.textContent = text;
    loadingOverlay.classList.add('show');
  }

  function hideLoader() {
    loadingOverlay.classList.remove('show');
  }

  // --- File Upload / Drag & Drop Handlers ---
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFile(files[0]);
    }
  });

  browseBtn.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  // --- Parse & Load CSV File ---
  function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
      showToast('Please upload a valid CSV file.', 'error');
      return;
    }

    currentFileName = file.name;
    showLoader(`Parsing ${file.name}...`);
    
    // Use Papa Parse to read CSV
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: function(results) {
        if (results.errors.length > 0) {
          console.warn('Papa Parse errors:', results.errors);
        }
        processData(results.data);
      },
      error: function(err) {
        hideLoader();
        showToast('Error parsing file: ' + err.message, 'error');
      }
    });
  }

  // --- Load Demo / Simulated Data ---
  sampleDataBtn.addEventListener('click', () => {
    currentFileName = 'demo_data.csv';
    showLoader('Generating demo sensor data...');
    setTimeout(() => {
      const demoData = generateDemoSensorData();
      processData(demoData);
      showToast('Demo data loaded successfully!', 'success');
    }, 500);
  });

  // --- Data Processor ---
  function processData(data) {
    if (!data || data.length === 0) {
      hideLoader();
      showToast('The loaded CSV file is empty.', 'error');
      return;
    }

    // Inspect columns
    const columns = Object.keys(data[0]);
    
    // Find time column
    const timeCol = columns.find(col => {
      const low = col.toLowerCase();
      return low.includes('time') || low === 't' || low === 's' || low === 'sec';
    });

    if (!timeCol) {
      hideLoader();
      showToast('Error: Could not identify a timestamp column (e.g. "Time (s)" or "timestamp").', 'error');
      return;
    }

    // Identify accel columns
    const accelXCol = columns.find(col => col.toLowerCase().includes('accel') && col.toLowerCase().includes('x'));
    const accelYCol = columns.find(col => col.toLowerCase().includes('accel') && col.toLowerCase().includes('y'));
    const accelZCol = columns.find(col => col.toLowerCase().includes('accel') && col.toLowerCase().includes('z'));
    
    // Identify gyro columns (optional)
    const gyroXCol = columns.find(col => col.toLowerCase().includes('gyro') && col.toLowerCase().includes('x'));
    const gyroYCol = columns.find(col => col.toLowerCase().includes('gyro') && col.toLowerCase().includes('y'));
    const gyroZCol = columns.find(col => col.toLowerCase().includes('gyro') && col.toLowerCase().includes('z'));
    
    // Identify speed column (optional)
    const speedCol = columns.find(col => col.toLowerCase().includes('speed') || col.toLowerCase().includes('vel'));

    // Check if we have at least Z-axis acceleration
    if (!accelZCol) {
      hideLoader();
      showToast('Error: Could not find an Accelerometer Z column. (e.g. "Acceleration z" or "accel_z").', 'error');
      return;
    }

    // Store raw data
    rawData = data;

    // Check if the time column is in seconds or milliseconds.
    // phyphox relative time is usually in seconds (e.g., 0.0, 0.02, ...).
    // System timestamp might be in milliseconds (e.g. 1686812400000).
    const sampleTimes = data.slice(0, 50).map(row => Number(row[timeCol]));
    const maxSampleTime = Math.max(...sampleTimes);
    const minSampleTime = Math.min(...sampleTimes);
    const isMilliseconds = (maxSampleTime - minSampleTime) > 200; // standard 50hz rate is 20ms per sample. If diff is > 200 over 50 rows, it's likely ms or unix.

    // Process and normalize values
    const chartLabels = [];
    const accelZ = [];
    const accelMag = [];
    const gyroPitch = []; // Show pitch for speed breaker tilt
    
    // Determine time offsets
    const baseTime = Number(data[0][timeCol]);

    for (let i = 0; i < data.length; i++) {
      const row = data[i];
      let timeSec = Number(row[timeCol]);
      if (isMilliseconds) {
        timeSec = (timeSec - baseTime) / 1000.0;
      } else if (timeSec > 1000000) {
        // Unix timestamp in seconds
        timeSec = timeSec - baseTime;
      }
      
      chartLabels.push(timeSec);
      
      const ax = accelXCol ? Number(row[accelXCol] || 0) : 0;
      const ay = accelYCol ? Number(row[accelYCol] || 0) : 0;
      const az = Number(row[accelZCol] || 9.81);
      
      accelZ.push(az);
      
      // Compute magnitude
      const mag = Math.sqrt(ax * ax + ay * ay + az * az);
      accelMag.push(mag);
      
      // Calculate a rough "gyro pitch" from Gyro y or z if available, or just Gyro Y
      let pitch = 0;
      if (gyroYCol) {
        pitch = Number(row[gyroYCol] || 0);
      }
      gyroPitch.push(pitch);
    }

    // Load labels from localStorage OR parse from CSV if not saved
    labeledRegions = [];
    let restoredFromLocalStorage = false;
    
    if (currentFileName) {
      const saved = localStorage.getItem(`labels_${currentFileName}`);
      if (saved) {
        try {
          labeledRegions = JSON.parse(saved);
          restoredFromLocalStorage = true;
        } catch (e) {
          console.error('Failed to parse labels from localStorage:', e);
        }
      }
    }

    // Fall back to pre-populating labeledRegions if CSV has label column and no localStorage copy exists
    if (!restoredFromLocalStorage && data[0].label !== undefined) {
      let currentLabel = null;
      let labelStart = null;
      
      for (let i = 0; i < data.length; i++) {
        const row = data[i];
        let timeSec = chartLabels[i];
        const rowLabel = row.label;
        
        if (rowLabel && rowLabel !== 'normal') {
          if (currentLabel === null) {
            currentLabel = rowLabel;
            labelStart = timeSec;
          } else if (rowLabel !== currentLabel) {
            // Label changed, save previous
            labeledRegions.push({
              id: generateId(),
              startTime: labelStart,
              endTime: chartLabels[i - 1],
              label: currentLabel
            });
            currentLabel = rowLabel;
            labelStart = timeSec;
          }
        } else {
          if (currentLabel !== null) {
            // Label ended
            labeledRegions.push({
              id: generateId(),
              startTime: labelStart,
              endTime: chartLabels[i - 1],
              label: currentLabel
            });
            currentLabel = null;
            labelStart = null;
          }
        }
      }
      // Capture trailing label
      if (currentLabel !== null) {
        labeledRegions.push({
          id: generateId(),
          startTime: labelStart,
          endTime: chartLabels[chartLabels.length - 1],
          label: currentLabel
        });
      }
      
      // Auto-save parsed labels to localStorage immediately
      saveToLocalStorage();
    }

    // Clear undo/redo stacks
    undoStack = [];
    redoStack = [];

    // Save processed chart data (decimate if dataset is too large to prevent canvas lag)
    let isDecimated = false;
    if (data.length > 15000) {
      chartData = decimateDataset(chartLabels, accelZ, accelMag, gyroPitch, 15000);
      isDecimated = true;
    } else {
      chartData = {
        labels: chartLabels,
        accelZ: accelZ,
        accelMag: accelMag,
        gyroPitch: gyroPitch
      };
    }

    // Update global UI count
    statTotalRows.textContent = data.length.toLocaleString();

    // Render chart
    renderChart();
    
    // Update labeled regions UI
    updateLabelsUI();
    
    // Make sure header counts are synchronized
    updateCounts();
    
    // Clear selection UI
    clearSelection();

    // Enable buttons
    exportCsvBtn.disabled = false;

    hideLoader();

    if (restoredFromLocalStorage) {
      showToast(`Loaded ${data.length} records. Restored saved labels.`, 'success');
    } else {
      showToast(`Loaded ${data.length} records successfully.`, 'success');
    }

    if (isDecimated) {
      console.log(`Data Decimation: Reduced rendering points from ${data.length} to ${chartData.labels.length} to prevent chart lag. Full resolution maintained for exporting.`);
    }
  }

  // --- Render Chart.js ---
  function renderChart() {
    // Show chart elements, hide placeholder
    chartPlaceholder.style.display = 'none';
    sensorChartCanvas.style.display = 'block';
    document.getElementById('chart-controls').style.display = 'flex';

    if (chartInstance) {
      chartInstance.destroy();
    }

    const ctx = sensorChartCanvas.getContext('2d');
    
    // Build datasets
    const datasets = [
      {
        label: 'Accel Z (Vertical m/s²)',
        data: chartData.accelZ,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.05)',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.1,
        yAxisID: 'y'
      },
      {
        label: 'Accel Magnitude (m/s²)',
        data: chartData.accelMag,
        borderColor: '#10b981',
        backgroundColor: 'transparent',
        borderWidth: 1.2,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.1,
        yAxisID: 'y',
        hidden: true // Hidden by default, let user toggle it
      },
      {
        label: 'Gyro Pitch/Roll (rad/s)',
        data: chartData.gyroPitch,
        borderColor: '#a855f7', // Purple
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        pointHoverRadius: 4,
        tension: 0.1,
        yAxisID: 'yGyro'
      }
    ];

    // Build annotations config
    const annotations = getAnnotationsConfig();

    // Configure options
    const options = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false, // Turn off animation for large performance boosts with datasets
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: 'Time (seconds)',
            color: '#94a3b8'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.04)'
          },
          ticks: {
            color: '#94a3b8',
            callback: function(value) {
              return value.toFixed(1) + 's';
            }
          }
        },
        y: {
          type: 'linear',
          position: 'left',
          title: {
            display: true,
            text: 'Acceleration (m/s²)',
            color: '#94a3b8'
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.06)'
          },
          ticks: {
            color: '#94a3b8'
          }
        },
        yGyro: {
          type: 'linear',
          position: 'right',
          title: {
            display: true,
            text: 'Angular Velocity (rad/s)',
            color: '#94a3b8'
          },
          grid: {
            drawOnChartArea: false // Avoid cluttering main grid
          },
          ticks: {
            color: '#94a3b8'
          }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: '#f8fafc',
            font: {
              family: 'Inter',
              size: 11
            }
          }
        },
        zoom: {
          zoom: {
            wheel: {
              enabled: true
            },
            pinch: {
              enabled: true
            },
            mode: zoomMode,
            onZoomComplete: function({chart}) {
              // Can inspect zoom bounds here if needed
            }
          },
          pan: {
            enabled: true,
            mode: 'x', // Lock panning to horizontal axis
            modifierKey: null
          }
        },
        annotation: {
          annotations: annotations
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            title: function(tooltipItems) {
              return `Time: ${Number(tooltipItems[0].parsed.x).toFixed(3)} s`;
            }
          }
        }
      },
      onClick: handleChartClick
    };

    chartInstance = new Chart(ctx, {
      type: 'line',
      data: {
        labels: chartData.labels,
        datasets: datasets
      },
      options: options
    });
  }

  // --- Zoom Mode Toggles ---
  btnZoomX.addEventListener('click', () => {
    zoomMode = 'x';
    btnZoomX.classList.add('active');
    btnZoomXY.classList.remove('active');
    if (chartInstance) {
      chartInstance.options.plugins.zoom.zoom.mode = 'x';
      chartInstance.update();
    }
  });

  btnZoomXY.addEventListener('click', () => {
    zoomMode = 'xy';
    btnZoomXY.classList.add('active');
    btnZoomX.classList.remove('active');
    if (chartInstance) {
      chartInstance.options.plugins.zoom.zoom.mode = 'xy';
      chartInstance.update();
    }
  });

  resetZoomBtn.addEventListener('click', () => {
    if (chartInstance) {
      chartInstance.resetZoom();
    }
  });

  // --- Chart Click Event (Initiates selection) ---
  function handleChartClick(e) {
    if (!chartInstance) return;
    
    // Get nearest index on scale
    const points = chartInstance.getElementsAtEventForMode(e, 'nearest', { intersect: false }, true);
    let clickedTime = null;
    
    if (points.length > 0) {
      const idx = points[0].index;
      clickedTime = chartData.labels[idx];
    } else {
      // Find from coordinates if clicked outside active data points
      const canvasPosition = Chart.helpers.getRelativePosition(e, chartInstance);
      clickedTime = chartInstance.scales.x.getValueForPixel(canvasPosition.x);
    }

    if (clickedTime === null || clickedTime < 0 || clickedTime > chartData.labels[chartData.labels.length - 1]) {
      return;
    }

    // Set selection anchor
    selectionAnchor = clickedTime;
    
    // Enable selection width inputs
    inputSelectionWidth.disabled = false;
    
    // Update range based on width
    updateSelectionRange();
  }

  // --- Selection Range Management ---
  inputSelectionWidth.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value);
    valSelectionWidth.textContent = val.toFixed(2);
    if (selectionAnchor !== null) {
      updateSelectionRange();
    }
  });

  function updateSelectionRange() {
    if (selectionAnchor === null) return;
    
    const width = parseFloat(inputSelectionWidth.value);
    const start = Math.max(0, selectionAnchor - (width / 2));
    const end = Math.min(chartData.labels[chartData.labels.length - 1], selectionAnchor + (width / 2));
    
    activeSelection = {
      startTime: start,
      endTime: end
    };

    // Update Input UI
    inputStartTime.value = start.toFixed(2);
    inputEndTime.value = end.toFixed(2);
    
    inputStartTime.disabled = false;
    inputEndTime.disabled = false;

    // Enable labeling action buttons
    btnLabelPothole.disabled = false;
    btnLabelSpeedbreaker.disabled = false;
    btnClearSelection.disabled = false;

    // Refresh chart annotations to show current active selection box
    refreshAnnotations();
  }

  // Manually editing inputs
  inputStartTime.addEventListener('change', handleManualTimeChange);
  inputEndTime.addEventListener('change', handleManualTimeChange);

  function handleManualTimeChange() {
    if (activeSelection === null) return;
    
    let start = parseFloat(inputStartTime.value);
    let end = parseFloat(inputEndTime.value);
    const maxTime = chartData.labels[chartData.labels.length - 1];

    if (isNaN(start) || start < 0) start = 0;
    if (isNaN(end) || end > maxTime) end = maxTime;
    if (start > end) {
      // Swap if out of order
      const temp = start;
      start = end;
      end = temp;
    }

    activeSelection = { startTime: start, endTime: end };
    selectionAnchor = (start + end) / 2; // Midpoint becomes new anchor
    
    const newWidth = end - start;
    inputSelectionWidth.value = newWidth;
    valSelectionWidth.textContent = newWidth.toFixed(2);

    inputStartTime.value = start.toFixed(2);
    inputEndTime.value = end.toFixed(2);

    refreshAnnotations();
  }

  function clearSelection() {
    selectionAnchor = null;
    activeSelection = null;
    selectedLabelId = null;

    inputStartTime.value = '';
    inputEndTime.value = '';
    inputStartTime.disabled = true;
    inputEndTime.disabled = true;
    inputSelectionWidth.disabled = true;
    
    btnLabelPothole.disabled = true;
    btnLabelSpeedbreaker.disabled = true;
    btnClearSelection.disabled = true;

    // Remove selection row active class
    const rows = labelsTableBody.querySelectorAll('tr');
    rows.forEach(r => r.classList.remove('selected'));

    refreshAnnotations();
  }

  btnClearSelection.addEventListener('click', clearSelection);

  // --- Annotations Config Generator ---
  function getAnnotationsConfig() {
    const annotationsObj = {};
    
    // Add existing labels
    labeledRegions.forEach((region, index) => {
      const isPothole = region.label === 'pothole';
      const color = isPothole ? 'rgba(239, 68, 68, 0.15)' : 'rgba(245, 158, 11, 0.15)';
      const border = isPothole ? '#ef4444' : '#f59e0b';
      const labelText = isPothole ? 'POTHOLE' : 'SPEED BREAKER';
      
      annotationsObj[`label_${region.id}`] = {
        type: 'box',
        xMin: region.startTime,
        xMax: region.endTime,
        backgroundColor: color,
        borderColor: border,
        borderWidth: 1.5,
        label: {
          display: true,
          content: labelText,
          color: border,
          font: {
            size: 9,
            weight: 'bold',
            family: 'Outfit'
          },
          position: 'start'
        }
      };
    });

    // Add active selection box
    if (activeSelection !== null) {
      annotationsObj['active_selection'] = {
        type: 'box',
        xMin: activeSelection.startTime,
        xMax: activeSelection.endTime,
        backgroundColor: 'rgba(59, 130, 246, 0.25)',
        borderColor: '#3b82f6',
        borderWidth: 2,
        borderDash: [5, 5],
        label: {
          display: true,
          content: 'SELECTING...',
          color: '#3b82f6',
          font: {
            size: 10,
            weight: 'bold',
            family: 'Outfit'
          },
          position: 'center'
        }
      };
    }

    return annotationsObj;
  }

  function refreshAnnotations() {
    if (chartInstance) {
      chartInstance.options.plugins.annotation.annotations = getAnnotationsConfig();
      chartInstance.update();
    }
  }

  // --- Save / Create Label ---
  btnLabelPothole.addEventListener('click', () => saveLabel('pothole'));
  btnLabelSpeedbreaker.addEventListener('click', () => saveLabel('speed_breaker'));

  function saveLabel(labelType) {
    if (activeSelection === null) return;

    // Keep history for undo
    pushHistory();

    const newLabel = {
      id: generateId(),
      startTime: activeSelection.startTime,
      endTime: activeSelection.endTime,
      label: labelType
    };

    // Handle label overlaps (truncating/splitting previous labels)
    resolveOverlaps(newLabel.startTime, newLabel.endTime);

    // Add new label
    labeledRegions.push(newLabel);
    
    // Sort labels by start time
    labeledRegions.sort((a, b) => a.startTime - b.startTime);

    // Update counts
    updateCounts();

    // Auto-save to localStorage
    saveToLocalStorage();

    // Redraw
    updateLabelsUI();
    clearSelection();
    
    showToast(`Labeled region (${newLabel.startTime.toFixed(2)}s - ${newLabel.endTime.toFixed(2)}s) as ${labelType.replace('_', ' ')}.`, 'success');
  }

  // --- Resolve Overlapping Labels ---
  function resolveOverlaps(newStart, newEnd) {
    const cleaned = [];

    labeledRegions.forEach(existing => {
      const estart = existing.startTime;
      const eend = existing.endTime;

      // 1. Completely inside the new label -> delete it (do not add to cleaned list)
      if (estart >= newStart && eend <= newEnd) {
        return; 
      }
      
      // 2. Completely covers the new label -> split it into two sections
      if (estart < newStart && eend > newEnd) {
        cleaned.push({
          id: generateId(),
          startTime: estart,
          endTime: newStart - 0.001,
          label: existing.label
        });
        cleaned.push({
          id: generateId(),
          startTime: newEnd + 0.001,
          endTime: eend,
          label: existing.label
        });
        return;
      }
      
      // 3. Overlaps start of new label -> truncate its end time
      if (estart < newStart && eend >= newStart && eend <= newEnd) {
        cleaned.push({
          id: existing.id,
          startTime: estart,
          endTime: newStart - 0.001,
          label: existing.label
        });
        return;
      }
      
      // 4. Overlaps end of new label -> truncate its start time
      if (estart >= newStart && estart <= newEnd && eend > newEnd) {
        cleaned.push({
          id: existing.id,
          startTime: newEnd + 0.001,
          endTime: eend,
          label: existing.label
        });
        return;
      }

      // No overlap -> keep as is
      cleaned.push(existing);
    });

    labeledRegions = cleaned;
  }

  // --- Delete Label ---
  function deleteLabel(id) {
    pushHistory();
    const index = labeledRegions.findIndex(r => r.id === id);
    if (index !== -1) {
      const deleted = labeledRegions.splice(index, 1)[0];
      updateCounts();
      updateLabelsUI();
      saveToLocalStorage();
      if (activeSelection === null) {
        refreshAnnotations();
      }
      showToast(`Removed ${deleted.label.replace('_', ' ')} label.`, 'info');
    }
  }

  // --- Update Labels UI Table List ---
  function updateLabelsUI() {
    labelsTableBody.innerHTML = '';
    
    if (labeledRegions.length === 0) {
      labelsTableBody.innerHTML = `
        <tr>
          <td colspan="3" class="no-labels">No regions labeled yet.</td>
        </tr>
      `;
      return;
    }

    labeledRegions.forEach(region => {
      const tr = document.createElement('tr');
      tr.dataset.id = region.id;
      
      const isPothole = region.label === 'pothole';
      const labelBadge = isPothole 
        ? `<span class="badge pothole">Pothole</span>` 
        : `<span class="badge speed_breaker">Speed Breaker</span>`;
      
      const duration = region.endTime - region.startTime;

      tr.innerHTML = `
        <td><strong>${region.startTime.toFixed(2)}s - ${region.endTime.toFixed(2)}s</strong><br><span style="font-size:10px; color:var(--text-muted)">Dur: ${duration.toFixed(2)}s</span></td>
        <td style="vertical-align: middle;">${labelBadge}</td>
        <td style="text-align: center; vertical-align: middle;">
          <button class="delete-btn" title="Delete label">🗑️</button>
        </td>
      `;

      // Jump to region on double-click or click
      tr.addEventListener('click', (e) => {
        if (e.target.classList.contains('delete-btn')) {
          deleteLabel(region.id);
          return;
        }

        // Highlight selection row
        labelsTableBody.querySelectorAll('tr').forEach(r => r.classList.remove('selected'));
        tr.classList.add('selected');
        
        selectedLabelId = region.id;
        
        // Pan and Zoom chart to this label (focus with a 1.5-second buffer on each side)
        if (chartInstance) {
          chartInstance.options.scales.x.min = Math.max(0, region.startTime - 1.5);
          chartInstance.options.scales.x.max = Math.min(chartData.labels[chartData.labels.length - 1], region.endTime + 1.5);
          chartInstance.update();
        }
      });

      labelsTableBody.appendChild(tr);
    });
  }

  // --- Update Label Counts ---
  function updateCounts() {
    const potholeCount = labeledRegions.filter(r => r.label === 'pothole').length;
    const speedBreakerCount = labeledRegions.filter(r => r.label === 'speed_breaker').length;
    
    statPotholes.textContent = potholeCount;
    statSpeedBreakers.textContent = speedBreakerCount;
  }

  // --- Undo/Redo States ---
  function pushHistory() {
    // Save deep copy of labeledRegions array
    undoStack.push(JSON.stringify(labeledRegions));
    // Clear redo stack on new action
    redoStack = [];
  }

  function undo() {
    if (undoStack.length === 0) {
      showToast('Nothing to undo', 'info');
      return;
    }
    
    redoStack.push(JSON.stringify(labeledRegions));
    labeledRegions = JSON.parse(undoStack.pop());
    
    updateCounts();
    updateLabelsUI();
    clearSelection();
    saveToLocalStorage();
    showToast('Undo completed', 'info');
  }

  function redo() {
    if (redoStack.length === 0) {
      showToast('Nothing to redo', 'info');
      return;
    }
    
    undoStack.push(JSON.stringify(labeledRegions));
    labeledRegions = JSON.parse(redoStack.pop());
    
    updateCounts();
    updateLabelsUI();
    clearSelection();
    saveToLocalStorage();
    showToast('Redo completed', 'info');
  }

  // --- Keyboard Shortcuts ---
  window.addEventListener('keydown', (e) => {
    // If typing inside an input field, do not execute global keyboard shortcuts
    if (document.activeElement.tagName === 'INPUT') {
      if (e.key === 'Escape') {
        document.activeElement.blur();
        clearSelection();
      }
      return;
    }

    const key = e.key.toLowerCase();
    
    // Labeling bindings
    if (activeSelection !== null) {
      if (key === 'p') {
        saveLabel('pothole');
      } else if (key === 's') {
        saveLabel('speed_breaker');
      } else if (e.key === 'Escape') {
        clearSelection();
      }
    }

    // List navigation bindings (Delete key)
    if (key === 'delete' && selectedLabelId !== null) {
      deleteLabel(selectedLabelId);
      selectedLabelId = null;
    }

    // Undo / Redo bindings (Ctrl+Z / Ctrl+Y)
    if (e.ctrlKey) {
      if (key === 'z') {
        e.preventDefault();
        undo();
      } else if (key === 'y') {
        e.preventDefault();
        redo();
      }
    }
  });

  // --- Export Labeled CSV ---
  exportCsvBtn.addEventListener('click', () => {
    if (rawData.length === 0) return;

    showLoader('Preparing CSV for download...');

    setTimeout(() => {
      // Find time column
      const columns = Object.keys(rawData[0]);
      const timeCol = columns.find(col => {
        const low = col.toLowerCase();
        return low.includes('time') || low === 't' || low === 's' || low === 'sec';
      });

      // Find relative scale
      const sampleTimes = rawData.slice(0, 50).map(row => Number(row[timeCol]));
      const maxSampleTime = Math.max(...sampleTimes);
      const minSampleTime = Math.min(...sampleTimes);
      const isMilliseconds = (maxSampleTime - minSampleTime) > 200;
      const baseTime = Number(rawData[0][timeCol]);

      // Add label column mapping to each row
      const labeledCsvData = rawData.map((row, index) => {
        // Calculate the row's relative timestamp in seconds
        let timeSec = Number(row[timeCol]);
        if (isMilliseconds) {
          timeSec = (timeSec - baseTime) / 1000.0;
        } else if (timeSec > 1000000) {
          timeSec = timeSec - baseTime;
        }

        // Determine if it falls inside any labeled region
        const matchingRegion = labeledRegions.find(region => {
          return timeSec >= region.startTime && timeSec <= region.endTime;
        });

        const label = matchingRegion ? matchingRegion.label : 'normal';
        
        return {
          ...row,
          label: label
        };
      });

      // Unparse array of objects back into CSV string
      const csvString = Papa.unparse(labeledCsvData);
      
      // Trigger browser download
      const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      
      let exportName = 'session_labeled.csv';
      if (currentFileName) {
        const dotIdx = currentFileName.lastIndexOf('.');
        if (dotIdx !== -1) {
          exportName = currentFileName.substring(0, dotIdx) + '_labeled' + currentFileName.substring(dotIdx);
        } else {
          exportName = currentFileName + '_labeled.csv';
        }
      }
      link.setAttribute('download', exportName);
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      hideLoader();
      showToast('Labeled CSV exported successfully!', 'success');
    }, 500);
  });

  // --- Utility Helpers ---
  function generateId() {
    return Math.random().toString(36).substring(2, 9);
  }

  // --- Auto-Save to LocalStorage ---
  function saveToLocalStorage() {
    if (currentFileName) {
      localStorage.setItem(`labels_${currentFileName}`, JSON.stringify(labeledRegions));
    }
  }

  // --- Peak-Preserving Min-Max Data Decimation ---
  function decimateDataset(labels, accelZ, accelMag, gyroPitch, targetPoints = 15000) {
    const length = labels.length;
    if (length <= targetPoints) {
      return { labels, accelZ, accelMag, gyroPitch };
    }
    
    const decimatedLabels = [];
    const decimatedAccelZ = [];
    const decimatedAccelMag = [];
    const decimatedGyroPitch = [];
    
    // We group points into buckets, and for each bucket we keep the min and max
    // to preserve all peak values (pothole spikes and speed breaker curves)
    const bucketSize = Math.ceil(length / targetPoints);
    
    for (let i = 0; i < length; i += bucketSize) {
      const end = Math.min(i + bucketSize, length);
      
      // Find min/max indices in this bucket based on vertical acceleration (accelZ)
      let minIdx = i;
      let maxIdx = i;
      let minVal = accelZ[i];
      let maxVal = accelZ[i];
      
      for (let j = i + 1; j < end; j++) {
        if (accelZ[j] < minVal) {
          minVal = accelZ[j];
          minIdx = j;
        }
        if (accelZ[j] > maxVal) {
          maxVal = accelZ[j];
          maxIdx = j;
        }
      }
      
      const addPoint = (idx) => {
        decimatedLabels.push(labels[idx]);
        decimatedAccelZ.push(accelZ[idx]);
        decimatedAccelMag.push(accelMag[idx]);
        decimatedGyroPitch.push(gyroPitch[idx]);
      };

      // Add the min and max points in their original chronological order
      if (minIdx < maxIdx) {
        addPoint(minIdx);
        addPoint(maxIdx);
      } else if (minIdx > maxIdx) {
        addPoint(maxIdx);
        addPoint(minIdx);
      } else {
        addPoint(minIdx);
      }
    }
    
    return {
      labels: decimatedLabels,
      accelZ: decimatedAccelZ,
      accelMag: decimatedAccelMag,
      gyroPitch: decimatedGyroPitch
    };
  }

  // --- Simulated Sensor Data Generator ---
  function generateDemoSensorData() {
    const rateHz = 50; // 50 samples per second
    const durationSec = 60; // 60 seconds of data
    const totalSamples = rateHz * durationSec;
    const data = [];

    // Inject anomalies configurations
    const anomalies = [
      { start: 12.0, end: 12.4, type: 'pothole' },
      { start: 22.0, end: 23.8, type: 'speed_breaker' },
      { start: 35.2, end: 35.6, type: 'pothole' },
      { start: 45.5, end: 47.3, type: 'speed_breaker' },
      { start: 54.0, end: 54.4, type: 'pothole' }
    ];

    for (let i = 0; i < totalSamples; i++) {
      const time = i / rateHz;
      
      // Normal road baseline values + noise
      let accel_x = 0.0 + (Math.random() - 0.5) * 0.15;
      let accel_y = 0.0 + (Math.random() - 0.5) * 0.15;
      let accel_z = 9.81 + (Math.random() - 0.5) * 0.3; // 9.81 m/s² due to gravity
      let gyro_x = (Math.random() - 0.5) * 0.04;
      let gyro_y = (Math.random() - 0.5) * 0.04;
      let gyro_z = (Math.random() - 0.5) * 0.02;
      let speed = 35.0 + (Math.random() - 0.5) * 2; // ~35 km/h

      // Inject anomalies if current timestamp falls within windows
      const activeAnomaly = anomalies.find(a => time >= a.start && time <= a.end);
      
      if (activeAnomaly) {
        const progress = (time - activeAnomaly.start) / (activeAnomaly.end - activeAnomaly.start);
        
        if (activeAnomaly.type === 'pothole') {
          // Pothole: Violent, fast vertical oscillations (Z-axis drops, spikes, then decays)
          const frequency = 25; // Hz
          const amplitude = 12.0;
          const decay = Math.exp(-progress * 5); // decay vertical acceleration
          
          accel_z += Math.sin(progress * Math.PI * 2 * frequency) * amplitude * decay;
          // Sudden lateral jerk
          accel_x += (Math.sin(progress * Math.PI * 8) * 3.5) * decay;
          // High speed jitter on Gyro pitch
          gyro_y += Math.sin(progress * Math.PI * 20) * 0.8 * decay;
          // Decelerate slightly on hitting pothole
          speed -= 4 * progress;
        } 
        else if (activeAnomaly.type === 'speed_breaker') {
          // Speed Breaker: Smooth wide peak on Z-axis + corresponding smooth pitch change (Gyro Y)
          const wave = Math.sin(progress * Math.PI); // Half sine wave
          const waveZ = Math.sin(progress * Math.PI * 2); // Full cycle for Z-axis bounce
          
          accel_z += waveZ * 4.5;
          // Smooth rotation (first nose up, then nose down)
          gyro_y += Math.cos(progress * Math.PI) * 0.45;
          // Significant deceleration when crossing hump
          speed -= 12 * Math.sin(progress * Math.PI);
        }
      }

      data.push({
        'Time (s)': time,
        'Acceleration x (m/s^2)': accel_x,
        'Acceleration y (m/s^2)': accel_y,
        'Acceleration z (m/s^2)': accel_z,
        'Gyroscope x (rad/s)': gyro_x,
        'Gyroscope y (rad/s)': gyro_y,
        'Gyroscope z (rad/s)': gyro_z,
        'Speed (km/h)': speed
      });
    }

    return data;
  }
});
