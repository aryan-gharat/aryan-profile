// Wait for DOM to load
document.addEventListener('DOMContentLoaded', () => {
  // Elements Selection
  const selectPreset = document.getElementById('function-preset');
  const inputW1 = document.getElementById('input-w1');
  const inputW2 = document.getElementById('input-w2');
  const inputBias = document.getElementById('input-bias');
  const inputLr = document.getElementById('input-lr');
  const inputEpochs = document.getElementById('input-epochs');
  const btnStep = document.getElementById('btn-step');
  const btnTrain = document.getElementById('btn-train');
  const btnReset = document.getElementById('btn-reset');
  const btnAddRow = document.getElementById('btn-add-row');
  const trainingDataTable = document.getElementById('training-data-table').querySelector('tbody');
  const canvas = document.getElementById('plot-canvas');
  const ctx = canvas.getContext('2d');
  
  // Stat values elements
  const statEpoch = document.getElementById('stat-epoch');
  const statWeights = document.getElementById('stat-weights');
  const statBias = document.getElementById('stat-bias');
  const statStatus = document.getElementById('stat-status');
  
  // Log list elements
  const logsBody = document.getElementById('logs-body');

  // Chatbot Elements
  const chatbotContainer = document.getElementById('chatbot-container');
  const chatbotToggle = document.getElementById('chatbot-toggle');
  const chatClose = document.getElementById('chat-close');
  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const chatSend = document.getElementById('chat-send');
  const chatChipsContainer = document.getElementById('chat-chips');
  const chatBadge = document.getElementById('chat-badge');

  // Perceptron State Variables
  let weights = [0.3, -0.2];
  let bias = -0.4;
  let learningRate = 0.1;
  let maxEpochs = 20;
  let currentEpoch = 0;
  let converged = false;
  let trainingData = [];
  let isTrainingInProgress = false;
  let trainingInterval = null;

  // Preset Configurations
  const presets = {
    and: [
      { inputs: [0, 0], target: 0 },
      { inputs: [0, 1], target: 0 },
      { inputs: [1, 0], target: 0 },
      { inputs: [1, 1], target: 1 }
    ],
    or: [
      { inputs: [0, 0], target: 0 },
      { inputs: [0, 1], target: 1 },
      { inputs: [1, 0], target: 1 },
      { inputs: [1, 1], target: 1 }
    ],
    custom: [
      { inputs: [0.2, 0.3], target: 0 },
      { inputs: [0.8, 0.1], target: 0 },
      { inputs: [0.3, 0.9], target: 1 },
      { inputs: [0.9, 0.8], target: 1 }
    ]
  };

  // Canvas Setup coordinates mapping: [-0.5, 1.5]
  const minCoord = -0.5;
  const maxCoord = 1.5;

  function toCanvasX(x) {
    return ((x - minCoord) / (maxCoord - minCoord)) * canvas.width;
  }

  function toCanvasY(y) {
    // Canvas coordinate system has Y pointing down, so we invert Y
    return canvas.height - ((y - minCoord) / (maxCoord - minCoord)) * canvas.height;
  }

  function toGraphX(cx) {
    return minCoord + (cx / canvas.width) * (maxCoord - minCoord);
  }

  function toGraphY(cy) {
    return minCoord + ((canvas.height - cy) / canvas.height) * (maxCoord - minCoord);
  }

  // --- INITIALIZATION ---
  function init() {
    loadPreset(selectPreset.value);
    resetSimulation();
    setupEventListeners();
    triggerGreetingBadge();
  }

  // Load target functions into the table
  function loadPreset(presetName) {
    trainingData = JSON.parse(JSON.stringify(presets[presetName]));
    renderTrainingDataTable();
    
    if (presetName === 'custom') {
      btnAddRow.style.display = 'inline-block';
    } else {
      btnAddRow.style.display = 'none';
    }
  }

  // Populate data table from state
  function renderTrainingDataTable() {
    trainingDataTable.innerHTML = '';
    trainingData.forEach((row, index) => {
      const tr = document.createElement('tr');
      
      const tdX1 = document.createElement('td');
      if (selectPreset.value === 'custom') {
        tdX1.innerHTML = `<input type="number" step="0.1" value="${row.inputs[0]}" data-idx="${index}" data-param="x1" class="table-input">`;
      } else {
        tdX1.textContent = row.inputs[0];
      }
      
      const tdX2 = document.createElement('td');
      if (selectPreset.value === 'custom') {
        tdX2.innerHTML = `<input type="number" step="0.1" value="${row.inputs[1]}" data-idx="${index}" data-param="x2" class="table-input">`;
      } else {
        tdX2.textContent = row.inputs[1];
      }
      
      const tdTarget = document.createElement('td');
      if (selectPreset.value === 'custom') {
        tdTarget.innerHTML = `
          <select data-idx="${index}" data-param="target" class="table-select">
            <option value="0" ${row.target === 0 ? 'selected' : ''}>0</option>
            <option value="1" ${row.target === 1 ? 'selected' : ''}>1</option>
          </select>
          <button type="button" class="btn-delete-row" data-idx="${index}">×</button>
        `;
      } else {
        tdTarget.textContent = row.target;
      }
      
      tr.appendChild(tdX1);
      tr.appendChild(tdX2);
      tr.appendChild(tdTarget);
      trainingDataTable.appendChild(tr);
    });
  }

  // Reset parameters and state
  function resetSimulation() {
    stopAutomaticTraining();
    weights = [parseFloat(inputW1.value), parseFloat(inputW2.value)];
    bias = parseFloat(inputBias.value);
    learningRate = parseFloat(inputLr.value);
    maxEpochs = parseInt(inputEpochs.value);
    currentEpoch = 0;
    converged = false;

    // UI Updates
    statEpoch.textContent = '0';
    updateStatsUI();
    
    logsBody.innerHTML = `
      <tr>
        <td colspan="8" class="empty-state">No steps run yet. Click "Step Epoch" or "Train Fully" to start.</td>
      </tr>
    `;
    
    drawPlot();
  }

  function updateStatsUI() {
    statWeights.textContent = `${weights[0].toFixed(3)}, ${weights[1].toFixed(3)}`;
    statBias.textContent = bias.toFixed(3);
    
    if (converged) {
      statStatus.textContent = 'Yes';
      statStatus.className = 'stat-value status-badge converged';
    } else {
      statStatus.textContent = 'No';
      statStatus.className = 'stat-value status-badge not-converged';
    }
  }

  // --- PERCEPTRON LEARNING ALGORITHM ---
  
  // Predict using Step Function activation
  function predict(inputs) {
    const netInput = inputs[0] * weights[0] + inputs[1] * weights[1] + bias;
    return netInput >= 0 ? 1 : 0;
  }

  // Execute one epoch of the training algorithm
  function trainStep() {
    if (converged || currentEpoch >= maxEpochs) {
      return false;
    }

    currentEpoch++;
    statEpoch.textContent = currentEpoch;
    
    let errorCount = 0;
    const epochLogs = [];

    // Clear logs table empty state if it's the first log
    if (currentEpoch === 1) {
      logsBody.innerHTML = '';
    }

    // Process each sample sequentially
    trainingData.forEach((sample) => {
      const inputs = sample.inputs;
      const target = sample.target;
      
      // Calculate net input (z)
      const netInput = inputs[0] * weights[0] + inputs[1] * weights[1] + bias;
      // Predict output (y)
      const output = netInput >= 0 ? 1 : 0;
      // Calculate error (e = t - y)
      const error = target - output;

      let deltaW1 = 0;
      let deltaW2 = 0;
      let deltaBias = 0;

      if (error !== 0) {
        errorCount++;
        // Weight updates: delta = eta * error * input
        deltaW1 = learningRate * error * inputs[0];
        deltaW2 = learningRate * error * inputs[1];
        deltaBias = learningRate * error;

        // Apply updates
        weights[0] += deltaW1;
        weights[1] += deltaW2;
        bias += deltaBias;
      }

      epochLogs.push({
        epoch: currentEpoch,
        inputs: [...inputs],
        target: target,
        netInput: netInput,
        output: output,
        error: error,
        deltas: [deltaW1, deltaW2, deltaBias],
        newWeights: [...weights],
        newBias: bias
      });
    });

    // Write to logs UI
    writeEpochLogsToUI(epochLogs);

    // Update Statuses
    if (errorCount === 0) {
      converged = true;
    }
    
    updateStatsUI();
    drawPlot();

    return !converged && currentEpoch < maxEpochs;
  }

  function writeEpochLogsToUI(epochLogs) {
    epochLogs.forEach((log) => {
      const tr = document.createElement('tr');
      if (log.error !== 0) {
        tr.className = 'highlight-row';
      }

      // Format weights delta string
      const deltaStr = log.error !== 0 
        ? `Δw₁: ${log.deltas[0].toFixed(2)}, Δw₂: ${log.deltas[1].toFixed(2)}, Δb: ${log.deltas[2].toFixed(2)}`
        : '0.00 (No change)';

      // Format weights string
      const weightsStr = `w: [${log.newWeights[0].toFixed(2)}, ${log.newWeights[1].toFixed(2)}], b: ${log.newBias.toFixed(2)}`;

      tr.innerHTML = `
        <td>${log.epoch}</td>
        <td>(${log.inputs[0]}, ${log.inputs[1]})</td>
        <td>${log.target}</td>
        <td>${log.netInput.toFixed(3)}</td>
        <td>${log.output}</td>
        <td><span class="error-val ${log.error === 0 ? 'no-err' : 'has-err'}">${log.error}</span></td>
        <td>${deltaStr}</td>
        <td>${weightsStr}</td>
      `;

      // Insert at the top of logs for ease of reading
      logsBody.insertBefore(tr, logsBody.firstChild);
    });
  }

  // --- DRAWING CANVAS DECISION BOUNDARY ---
  function drawPlot() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 1. Draw heatmap background representing classification regions
    const step = 6;
    for (let px = 0; px < canvas.width; px += step) {
      for (let py = 0; py < canvas.height; py += step) {
        const x1 = toGraphX(px);
        const x2 = toGraphY(py);
        const z = x1 * weights[0] + x2 * weights[1] + bias;
        
        ctx.fillStyle = z >= 0 
          ? 'rgba(16, 185, 129, 0.05)' // green glow for class 1
          : 'rgba(239, 68, 68, 0.05)'; // red glow for class 0
        ctx.fillRect(px, py, step, step);
      }
    }

    // 2. Draw grid lines & axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    
    // Vertical grids
    for (let g = minCoord; g <= maxCoord; g += 0.5) {
      const cx = toCanvasX(g);
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, canvas.height);
      ctx.stroke();

      // Axis labels text
      ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.font = '10px Space Grotesk';
      ctx.fillText(g.toFixed(1), cx + 5, canvas.height - 5);
    }

    // Horizontal grids
    for (let g = minCoord; g <= maxCoord; g += 0.5) {
      const cy = toCanvasY(g);
      ctx.beginPath();
      ctx.moveTo(0, cy);
      ctx.lineTo(canvas.width, cy);
      ctx.stroke();

      ctx.fillText(g.toFixed(1), 5, cy - 5);
    }

    // Draw main origin axes (X1=0, X2=0) in stronger lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1.5;
    
    // Axis Y
    ctx.beginPath();
    ctx.moveTo(toCanvasX(0), 0);
    ctx.lineTo(toCanvasX(0), canvas.height);
    ctx.stroke();

    // Axis X
    ctx.beginPath();
    ctx.moveTo(0, toCanvasY(0));
    ctx.lineTo(canvas.width, toCanvasY(0));
    ctx.stroke();

    // Label coordinates axes
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.font = '12px Space Grotesk';
    ctx.fillText('X₁', canvas.width - 20, toCanvasY(0) - 8);
    ctx.fillText('X₂', toCanvasX(0) + 8, 20);

    // 3. Draw decision boundary line: w1*x1 + w2*x2 + b = 0
    // x2 = -(w1*x1 + b) / w2
    if (weights[1] !== 0) {
      const calcX2 = (x1) => -(weights[0] * x1 + bias) / weights[1];
      
      const xStart = minCoord;
      const yStart = calcX2(xStart);
      
      const xEnd = maxCoord;
      const yEnd = calcX2(xEnd);

      ctx.strokeStyle = '#e0f2fe';
      ctx.lineWidth = 3;
      ctx.shadowBlur = 10;
      ctx.shadowColor = '#0284c7';
      ctx.beginPath();
      ctx.moveTo(toCanvasX(xStart), toCanvasY(yStart));
      ctx.lineTo(toCanvasX(xEnd), toCanvasY(yEnd));
      ctx.stroke();
      ctx.shadowBlur = 0; // reset shadow
    } else if (weights[0] !== 0) {
      // vertical line w2 = 0 => x1 = -b/w1
      const xLine = -bias / weights[0];
      ctx.strokeStyle = '#e0f2fe';
      ctx.lineWidth = 3;
      ctx.shadowBlur = 10;
      ctx.shadowColor = '#0284c7';
      ctx.beginPath();
      ctx.moveTo(toCanvasX(xLine), 0);
      ctx.lineTo(toCanvasX(xLine), canvas.height);
      ctx.stroke();
      ctx.shadowBlur = 0; // reset shadow
    }

    // 4. Draw data points
    trainingData.forEach((sample) => {
      const cx = toCanvasX(sample.inputs[0]);
      const cy = toCanvasY(sample.inputs[1]);
      const prediction = predict(sample.inputs);
      const isError = prediction !== sample.target;

      // Draw prediction halo if there is classification error
      if (isError) {
        ctx.strokeStyle = 'rgba(239, 68, 68, 0.6)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(cx, cy, 14, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = 'rgba(239, 68, 68, 0.3)';
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.arc(cx, cy, 18, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Point core circle
      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = sample.target === 1 ? '#10b981' : '#ef4444';
      ctx.shadowBlur = 6;
      ctx.shadowColor = sample.target === 1 ? '#10b981' : '#ef4444';
      ctx.fill();
      ctx.shadowBlur = 0; // reset shadow

      // Point outer white border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Print text labels for coordinates next to points
      ctx.fillStyle = '#9ca3af';
      ctx.font = '10px Plus Jakarta Sans';
      ctx.fillText(`(${sample.inputs[0]}, ${sample.inputs[1]})`, cx + 12, cy + 4);
    });
  }

  // --- AUTOMATIC TRAINING CONTROLLER ---
  function startAutomaticTraining() {
    isTrainingInProgress = true;
    btnTrain.textContent = 'Pause Training';
    btnTrain.className = 'btn btn-primary-outline';
    btnStep.disabled = true;

    trainingInterval = setInterval(() => {
      const keepGoing = trainStep();
      if (!keepGoing) {
        stopAutomaticTraining();
      }
    }, 400); // 400ms step updates for visual flow
  }

  function stopAutomaticTraining() {
    isTrainingInProgress = false;
    btnTrain.textContent = 'Train Fully';
    btnTrain.className = 'btn btn-primary';
    btnStep.disabled = false;
    if (trainingInterval) {
      clearInterval(trainingInterval);
      trainingInterval = null;
    }
  }

  // --- EVENT LISTENERS ---
  function setupEventListeners() {
    // Preset change
    selectPreset.addEventListener('change', (e) => {
      loadPreset(e.target.value);
      resetSimulation();
    });

    // Weight/parameter input edits
    [inputW1, inputW2, inputBias, inputLr, inputEpochs].forEach(elem => {
      elem.addEventListener('change', () => {
        resetSimulation();
      });
    });

    // Action buttons
    btnStep.addEventListener('click', () => {
      trainStep();
    });

    btnTrain.addEventListener('click', () => {
      if (isTrainingInProgress) {
        stopAutomaticTraining();
      } else {
        if (converged || currentEpoch >= maxEpochs) {
          resetSimulation();
        }
        startAutomaticTraining();
      }
    });

    btnReset.addEventListener('click', () => {
      resetSimulation();
    });

    // Custom Data Table input listeners
    trainingDataTable.addEventListener('input', (e) => {
      const target = e.target;
      if (target.classList.contains('table-input') || target.classList.contains('table-select')) {
        const idx = parseInt(target.getAttribute('data-idx'));
        const param = target.getAttribute('data-param');
        const val = parseFloat(target.value);
        
        if (param === 'x1') {
          trainingData[idx].inputs[0] = val;
        } else if (param === 'x2') {
          trainingData[idx].inputs[1] = val;
        } else if (param === 'target') {
          trainingData[idx].target = parseInt(target.value);
        }
        
        resetSimulation();
      }
    });

    // Custom table delete row click listener
    trainingDataTable.addEventListener('click', (e) => {
      const target = e.target;
      if (target.classList.contains('btn-delete-row')) {
        const idx = parseInt(target.getAttribute('data-idx'));
        trainingData.splice(idx, 1);
        renderTrainingDataTable();
        resetSimulation();
      }
    });

    // Add custom row button
    btnAddRow.addEventListener('click', () => {
      trainingData.push({ inputs: [0.5, 0.5], target: 0 });
      renderTrainingDataTable();
      resetSimulation();
    });

    // Chatbot Event Listeners
    chatbotToggle.addEventListener('click', () => {
      chatbotContainer.className = 'chatbot-expanded';
      chatBadge.style.display = 'none';
      scrollToLatestMessage();
    });

    chatClose.addEventListener('click', () => {
      chatbotContainer.className = 'chatbot-collapsed';
    });

    chatSend.addEventListener('click', () => {
      handleChatInput();
    });

    chatInput.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') {
        handleChatInput();
      }
    });

    // Click on suggestion chips
    chatChipsContainer.addEventListener('click', (e) => {
      const target = e.target;
      if (target.classList.contains('chip-btn')) {
        const question = target.getAttribute('data-question');
        sendUserMessage(question);
        respondToQuestion(question);
      }
    });
  }

  // --- CHATBOT TUTOR LOGIC ---
  function triggerGreetingBadge() {
    setTimeout(() => {
      if (chatbotContainer.className === 'chatbot-collapsed') {
        chatBadge.style.display = 'inline-block';
      }
    }, 4000);
  }

  function scrollToLatestMessage() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function sendUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user-message';
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);
    scrollToLatestMessage();
  }

  function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message bot-message typing-indicator';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = 'Thinking<span>.</span><span>.</span><span>.</span>';
    chatMessages.appendChild(typingDiv);
    scrollToLatestMessage();
  }

  function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
      indicator.remove();
    }
  }

  function handleChatInput() {
    const text = chatInput.value.trim();
    if (text === '') return;
    
    sendUserMessage(text);
    chatInput.value = '';
    
    respondToQuestion(text);
  }

  // Intelligent responder matching keywords/questions
  function respondToQuestion(question) {
    showTypingIndicator();
    
    // Simulate thinking delay (sleek user experience)
    setTimeout(() => {
      removeTypingIndicator();
      
      const lowerQ = question.toLowerCase();
      let reply = '';
      const match = (pattern) => new RegExp(pattern, 'i').test(lowerQ);

      if (match('\\bxor\\b|exclusive|fail|limit')) {
        reply = `
          <strong>The XOR Gate Limitation:</strong> ⚠️<br><br>
          A single-layer perceptron <strong>cannot</strong> learn the XOR logic function. <br><br>
          <strong>Why?</strong> The outputs of XOR are:<br>
          (0,0) → 0, (1,1) → 0<br>
          (0,1) → 1, (1,0) → 1<br><br>
          Plotting these points on a grid shows that you cannot draw a single straight line to separate the 0s from the 1s. This is called a <strong>non-linearly separable</strong> problem. It was mathematically proved by Marvin Minsky and Seymour Papert in 1969, which temporarily halted neural network research. <br><br>
          To solve XOR, we need a <strong>Multi-Layer Perceptron (MLP)</strong> with a hidden layer.
        `;
      } 
      else if (match('\\band\\b')) {
        reply = `
          <strong>AND Logic Gate Learning:</strong><br><br>
          An AND gate outputs <span class="math-formula">1</span> only if both inputs are <span class="math-formula">1</span>.<br>
          <ul class="bullet-list">
            <li>(0,0) → 0</li>
            <li>(0,1) → 0</li>
            <li>(1,0) → 0</li>
            <li>(1,1) → 1</li>
          </ul>
          A single perceptron can easily learn this decision line because the classes (0 and 1) are <strong>linearly separable</strong>. Set weights around <span class="math-formula">0.5, 0.5</span> and bias to <span class="math-formula">-0.8</span> to see it separate the green dot from red dots!
        `;
      } 
      else if (match('\\bor\\b')) {
        reply = `
          <strong>OR Logic Gate Learning:</strong><br><br>
          An OR gate outputs <span class="math-formula">1</span> if at least one input is <span class="math-formula">1</span>.<br>
          <ul class="bullet-list">
            <li>(0,0) → 0</li>
            <li>(0,1) → 1</li>
            <li>(1,0) → 1</li>
            <li>(1,1) → 1</li>
          </ul>
          Like the AND gate, it is <strong>linearly separable</strong>. Set weights to <span class="math-formula">0.5, 0.5</span> and bias to <span class="math-formula">-0.3</span> to represent the OR gate.
        `;
      } 
      else if (match('rule|formula|math|equation')) {
        reply = `
          <strong>Perceptron Learning Rule Formulas:</strong><br><br>
          1. <strong>Net Input Calculation:</strong><br>
          <span class="math-formula">z = w₁x₁ + w₂x₂ + b</span><br><br>
          2. <strong>Activation Output (y):</strong><br>
          <span class="math-formula">y = 1</span> if <span class="math-formula">z ≥ 0</span>, else <span class="math-formula">0</span><br><br>
          3. <strong>Error (e):</strong><br>
          <span class="math-formula">e = target - y</span><br><br>
          4. <strong>Weight Update:</strong><br>
          <span class="math-formula">wᵢ = wᵢ + η × e × xᵢ</span><br>
          <span class="math-formula">b = b + η × e</span><br><br>
          Where <span class="math-formula">η (eta)</span> is the <strong>learning rate</strong>.
        `;
      } 
      else if (match('learning rate|\\blr\\b|\\beta\\b|\\bη\\b')) {
        reply = `
          <strong>What is Learning Rate (η)?</strong><br><br>
          The learning rate controls how big of a step the perceptron takes when updating its weights.<br>
          <ul class="bullet-list">
            <li><strong>Too high (e.g. 0.9):</strong> May overshoot the solution, causing the decision line to bounce back and forth wildly (oscillate).</li>
            <li><strong>Too low (e.g. 0.01):</strong> Learning becomes extremely stable but requires many epochs to reach convergence.</li>
          </ul>
          Try varying η in the Parameters panel and see how it alters the number of steps required to converge!
        `;
      } 
      else if (match('bias|theta|\\bθ\\b')) {
        reply = `
          <strong>Role of the Bias (θ):</strong><br><br>
          The bias translates the decision boundary line. Without a bias, the line <span class="math-formula">w₁x₁ + w₂x₂ = 0</span> would always pass directly through the origin (0,0).<br><br>
          By adding bias <span class="math-formula">b</span> (where <span class="math-formula">w₁x₁ + w₂x₂ + b = 0</span>), we allow the boundary line to shift up, down, left, or right, which is essential to classify datasets that don't cluster symmetric to the origin.
        `;
      } 
      else if (match('epoch')) {
        reply = `
          <strong>What is an Epoch?</strong><br><br>
          An <strong>epoch</strong> represents one complete cycle through the entire training dataset.<br><br>
          During one epoch, the perceptron inspects all data points, checks its predictions, calculates errors, and applies updates to the weights and bias. Training stops when no errors are found (convergence) or the maximum epoch limit is met.
        `;
      } 
      else if (match('perceptron|hello|hi')) {
        reply = `
          <strong>About Perceptrons:</strong><br><br>
          Invented by <strong>Frank Rosenblatt</strong> in 1957, the Perceptron is the simplest type of artificial neural network. It mimics a biological neuron by taking binary inputs, applying weights, summing them up with a bias, and applying a step threshold function to make a binary decision (0 or 1).
        `;
      } 
      else {
        reply = `
          Interesting question! I am specialized in the **Perceptron Learning Algorithm**. <br><br>
          Would you like to know about:
          <ul class="bullet-list">
            <li>How the <strong>AND</strong> or <strong>OR</strong> gates learn?</li>
            <li>Why <strong>XOR</strong> fails?</li>
            <li>The mathematical <strong>formulas</strong>?</li>
            <li>The role of **learning rate** or **bias**?</li>
          </ul>
          Feel free to ask or click one of the quick chips above!
        `;
      }
      
      const botMsgDiv = document.createElement('div');
      botMsgDiv.className = 'message bot-message';
      botMsgDiv.innerHTML = reply;
      chatMessages.appendChild(botMsgDiv);
      scrollToLatestMessage();
      
    }, 600); // 600ms delay
  }

  // --- START THE SIMULATOR ---
  init();
});
