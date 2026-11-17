/* Unikernel.ML report — binds window.BENCH_DATA to the document.
   Everything rendered here comes from a captured benchmark run. */
(function () {
  'use strict';

  var root = document.documentElement;
  root.classList.add('js');

  var data = window.BENCH_DATA;
  var REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var SVG_NS = 'http://www.w3.org/2000/svg';
  var formatInt = new Intl.NumberFormat('en-US');

  var $ = function (selector) { return document.querySelector(selector); };
  var byId = function (id) { return document.getElementById(id); };

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function svg(tag, attrs, text) {
    var node = document.createElementNS(SVG_NS, tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) { node.setAttribute(key, attrs[key]); });
    }
    if (text != null) node.textContent = text;
    return node;
  }

  function us(ns) { return (Number(ns) || 0) / 1000; }
  function fmtUs(ns, digits) { return us(ns).toFixed(digits == null ? 2 : digits); }
  function fixed(value, digits) { return Number(value || 0).toFixed(digits == null ? 2 : digits); }
  function bytes(value) {
    var n = Number(value) || 0;
    if (n >= 1048576) return (n / 1048576).toFixed(2) + ' MiB';
    if (n >= 1024) return (n / 1024).toFixed(1) + ' KiB';
    return n + ' B';
  }
  function pad2(n) { return String(n).length < 2 ? '0' + n : String(n); }

  if (!data || !data.unikernel || !data.unikernel.passes) {
    byId('noData').hidden = false;
    byId('statusText').textContent = 'no data';
    initMotion();
    return;
  }

  var uk = data.unikernel;
  var host = data.host || {};
  var model = data.model || {};
  var history = (model.training && model.training.history) || [];
  var evaluation = model.evaluation || {};
  var scenario = data.scenario || {};

  /* ---------------------------------------------------------------- hero */

  function countUp(node, target, digits) {
    var format = function (value) { return Number(value).toFixed(digits); };
    if (REDUCED || !isFinite(target)) {
      node.textContent = format(target || 0);
      return;
    }
    var start = null;
    var duration = 1050;
    var settled = false;
    function frame(now) {
      if (start === null) start = now;
      var t = Math.min(1, (now - start) / duration);
      var eased = 1 - Math.pow(1 - t, 3);
      node.textContent = format(target * eased);
      if (t < 1) {
        window.requestAnimationFrame(frame);
      } else {
        settled = true;
      }
    }
    window.requestAnimationFrame(frame);
    // Frames are throttled in background tabs; the figure must still be right.
    window.setTimeout(function () {
      if (!settled) node.textContent = format(target);
    }, duration + 250);
  }

  var counters = {
    p50: { value: us(uk.p50_ns), digits: 2 },
    p99: { value: us(uk.p99_ns), digits: 2 },
    footprint: { value: (Number(uk.total_resident_bytes) || 0) / 1048576, digits: 2 },
    accuracy: { value: (Number(uk.accuracy) || 0) * 100, digits: 2 }
  };

  Array.prototype.forEach.call(document.querySelectorAll('[data-counter]'), function (node) {
    var spec = counters[node.getAttribute('data-counter')];
    if (spec) countUp(node, spec.value, spec.digits);
  });

  byId('jitterValue').textContent = uk.jitter_ratio ? fixed(uk.jitter_ratio) + '\u00d7' : '\u2014';
  byId('throughputValue').textContent =
    uk.throughput_per_second ? formatInt.format(uk.throughput_per_second) + '/s' : '\u2014';
  byId('timerValue').textContent =
    uk.cycles_per_second ? fixed(uk.cycles_per_second / 1e9) + ' GHz' : 'uncalibrated';

  byId('scenarioBadge').textContent = scenario.name || 'MNIST forward pass';
  byId('accuracyHint').textContent =
    'of ' + formatInt.format(uk.samples) + ' frozen benchmark samples';

  byId('runStamp').textContent =
    'run ' + (data.generated_at || 'unknown') +
    ' \u00b7 ' + formatInt.format(uk.passes) + ' timed passes' +
    ' \u00b7 ' + formatInt.format(uk.samples) + ' scored images' +
    ' \u00b7 resident image ' + bytes(uk.kernel_bytes);

  byId('bootNote').textContent =
    'Emulated boot: ' + fixed((Number(uk.boot_to_inference_ns) || 0) / 1e9) +
    ' s from kernel entry through the first forward pass. QEMU translates guest ' +
    'code on first execution, so this figure describes the emulator rather than ' +
    'silicon; the per-pass latencies below are measured after that warm-up.';

  byId('statusText').textContent = 'qemu boot captured';
  byId('footerStamp').textContent = 'run ' + (data.generated_at || 'unknown');
  var generatedAt = Date.parse(data.generated_at || '');
  if (isFinite(generatedAt)) {
    var ageHours = Math.max(0, (Date.now() - generatedAt) / 3600000);
    if (ageHours > 72) {
      byId('statusText').textContent = 'measurement is stale';
      document.querySelector('.runstatus').classList.add('runstatus--stale');
    }
  }

  function downloadReport() {
    var blob = new Blob([JSON.stringify(data, null, 2) + '\n'], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var link = document.createElement('a');
    link.href = url;
    link.download = 'benchmark-' + (data.generated_at || 'report').replace(/[^a-z0-9]+/gi, '-') + '.json';
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 0);
  }

  function copySummary() {
    var summary = [
      'Unikernel benchmark',
      'Run: ' + (data.generated_at || 'unknown'),
      'Accuracy: ' + fixed((Number(uk.accuracy) || 0) * 100) + '% (' +
        formatInt.format(uk.correct || 0) + '/' + formatInt.format(uk.samples || 0) + ')',
      'Latency: p50 ' + fmtUs(uk.p50_ns) + ' us, p99 ' + fmtUs(uk.p99_ns) + ' us',
      'Selected kernel: ' + ((data.dispatch || {}).unikernel || {}).selected || uk.kernel || 'unknown'
    ].join('\n');
    var button = byId('copySummary');
    function copied() {
      button.textContent = 'Copied';
      window.setTimeout(function () { button.textContent = 'Copy summary'; }, 1600);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(summary).then(copied).catch(function () {
        button.textContent = 'Copy unavailable';
        window.setTimeout(function () { button.textContent = 'Copy summary'; }, 1600);
      });
    } else {
      var area = document.createElement('textarea');
      area.value = summary;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      var copiedByFallback = document.execCommand('copy');
      area.remove();
      if (copiedByFallback) copied();
    }
  }

  byId('downloadReport').addEventListener('click', downloadReport);
  byId('copySummary').addEventListener('click', copySummary);

  /* ------------------------------------------------------------ pipeline */

  function renderStages() {
    var lines = (data.logs && data.logs.unikernel) || [];
    var stagePattern = /\[(\d+)\/(\d+)\]\s+(.*?)\s*\.{2,}\s*(\S+)\s*$/;
    var stages = [];
    var current = null;

    lines.forEach(function (line) {
      var match = line.match(stagePattern);
      if (match) {
        current = { index: match[1], total: match[2], title: match[3], status: match[4], details: [] };
        stages.push(current);
        return;
      }
      if (current && /^\s{8,}\S/.test(line) && !/@@|^=+|^-+\s*$/.test(line.trim())) {
        var detail = line.trim();
        if (detail.length && current.details.length < 2) current.details.push(detail);
      }
    });

    var list = byId('stageList');
    if (!stages.length) {
      list.replaceChildren(el('li', 'stage', 'The boot transcript did not contain stage lines.'));
      return;
    }

    list.replaceChildren.apply(list, stages.map(function (stage, i) {
      var item = el('li', 'stage');
      item.style.setProperty('--delay', (i * 55) + 'ms');
      item.appendChild(el('span', 'stage__index', stage.index));

      var body = el('div');
      body.appendChild(el('h3', null, stage.title));
      stage.details.forEach(function (detail) {
        body.appendChild(el('p', 'stage__detail', detail));
      });
      var ok = /ready|ok|verified|loaded/i.test(stage.status);
      var status = el('span', 'stage__status' + (ok ? '' : ' stage__status--warn'),
        ok ? 'ready' : stage.status.toLowerCase());
      body.appendChild(status);
      item.appendChild(body);
      return item;
    }));
  }

  /* ------------------------------------------------------------ histogram */

  var HIST_W = 900;
  var HIST_H = 300;

  function renderHistogram() {
    var buckets = (data.histogram && data.histogram.unikernel) || [];
    var container = byId('histogram');
    if (!buckets.length) {
      byId('histNote').textContent = 'No histogram was emitted by this run.';
      return;
    }

    var pad = { top: 18, right: 12, bottom: 34, left: 48 };
    var innerW = HIST_W - pad.left - pad.right;
    var innerH = HIST_H - pad.top - pad.bottom;
    var maxCount = Math.max.apply(null, buckets.map(function (b) { return b.count; })) || 1;
    var step = innerW / buckets.length;
    var barW = Math.max(6, step * 0.62);

    var chart = svg('svg', { viewBox: '0 0 ' + HIST_W + ' ' + HIST_H, role: 'presentation' });
    var defs = svg('defs');
    var gradient = svg('linearGradient', { id: 'barGrad', x1: '0', y1: '0', x2: '0', y2: '1' });
    gradient.appendChild(svg('stop', { offset: '0%', 'stop-color': '#4fd1c5', 'stop-opacity': '0.95' }));
    gradient.appendChild(svg('stop', { offset: '100%', 'stop-color': '#4fd1c5', 'stop-opacity': '0.28' }));
    defs.appendChild(gradient);
    chart.appendChild(defs);

    for (var g = 0; g <= 4; g += 1) {
      var y = pad.top + innerH * (g / 4);
      chart.appendChild(svg('line', {
        x1: pad.left, x2: HIST_W - pad.right, y1: y, y2: y, class: 'grid-line'
      }));
      chart.appendChild(svg('text', {
        x: pad.left - 10, y: y + 3.5, 'text-anchor': 'end', class: 'axis-text'
      }, formatInt.format(Math.round(maxCount * (1 - g / 4)))));
    }

    buckets.forEach(function (bucket, index) {
      var height = Math.max(2, (bucket.count / maxCount) * innerH);
      var x = pad.left + index * step + (step - barW) / 2;
      var rect = svg('rect', {
        x: x,
        y: pad.top + innerH - height,
        width: barW,
        height: height,
        rx: 4,
        class: 'hist-bar',
        style: '--d:' + (index * 26) + 'ms'
      });
      rect.appendChild(svg('title', {},
        fmtUs(bucket.from_ns) + '\u2013' + fmtUs(bucket.to_ns) + ' \u00b5s \u00b7 ' +
        formatInt.format(bucket.count) + ' passes'));
      chart.appendChild(rect);

      if (index % 3 === 0) {
        chart.appendChild(svg('text', {
          x: pad.left + index * step + step / 2,
          y: HIST_H - 12,
          'text-anchor': 'middle',
          class: 'axis-text'
        }, fmtUs(bucket.from_ns, 0)));
      }
    });

    container.replaceChildren(chart);

    byId('histNote').textContent =
      formatInt.format(uk.passes) + ' passes \u00b7 mean ' + fmtUs(uk.mean_ns) +
      ' \u00b5s \u00b7 min ' + fmtUs(uk.min_ns) + ' \u00b5s \u00b7 max ' + fmtUs(uk.max_ns) + ' \u00b5s';

    byId('histAxis').replaceChildren(
      el('span', null, fmtUs(uk.hist_min_ns, 0) + ' \u00b5s'),
      el('span', null, buckets.length + ' buckets'),
      el('span', null, fmtUs(uk.hist_max_ns, 0) + ' \u00b5s')
    );
  }

  /* ----------------------------------------------------------- percentiles */

  function renderPercentiles(container, variant, isHost) {
    var rows = [
      ['p50', variant.p50_ns],
      ['p90', variant.p90_ns],
      ['p99', variant.p99_ns],
      ['max', variant.max_ns]
    ];
    var max = Math.max.apply(null, rows.map(function (row) { return Number(row[1]) || 0; })) || 1;

    container.replaceChildren.apply(container, rows.map(function (row, index) {
      var wrapper = el('div', 'pctrow');
      wrapper.appendChild(el('span', 'pctrow__label', row[0]));

      var track = el('div', 'pctrow__track');
      var fill = el('span', 'pctrow__fill');
      fill.style.setProperty('--d', (index * 90) + 'ms');
      track.appendChild(fill);
      wrapper.appendChild(track);

      var pct = Math.max(2, (Number(row[1]) / max) * 100);
      window.requestAnimationFrame(function () {
        window.requestAnimationFrame(function () { fill.style.width = pct + '%'; });
      });

      wrapper.appendChild(el('span', 'pctrow__value', fmtUs(row[1]) + ' \u00b5s'));
      return wrapper;
    }));

    if (isHost) {
      byId('hostBadge').textContent = host.passes
        ? formatInt.format(host.passes) + ' passes'
        : 'native run';
    }
  }

  /* -------------------------------------------------------------- dispatch */

  function renderDispatchTable(table, record, cyclesPerSecond) {
    var body = table.querySelector('tbody');
    var order = ['scalar', 'avx2', 'fma'];
    var measured = record && record.measured || {};
    var rows = [];

    order.forEach(function (name) {
      var entry = measured[name];
      var selected = record && record.selected === name;
      var eligible = record && record.eligible_mask;
      var bit = order.indexOf(name);

      if (!entry && !(eligible & (1 << bit))) {
        return;
      }
      var tr = el('tr');
      var kernelCell = el('td', null, name + (selected ? ' \u2713' : ''));
      if (selected) {
        kernelCell.style.color = '#4fd1c5';
        kernelCell.style.fontWeight = '600';
      }
      tr.appendChild(kernelCell);

      if (entry) {
        var cycles = entry.cycles;
        var micros = cyclesPerSecond
          ? (cycles / (cyclesPerSecond / 1000) / 1000).toFixed(2) + ' \u00b5s'
          : 'n/a';
        tr.appendChild(el('td', null, formatInt.format(cycles) + ' (' + micros + ')'));
        var gate = record.banned_mask & (1 << bit)
          ? 'FAILED'
          : (entry.diff != null ? 'diff ' + entry.diff + 'e-6' : 'reference');
        tr.appendChild(el('td', null, gate));
      } else {
        tr.appendChild(el('td', 'empty', 'not measured'));
        tr.appendChild(el('td', 'empty', '\u2014'));
      }
      rows.push(tr);
    });

    if (!rows.length) {
      var emptyRow = el('tr');
      var emptyCell = el('td', 'empty', 'No dispatch record in this run.');
      emptyCell.colSpan = 3;
      emptyRow.appendChild(emptyCell);
      body.replaceChildren(emptyRow);
      return;
    }
    body.replaceChildren.apply(body, rows);
  }

  function renderDispatch() {
    var dispatch = data.dispatch || {};
    var ukRecord = dispatch.unikernel || {};
    var hostRecord = dispatch.host || {};

    renderDispatchTable(byId('dispatchUkTable'), ukRecord,
      uk.cycles_per_second || 0);
    renderDispatchTable(byId('dispatchHostTable'), hostRecord, 0);

    var ukCaps = (ukRecord.caps || {});
    var hostCaps = (hostRecord.caps || {});
    byId('dispatchUkBadge').textContent = ukRecord.selected
      ? 'selected: ' + ukRecord.selected
      : 'measured at boot';
    byId('dispatchHostBadge').textContent = hostRecord.selected
      ? 'selected: ' + hostRecord.selected
      : 'host policy';

    byId('dispatchUkNote').textContent =
      'Guest CPU: ' + (ukCaps.brand || 'unknown') + ' \u00b7 avx2=' +
      (ukCaps.avx2 ? 'on' : 'off') + ' \u00b7 fma=' +
      (ukCaps.fma ? 'on' : 'off') + ' \u00b7 avx state ' +
      (ukCaps.avx_state ? 'enabled by the kernel' : 'unavailable') + '. ' +
      (ukRecord.selected === 'scalar' && ukCaps.avx2
        ? 'SIMD is advertised but measured SLOWER under this emulator, so the policy correctly stays scalar.'
        : 'The selected kernel was the fastest measured candidate that passed the gate.');

    byId('dispatchHostNote').textContent =
      'Host CPU: ' + (hostCaps.brand || 'unknown') + ' \u00b7 avx2=' +
      (hostCaps.avx2 ? 'on' : 'off') + ' \u00b7 fma=' +
      (hostCaps.fma ? 'on' : 'off') + '. The identical dispatch policy runs natively; ' +
      'cycles are only comparable within one environment, so the table shows the host\u2019s own ranking.';

    byId('dispatchNote').textContent = dispatch.note || '';
  }

  /* -------------------------------------------------------------- accuracy */

  function renderAccuracy() {
    var accuracy = Number(uk.accuracy) || 0;
    var circumference = 2 * Math.PI * 66;
    var ring = byId('ringValue');
    var svgRoot = ring.ownerSVGElement;

    var defs = svg('defs');
    var gradient = svg('linearGradient', {
      id: 'ringGrad', x1: '0', y1: '0', x2: '1', y2: '1'
    });
    gradient.appendChild(svg('stop', { offset: '0%', 'stop-color': '#4fd1c5' }));
    gradient.appendChild(svg('stop', { offset: '100%', 'stop-color': '#7aa2ff' }));
    defs.appendChild(gradient);
    svgRoot.insertBefore(defs, svgRoot.firstChild);

    ring.style.strokeDasharray = String(circumference);
    ring.style.strokeDashoffset = String(circumference);
    window.requestAnimationFrame(function () {
      window.requestAnimationFrame(function () {
        ring.style.strokeDashoffset = String(circumference * (1 - accuracy));
      });
    });

    byId('ringNumber').textContent = fixed(accuracy * 100) + '%';
    byId('ringCaption').textContent =
      formatInt.format(uk.correct) + ' / ' + formatInt.format(uk.samples) + ' correct';

    var architecture = model.architecture || {};
    var facts = [
      ['Full test set', evaluation.test_accuracy != null
        ? fixed(evaluation.test_accuracy * 100) + '%'
        : '\u2014'],
      ['Parameters', architecture.parameters ? formatInt.format(architecture.parameters) : '\u2014'],
      ['Topology', architecture.hidden
        ? architecture.input + ' \u2192 ' + architecture.hidden + ' \u2192 ' + architecture.output
        : '\u2014'],
      ['Boot artifacts', bytes(uk.total_resident_bytes)],
      ['Both environments agree', data.comparison && data.comparison.accuracy_match ? 'yes' : 'check']
    ];

    byId('accuracyFacts').replaceChildren.apply(byId('accuracyFacts'), facts.map(function (fact) {
      var row = el('div');
      row.appendChild(el('dt', null, fact[0]));
      row.appendChild(el('dd', null, String(fact[1])));
      return row;
    }));
  }

  /* ---------------------------------------------------------------- matrix */

  function renderMatrix() {
    var matrix = evaluation.confusion_matrix;
    var container = byId('confusionMatrix');
    byId('matrixBadge').textContent =
      evaluation.test_accuracy != null ? 'Full-set evaluation' : 'No full-set evaluation';
    if (!matrix || !matrix.length) {
      container.replaceChildren(el('p', 'card__note', 'No confusion matrix in this run.'));
      return;
    }

    var nodes = [];
    nodes.push(el('span', 'matrix__label', ''));

    for (var c = 0; c < 10; c += 1) {
      nodes.push(el('span', 'matrix__label', String(c)));
    }

    matrix.forEach(function (row, rowIndex) {
      nodes.push(el('span', 'matrix__label', String(rowIndex)));
      var rowMax = Math.max.apply(null, row.map(Number)) || 1;
      row.forEach(function (value, colIndex) {
        var ratio = Number(value) / rowMax;
        var alpha = 0.05 + 0.9 * ratio;
        var cell = el('span', 'matrix__cell' + (alpha > 0.5 ? ' matrix__cell--hot' : ''), formatInt.format(value));
        cell.style.setProperty('--a', alpha.toFixed(3));
        cell.style.setProperty('--d', ((rowIndex * 10 + colIndex) * 3) + 'ms');
        cell.title = formatInt.format(value) + ' images: true ' + rowIndex +
          ' predicted as ' + colIndex;
        nodes.push(cell);
      });
    });

    container.replaceChildren.apply(container, nodes);
  }

  /* ---------------------------------------------------------------- digits */

  function renderDigits() {
    var previews = model.previews || [];
    var container = byId('digitGrid');
    if (!previews.length) {
      container.replaceChildren(el('p', 'card__note', 'No previews in this run.'));
      return;
    }

    container.replaceChildren.apply(container, previews.map(function (preview, index) {
      var card = el('div', 'digit');
      var canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      var context = canvas.getContext('2d');
      var image = context.createImageData(28, 28);
      for (var i = 0; i < 784; i += 1) {
        var value = preview.pixels[i] || 0;
        image.data[i * 4] = value;
        image.data[i * 4 + 1] = value;
        image.data[i * 4 + 2] = value;
        image.data[i * 4 + 3] = 255;
      }
      context.putImageData(image, 0, 0);
      canvas.setAttribute('aria-label', 'MNIST digit ' + preview.label);

      var caption = el('div', 'digit__caption');
      caption.appendChild(el('span', null, 'true ' + preview.label + ' \u00b7 pred ' + preview.prediction));
      var correct = preview.label === preview.prediction;
      caption.appendChild(el('span', 'digit__verdict' + (correct ? '' : ' digit__verdict--wrong'),
        correct ? 'correct' : 'misclassified'));

      card.appendChild(canvas);
      card.appendChild(caption);
      card.style.setProperty('--delay', (index * 70) + 'ms');
      return card;
    }));
  }

  /* ----------------------------------------------------------- line charts */

  function renderLineChart(container, points, options) {
    if (!points.length) {
      container.replaceChildren(el('p', 'card__note', 'No training history in this run.'));
      return;
    }

    var width = 540;
    var height = 210;
    var pad = { top: 16, right: 18, bottom: 30, left: 46 };
    var innerW = width - pad.left - pad.right;
    var innerH = height - pad.top - pad.bottom;

    var values = points.map(function (p) { return p.value; });
    var minValue = Math.min.apply(null, values);
    var maxValue = Math.max.apply(null, values);
    var span = maxValue - minValue;
    if (span === 0) span = Math.max(Math.abs(maxValue), 1) * 0.2;
    var lower = minValue - span * 0.25;
    var upper = maxValue + span * 0.25;

    var xAt = function (i) {
      return points.length === 1 ? pad.left + innerW / 2 : pad.left + (i / (points.length - 1)) * innerW;
    };
    var yAt = function (v) {
      return pad.top + innerH - ((v - lower) / (upper - lower)) * innerH;
    };

    var chart = svg('svg', { viewBox: '0 0 ' + width + ' ' + height });
    var gradientId = 'area-' + options.key;
    var defs = svg('defs');
    var gradient = svg('linearGradient', { id: gradientId, x1: '0', y1: '0', x2: '0', y2: '1' });
    gradient.appendChild(svg('stop', { offset: '0%', 'stop-color': options.color, 'stop-opacity': '0.4' }));
    gradient.appendChild(svg('stop', { offset: '100%', 'stop-color': options.color, 'stop-opacity': '0' }));
    defs.appendChild(gradient);
    chart.appendChild(defs);

    [0, 0.5, 1].forEach(function (fraction) {
      var value = upper - (upper - lower) * fraction;
      var y = pad.top + innerH * fraction;
      chart.appendChild(svg('line', {
        x1: pad.left, x2: width - pad.right, y1: y, y2: y, class: 'grid-line'
      }));
      chart.appendChild(svg('text', {
        x: pad.left - 9, y: y + 3.5, 'text-anchor': 'end', class: 'axis-text'
      }, options.format(value)));
    });

    var line = points.map(function (p, i) {
      return (i === 0 ? 'M' : 'L') + xAt(i).toFixed(2) + ' ' + yAt(p.value).toFixed(2);
    }).join(' ');

    var area = line + ' L ' + xAt(points.length - 1).toFixed(2) + ' ' + (pad.top + innerH) +
      ' L ' + xAt(0).toFixed(2) + ' ' + (pad.top + innerH) + ' Z';

    var areaPath = svg('path', { d: area, class: 'area-path' });
    areaPath.style.fill = 'url(#' + gradientId + ')';
    chart.appendChild(areaPath);

    chart.appendChild(svg('path', {
      d: line,
      class: 'line-path' + (options.key === 'loss' ? ' line-path--loss' : ''),
      pathLength: '1'
    }));

    points.forEach(function (point, i) {
      var dot = svg('circle', {
        cx: xAt(i), cy: yAt(point.value), r: 4,
        class: 'dot' + (options.key === 'loss' ? ' dot--loss' : '')
      });
      dot.appendChild(svg('title', {},
        options.label(point) + ': ' + options.format(point.value)));
      chart.appendChild(dot);
    });

    points.forEach(function (point, i) {
      chart.appendChild(svg('text', {
        x: xAt(i), y: height - 10, 'text-anchor': 'middle', class: 'axis-text'
      }, String(point.epoch)));
    });

    container.replaceChildren(chart);
  }

  function renderTraining() {
    var lossPoints = history.map(function (entry, index) {
      return {
        epoch: entry.epoch != null ? entry.epoch : index + 1,
        value: entry.loss != null ? entry.loss : (entry.train_loss != null ? entry.train_loss : 0)
      };
    });
    var accuracyPoints = history.map(function (entry, index) {
      return {
        epoch: entry.epoch != null ? entry.epoch : index + 1,
        value: (entry.test_accuracy != null ? entry.test_accuracy : (entry.accuracy || 0)) * 100
      };
    });

    renderLineChart(byId('lossChart'), lossPoints, {
      key: 'loss',
      color: '#7aa2ff',
      format: function (value) { return value.toFixed(3); },
      label: function (point) { return 'epoch ' + point.epoch; }
    });

    renderLineChart(byId('accuracyChart'), accuracyPoints, {
      key: 'accuracy',
      color: '#4fd1c5',
      format: function (value) { return value.toFixed(1) + '%'; },
      label: function (point) { return 'epoch ' + point.epoch; }
    });

    var training = model.training || {};
    var facts = [
      ['Dataset', model.dataset || '\u2014'],
      ['Optimizer', training.optimizer || '\u2014'],
      ['Learning rate', training.learning_rate != null ? String(training.learning_rate) : '\u2014'],
      ['Batch size', training.batch_size != null ? String(training.batch_size) : '\u2014'],
      ['Epochs', training.epochs != null ? String(training.epochs) : '\u2014'],
      ['Seed', training.seed != null ? String(training.seed) : '\u2014'],
      ['Training time', training.seconds != null ? fixed(training.seconds, 1) + ' s' : '\u2014'],
      ['Weight range', model.weights
        ? fixed(model.weights.min, 3) + ' \u2026 ' + fixed(model.weights.max, 3)
        : '\u2014']
    ];

    byId('trainingFacts').replaceChildren.apply(byId('trainingFacts'), facts.map(function (fact) {
      var row = el('div');
      row.appendChild(el('dt', null, fact[0]));
      row.appendChild(el('dd', null, String(fact[1])));
      return row;
    }));
  }

  /* ------------------------------------------------------------- artifacts */

  function renderArtifacts() {
    var body = byId('artifactTable').querySelector('tbody');
    var items = (data.artifacts || []).filter(function (item) { return item.present; });

    if (!items.length) {
      var emptyRow = el('tr');
      var emptyCell = el('td', 'empty', 'No artifacts recorded.');
      emptyCell.colSpan = 3;
      emptyRow.appendChild(emptyCell);
      body.replaceChildren(emptyRow);
      return;
    }

    body.replaceChildren.apply(body, items.map(function (item) {
      var row = el('tr');
      row.appendChild(el('td', null, item.label));
      row.appendChild(el('td', null, bytes(item.bytes)));

      var hashCell = el('td', 'hash', (item.sha256_short || '') + '\u2026');
      hashCell.title = item.sha256 || '';
      hashCell.style.cursor = 'help';
      row.appendChild(hashCell);
      return row;
    }));
  }

  /* --------------------------------------------------------------- terminal */

  function escapeHtml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function renderLog() {
    var lines = (data.logs && data.logs.unikernel) || [];
    var pre = byId('bootLog');

    if (!lines.length) {
      pre.textContent = 'No serial transcript captured.';
      return;
    }

    pre.innerHTML = lines.map(function (line) {
      var safe = escapeHtml(line);
      safe = safe.replace(/\bREADY\b/g, '<span class="term-ok">READY</span>');
      safe = safe.replace(/\b(Verified|verified)\b/g, '<span class="term-ok">$1</span>');
      safe = safe.replace(/(^|\s)([\w_]+)=/g, '$1<span class="term-key">$2</span>=');
      safe = safe.replace(/=(\d+(?:\.\d+)?)/g, '=<span class="term-val">$1</span>');
      return safe;
    }).join('\n');
  }

  /* ----------------------------------------------------------------- method */

  function renderMethodNotes() {
    var cpu = (data.host_environment && data.host_environment.cpu_model) || 'the host CPU';
    byId('hostNote').textContent =
      'The baseline runs the identical engine as an ordinary process (' +
      formatInt.format(host.passes || 0) + ' timed passes on ' + cpu +
      ', kernel: ' + (host.kernel || 'unknown') +
      '). Not like-for-like: the unikernel numbers above come from a CPU ' +
      'emulated by QEMU.';

    var timer = scenario.timer || 'RDTSC';
    byId('methodNote').textContent =
      timer + ' \u00b7 ' + fixed((uk.cycles_per_second || 0) / 1e9) + ' GHz \u00b7 ' +
      formatInt.format(uk.passes) + ' passes per environment \u00b7 ' +
      formatInt.format(uk.samples) + ' images per environment. Identical model and sample ' +
      'files were used in both environments; the execution kernel is chosen by the ' +
      'measured dispatch policy in each.';
  }

  /* ------------------------------------------------------------------ motion */

  function initMotion() {
    var revealables = document.querySelectorAll('.reveal');

    if (REDUCED || !('IntersectionObserver' in window)) {
      Array.prototype.forEach.call(revealables, function (node) { node.classList.add('is-visible'); });
    } else {
      var revealObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          entry.target.classList.add('is-visible');
          revealObserver.unobserve(entry.target);
        });
      }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });

      Array.prototype.forEach.call(revealables, function (node, index) {
        if (!node.style.getPropertyValue('--delay')) {
          node.style.setProperty('--delay', Math.min(index * 40, 240) + 'ms');
        }
        revealObserver.observe(node);
      });
    }

    var bar = byId('scrollBar');
    var ticking = false;

    function updateScroll() {
      var scrollable = document.documentElement.scrollHeight - window.innerHeight;
      var progress = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
      bar.style.width = Math.min(100, Math.max(0, progress)) + '%';
      ticking = false;
    }

    window.addEventListener('scroll', function () {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(updateScroll);
    }, { passive: true });
    updateScroll();

    var links = Array.prototype.slice.call(document.querySelectorAll('.nav a'));
    var sections = links.map(function (link) {
      return document.querySelector(link.getAttribute('href'));
    }).filter(Boolean);

    if ('IntersectionObserver' in window && sections.length) {
      var sectionObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          links.forEach(function (link) {
            link.classList.toggle('is-active',
              link.getAttribute('href') === '#' + entry.target.id);
          });
        });
      }, { rootMargin: '-45% 0px -50% 0px' });

      sections.forEach(function (section) { sectionObserver.observe(section); });
    }

    var toggle = byId('logToggle');
    var log = byId('bootLog');
    if (toggle && log) {
      toggle.addEventListener('click', function () {
        var collapsed = log.classList.toggle('is-collapsed');
        toggle.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
        toggle.textContent = collapsed ? 'expand' : 'collapse';
      });
    }

    var navToggle = byId('navToggle');
    var nav = byId('reportNav');
    if (navToggle && nav) {
      navToggle.addEventListener('click', function () {
        var open = nav.classList.toggle('is-open');
        navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
      nav.addEventListener('click', function (event) {
        if (event.target.tagName === 'A') {
          nav.classList.remove('is-open');
          navToggle.setAttribute('aria-expanded', 'false');
        }
      });
    }
  }

  /* -------------------------------------------------------------------- run */

  renderStages();
  renderHistogram();
  renderPercentiles(byId('percentileList'), uk, false);
  if (host.passes) renderPercentiles(byId('hostPercentiles'), host, true);
  renderDispatch();
  renderAccuracy();
  renderMatrix();
  renderDigits();
  renderTraining();
  renderArtifacts();
  renderLog();
  renderMethodNotes();
  initMotion();
})();
