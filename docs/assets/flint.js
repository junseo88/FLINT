// FLINT project page: trajectory scrubber, before/after comparison, BibTeX copy. No dependencies.
(function () {
  'use strict';

  // ---- trajectory scrubber -------------------------------------------------
  // <div class="traj" data-example="ffhq_sr8_00198" data-base="../assets/traj/"> ... </div>
  function initTraj(root) {
    const ex = root.dataset.example;
    const base = root.dataset.base || 'assets/traj/';
    const data = (window.FLINT_TRAJ || {})[ex];
    if (!data) return;
    const pad = (i) => String(i).padStart(2, '0');
    const imgR = root.querySelector('[data-role="random"]');
    const imgF = root.querySelector('[data-role="flint"]');
    const range = root.querySelector('input[type="range"]');
    const stepOut = root.querySelector('[data-role="step"]');
    const tOut = root.querySelector('[data-role="t"]');
    const pR = root.querySelector('[data-role="psnr-random"]');
    const pF = root.querySelector('[data-role="psnr-flint"]');
    range.max = data.steps - 1;
    // preload frames so scrubbing is instant
    for (let i = 0; i < data.steps; i++) {
      new Image().src = `${base}${ex}/random_${pad(i)}.jpg`;
      new Image().src = `${base}${ex}/flint_${pad(i)}.jpg`;
    }
    function show(i) {
      imgR.src = `${base}${ex}/random_${pad(i)}.jpg`;
      imgF.src = `${base}${ex}/flint_${pad(i)}.jpg`;
      const last = i === data.steps - 1;
      if (stepOut) stepOut.textContent = last ? `${i} (last)` : String(i);
      if (tOut) tOut.textContent = data.sigma[i].toFixed(2);
      if (pR) pR.textContent = `${data.psnr_random[i].toFixed(1)} dB`;
      if (pF) pF.textContent = `${data.psnr_flint[i].toFixed(1)} dB`;
      range.setAttribute('aria-valuetext', `step ${i}, random start ${data.psnr_random[i].toFixed(1)} dB, FLINT start ${data.psnr_flint[i].toFixed(1)} dB`);
    }
    range.addEventListener('input', () => show(+range.value));
    show(+range.value || 0);
    const play = root.querySelector('[data-role="play"]');
    if (play) {
      let timer = null;
      play.addEventListener('click', () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = 'Play'; return; }
        if (+range.value >= data.steps - 1) range.value = 0;
        play.textContent = 'Pause';
        timer = setInterval(() => {
          const v = +range.value + 1;
          range.value = v; show(v);
          if (v >= data.steps - 1) { clearInterval(timer); timer = null; play.textContent = 'Play'; }
        }, 180);
      });
    }
  }

  // ---- before / after comparison --------------------------------------------
  // <div class="compare"><img class="compare-under" ...><div class="compare-top"><img ...></div>
  //  <input type="range" class="compare-range" min="0" max="100" value="50" aria-label="..."></div>
  function initCompare(root) {
    const top = root.querySelector('.compare-top');
    const range = root.querySelector('.compare-range');
    const handle = root.querySelector('.compare-handle');
    function set(v) {
      top.style.clipPath = `inset(0 ${100 - v}% 0 0)`;
      if (handle) handle.style.left = `${v}%`;
    }
    range.addEventListener('input', () => set(+range.value));
    function fromPointer(e) {
      const r = root.getBoundingClientRect();
      const v = Math.max(0, Math.min(100, ((e.clientX - r.left) / r.width) * 100));
      range.value = v; set(v);
    }
    let dragging = false;
    root.addEventListener('pointerdown', (e) => { dragging = true; root.setPointerCapture(e.pointerId); fromPointer(e); });
    root.addEventListener('pointermove', (e) => { if (dragging) fromPointer(e); });
    root.addEventListener('pointerup', () => { dragging = false; });
    set(+range.value);
  }

  // ---- results gallery --------------------------------------------------------
  // <div class="gallery" data-base="../assets/results/"></div>; examples from window.FLINT_RESULTS, in display order.
  // Paged cards (three per page on wide screens): each card is a measurement / FLINT slider with the zoom window
  // [x, y, size] (fractions of the image) boxed, and below it the zoomed region of the measurement, FLINT and the
  // ground truth.
  const GAL_CSS = `.gal-top { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 0 0 12px; }
    .gal-count { font-size: 14px; opacity: .75; font-variant-numeric: tabular-nums; }
    .gal-navs { display: flex; gap: 8px; }
    .gal-nav { font: inherit; font-size: 22px; line-height: 1; width: 42px; height: 42px; border-radius: 999px; cursor: pointer;
      border: 1px solid rgba(0,0,0,.18); background: #fff; color: inherit; display: grid; place-items: center; padding: 0 0 3px; }
    .gal-nav:hover { background: rgba(0,0,0,.05); }
    .gal-nav:focus-visible, .gal-dots button:focus-visible { outline: 3px solid #2a78d6; outline-offset: 2px; }
    .gal-grid { display: grid; grid-template-columns: repeat(var(--per, 3), minmax(0, 1fr)); gap: 18px; }
    .gal-card { margin: 0; min-width: 0; }
    .gal-box { position: absolute; border: 2px solid #ffd60a; box-shadow: 0 0 0 1px rgba(0,0,0,.45); pointer-events: none; }
    .gal-strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 6px; }
    .gal-strip figure { margin: 0; position: relative; }
    .gal-strip img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; border: 2px solid #ffd60a; }
    .gal-strip figcaption { position: absolute; left: 6px; bottom: 6px; margin: 0; font-size: 12px; line-height: 1.4;
      color: #fff; background: rgba(0,0,0,.6); padding: 0 6px; border-radius: 4px; pointer-events: none; }
    .gal-strip figure.flint figcaption { font-weight: 700; }
    .gal-dots { display: flex; gap: 8px; justify-content: center; margin-top: 16px; }
    .gal-dots button { width: 10px; height: 10px; border-radius: 50%; border: 0; padding: 0; cursor: pointer; background: rgba(0,0,0,.22); }
    .gal-dots button[aria-current="true"] { background: currentColor; }`;
  function initGallery(root) {
    const items = window.FLINT_RESULTS || [];
    if (!items.length) return;
    if (!document.getElementById('gal-css')) {
      const st = document.createElement('style'); st.id = 'gal-css'; st.textContent = GAL_CSS; document.head.appendChild(st);
    }
    const base = root.dataset.base || 'assets/results/';
    const perPage = () => { const w = root.clientWidth; return w >= 860 ? 3 : (w >= 540 ? 2 : 1); };
    let per = perPage(), first = 0;
    root.innerHTML = `<div class="gal-top"><span class="gal-count" aria-live="polite"></span><div class="gal-navs">
        <button type="button" class="gal-nav" data-step="-1" aria-label="Previous examples">&#8249;</button>
        <button type="button" class="gal-nav" data-step="1" aria-label="Next examples">&#8250;</button></div></div>
      <div class="gal-grid"></div><div class="gal-dots"></div>`;
    const grid = root.querySelector('.gal-grid'), dots = root.querySelector('.gal-dots');
    function card(r) {
      const dir = `${base}${r.id}/`, what = `${r.task.toLowerCase()}, ${r.dataset} image ${r.name}`;
      const [x, y, w] = r.zoom;
      const box = `left:${100 * x}%;top:${100 * y}%;width:${100 * w}%;height:${100 * w}%`;
      const fig = document.createElement('figure');
      fig.className = 'gal-card';
      fig.title = `${r.dataset} image ${r.name}, ${r.task.toLowerCase()}: PSNR ${r.psnr.toFixed(2)} dB, LPIPS ${r.lpips.toFixed(3)}`;
      fig.innerHTML = `<div class="compare"><img src="${dir}recon.jpg" alt="FLINT reconstruction, ${what}">
          <div class="compare-top"><img src="${dir}meas.jpg" alt="Measurement, ${what}"></div><span class="gal-box" style="${box}"></span>
          <input class="compare-range" type="range" min="0" max="100" value="50" aria-label="Measurement versus FLINT, ${what}">
          <span class="compare-handle"></span><span class="lab l">${r.kind}</span><span class="lab r">FLINT</span></div>
        <div class="gal-strip">
          <figure><img src="${dir}zoom_meas.jpg" alt="Zoom on the boxed region of the measurement, ${what}" loading="lazy"><figcaption>input</figcaption></figure>
          <figure class="flint"><img src="${dir}zoom_recon.jpg" alt="Zoom on the boxed region of the FLINT reconstruction, ${what}" loading="lazy"><figcaption>FLINT</figcaption></figure>
          <figure><img src="${dir}zoom_gt.jpg" alt="Zoom on the boxed region of the ground truth, ${what}" loading="lazy"><figcaption>ground truth</figcaption></figure>
        </div>`;
      initCompare(fig.querySelector('.compare'));
      return fig;
    }
    function render() {
      const pages = Math.ceil(items.length / per);
      first = Math.min(first - (first % per), (pages - 1) * per);
      grid.style.setProperty('--per', per);
      grid.replaceChildren(...items.slice(first, first + per).map(card));
      const last = Math.min(first + per, items.length);
      root.querySelector('.gal-count').textContent = per === 1 ? `${first + 1} / ${items.length}` : `${first + 1}–${last} / ${items.length}`;
      dots.innerHTML = Array.from({ length: pages }, (_, k) =>
        `<button type="button" data-page="${k}" aria-label="Examples page ${k + 1} of ${pages}"${k * per === first ? ' aria-current="true"' : ''}></button>`).join('');
    }
    function go(step) {
      const pages = Math.ceil(items.length / per);
      first = (((first / per + step) % pages + pages) % pages) * per;
      render();
    }
    root.addEventListener('click', (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      if (b.dataset.step) go(+b.dataset.step);
      if (b.dataset.page) { first = +b.dataset.page * per; render(); }
    });
    root.addEventListener('keydown', (e) => {
      if (e.target.classList && e.target.classList.contains('compare-range')) return;   // arrows move the slider there
      if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); go(1); }
    });
    let t;
    window.addEventListener('resize', () => {
      clearTimeout(t);
      t = setTimeout(() => { const p = perPage(); if (p !== per) { per = p; render(); } }, 150);
    });
    render();
  }

  // ---- BibTeX copy ------------------------------------------------------------
  function initCopy(btn) {
    btn.addEventListener('click', async () => {
      const src = document.querySelector(btn.dataset.target);
      try {
        await navigator.clipboard.writeText(src.textContent.trim());
        btn.textContent = 'Copied';
      } catch (e) {
        const r = document.createRange(); r.selectNodeContents(src);
        const s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
        btn.textContent = 'Selected, press Ctrl+C';
      }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1800);
    });
  }

  window.FLINT_initTraj = initTraj;
  window.FLINT_initCompare = initCompare;
  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.traj').forEach(initTraj);
    document.querySelectorAll('.compare').forEach(initCompare);
    document.querySelectorAll('.gallery').forEach(initGallery);
    document.querySelectorAll('[data-copy]').forEach(initCopy);
  });
})();
