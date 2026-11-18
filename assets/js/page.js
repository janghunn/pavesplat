/* PaveSplat page behaviour. Deferred, loaded before assets/js/kit.js; the inline renderer in index.html has already
   drawn the tables, plots and coverage chart when this runs.
   1. Korean for the words the renderer generates. Static output (table captions and headers, the coverage chart,
      its note) gets data-ko / data-ko-label here, so kit.js switches it with the rest of the page. The poster plots
      are redrawn on resize, so their <title>, <desc> and axis label are repainted after every redraw and on ws:lang.
      Method names, numbers and units are language-neutral and stay as drawn; a string missing from KO stays English.
   2. The top video (video[data-autoplay]): plays muted unless reduced motion is set, pauses while the tab is hidden,
      never restarts after the viewer pauses it, and gets a pause/play button (native controls stay). */
(() => {
  'use strict';
  const root = document.documentElement;
  const isKo = () => root.lang === 'ko';

  // English -> Korean, keyed by the exact strings the renderer and data/*.json produce.
  const KO = {
    'Method': '방법',
    'Scale': '배율',
    'Median (%)': '중앙값 (%)',
    'Mean (%)': '평균 (%)',
    'main Tab. 1 — Novel view synthesis on RealEstate10K across render resolutions. All cells are our measurements on the common index; multi-Gaussian methods (pixelSplat, HiSplat) are ruled off above the single-Gaussian-per-pixel group. SurfSplat‡: our same-recipe reproduction.':
      '본문 표 1 — RealEstate10K에서 렌더 해상도별 novel view synthesis 결과. 모든 값은 공통 인덱스에서 직접 측정한 값이며, 다중 Gaussian 방법(pixelSplat, HiSplat)은 픽셀당 단일 Gaussian 그룹 위에 선을 그어 구분했습니다. SurfSplat‡: 동일 레시피로 재현한 결과.',
    'main Fig. 4a — Per-scene medians, means and quartiles in percent, rounded to six decimal places. ‡ same-recipe reproduction.':
      '본문 그림 4a — 장면별 중앙값, 평균, 사분위수(%)이며 소수점 여섯째 자리까지 반올림했습니다. ‡ 동일 레시피 재현.',
    'Coverage holes: per-scene median, IQR band': '커버리지 구멍: 장면별 중앙값, IQR 띠',
    'Black-hole pixels (%) on a log axis across render scales. Values below 0.0001% are clamped for display; the data table reports unclamped medians and means.':
      '렌더 배율에 따른 black-hole 픽셀 비율(%)을 로그 축에 나타냅니다. 0.0001% 미만 값은 표시할 때만 고정하며, 데이터 표에는 고정하지 않은 중앙값과 평균을 싣습니다.',
    'Black-hole pixels (%) · log axis': 'Black-hole 픽셀 (%) · 로그 축',
    'Render scale': '렌더 배율',
    'per-scene median, IQR band. Values below 0.0001% (including zeros at 1×) are clamped to 0.0001% for log display only. Table values are unclamped. ‡ same-recipe reproduction.':
      '장면별 중앙값, IQR 띠. 0.0001% 미만 값(1×의 zero 값 포함)은 로그 축 표시를 위해서만 0.0001%로 고정합니다. 표의 값은 고정하지 않았습니다. ‡ 동일 레시피 재현.',
    'Coverage chart, scroll horizontally on small screens': '커버리지 차트, 작은 화면에서는 가로로 스크롤하세요',
    'RealEstate10K PSNR (dB) versus render scale': 'RealEstate10K 렌더 배율에 따른 PSNR (dB)',
    'Zero-shot DL3DV PSNR (dB) versus render scale, trained on RealEstate10K':
      'Zero-shot DL3DV 렌더 배율에 따른 PSNR (dB), RealEstate10K로 학습',
    'Coverage holes (% of pixels, log axis) versus render scale': '렌더 배율에 따른 커버리지 구멍 (픽셀의 %, 로그 축)',
    'holes (% of pixels)': '구멍 (픽셀의 %)',
  };
  const SOURCES = {'main Tab. 1': '본문 표 1', 'main Tab. 2': '본문 표 2',
    'supp Tab. 3 (mean, ε = 0.02)': '보충 자료 표 3 (평균, ε = 0.02)'};
  const PATTERNS = [
    // coverage chart: the <title> of each median line
    [/^(.+) per-scene median$/, (_, m) => `${m} 장면별 중앙값`],
    // poster plot <desc>: "<method>: <values>; ... at <scales>. Source: <source>."
    [/^(.+) at (.+)\. Source: (.+)\.$/, (_, series, scales, source) => `${series} (배율 ${scales}). 출처: ${SOURCES[source] || source}.`],
  ];
  const ko = en => {
    if (Object.hasOwn(KO, en)) return KO[en];
    for (const [re, to] of PATTERNS) if (re.test(en)) return en.replace(re, to);
    return null;
  };

  // 1a. Static renderer output: mark it for kit.js, which runs next.
  document.querySelectorAll('#re10k-table caption, #re10k-table thead th, #coverage-table caption, #coverage-table thead th,'
    + ' #coverage svg title, #coverage svg desc, #coverage svg text, #coverage-note').forEach(el => {
    const k = ko(el.textContent);
    if (k !== null) el.setAttribute('data-ko', k);
  });
  document.querySelectorAll('#re10k-table [aria-label], #coverage-table [aria-label], #coverage[aria-label]').forEach(el => {
    const k = ko(el.getAttribute('aria-label'));
    if (k !== null) el.setAttribute('data-ko-label', k);
  });

  // 1b. Poster plots: repaint after each redraw (the host's child svg is replaced) and on every language switch.
  const english = new WeakMap();
  const paintPlot = host => host.querySelectorAll('title, desc, .axis-label').forEach(el => {
    if (!english.has(el)) english.set(el, el.textContent);
    const en = english.get(el), k = isKo() ? ko(en) : null;
    el.textContent = k === null ? en : k;
  });
  const hosts = [...document.querySelectorAll('.plot-svg[data-plot]')];
  const redraws = new MutationObserver(records => records.forEach(r => paintPlot(r.target)));
  hosts.forEach(host => redraws.observe(host, {childList: true}));

  // 2. Top video.
  const video = document.querySelector('video[data-autoplay]');
  let paintVideo = () => {};
  if (video) {
    const labels = {en: {pause: video.dataset.labelPause || 'Pause video', play: video.dataset.labelPlay || 'Play video'},
      ko: {pause: '영상 일시정지', play: '영상 재생'}};
    const reduce = matchMedia('(prefers-reduced-motion: reduce)');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'ctl icon video-toggle';
    button.innerHTML = '<svg class="ico-pause" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>'
      + '<svg class="ico-play" viewBox="0 0 24 24" aria-hidden="true" focusable="false"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10-6.5a1 1 0 0 0 0-1.72l-10-6.5A1 1 0 0 0 8 5.5z"/></svg>';
    video.after(button);
    let viewerPaused = false, ownPause = false;
    paintVideo = () => {
      button.classList.toggle('is-paused', video.paused);
      button.setAttribute('aria-label', labels[isKo() ? 'ko' : 'en'][video.paused ? 'play' : 'pause']);
    };
    const play = () => {
      const p = video.play();
      // Blocked playback keeps the native controls; an AbortError only means a pause came first.
      if (p && p.catch) p.catch(e => { if (!e || e.name !== 'AbortError') video.controls = true; paintVideo(); });
    };
    const hold = () => { if (!video.paused) { ownPause = true; video.pause(); } };
    // A pause made by the viewer (this button or the native controls) is remembered; the page's own pauses are not.
    video.addEventListener('pause', () => { if (ownPause) ownPause = false; else viewerPaused = true; paintVideo(); });
    video.addEventListener('play', () => { viewerPaused = false; paintVideo(); });
    button.addEventListener('click', () => { if (video.paused) play(); else video.pause(); });
    const sync = () => {
      if (document.hidden || reduce.matches || viewerPaused) hold(); else play();
      paintVideo();
    };
    video.muted = true;
    document.addEventListener('visibilitychange', sync);
    reduce.addEventListener('change', sync);
    sync();
  }

  // kit.js fires ws:lang once when it starts, then on every switch.
  document.addEventListener('ws:lang', () => { hosts.forEach(paintPlot); paintVideo(); });
})();
