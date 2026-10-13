// Runs the original dashboard inside a Streamlit component.
// GET /api/... is answered from bundled JSON files; POST /api/predict is sent to Python and the
// answer comes back in the component's render arguments.
(function () {
  const pending = new Map();
  let seq = 0;
  const send = (type, data) => window.parent.postMessage(Object.assign({ isStreamlitMessage: true, type }, data), '*');
  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (!msg || msg.type !== 'streamlit:render') return;
    lastHeight = 0; setTimeout(() => { fit(); pin(); }, 0);  // a rerun can reset the frame height
    const responses = (msg.args && msg.args.responses) || {};
    for (const [id, answer] of Object.entries(responses)) {
      const resolve = pending.get(id);
      if (resolve) { pending.delete(id); resolve(answer); }
    }
  });
  send('streamlit:componentReady', { apiVersion: 1 });

  // Streamlit component iframes cannot scroll themselves, so the iframe grows to the page's full
  // height and the Streamlit page scrolls instead. The sidebar follows that scroll to stay in view.
  var lastHeight = 0;
  function fit() {
    // Measure the content, not the document: the document is never shorter than the frame itself.
    const main = document.querySelector('main'), aside = document.querySelector('aside');
    const height = main ? Math.ceil(main.getBoundingClientRect().height + (window.innerWidth <= 700 && aside ? aside.offsetHeight : 0))
      : Math.ceil(document.documentElement.scrollHeight);
    if (Math.abs(height - lastHeight) > 1) { lastHeight = height; send('streamlit:setFrameHeight', { height }); }
  }
  function pin() {
    const aside = document.querySelector('aside');
    const host = window.frameElement;
    if (!aside || !host) return;
    if (window.innerWidth <= 700) { aside.style.cssText = ''; return; }
    const top = Math.max(0, Math.min(-host.getBoundingClientRect().top, lastHeight - window.parent.innerHeight));
    aside.style.cssText = `position:absolute;top:${top}px;bottom:auto;height:${window.parent.innerHeight}px`;
  }
  document.addEventListener('DOMContentLoaded', () => {
    const watch = new ResizeObserver(() => { fit(); pin(); });
    watch.observe(document.querySelector('main') || document.body);
    setInterval(fit, 400);  // backstop for content that changes size without resizing <main>
    fit(); pin();
  });
  try {
    window.parent.document.addEventListener('scroll', pin, { capture: true, passive: true });
    window.parent.addEventListener('resize', pin);
  } catch (e) { /* parent not reachable: the sidebar simply scrolls with the page */ }

  // Chrome does not pass wheel, touch or key scrolling from a scrolling="no" frame to the page around
  // it, so forward it to Streamlit's scroll container unless something inside the dashboard used it.
  const outer = () => {
    try {
      let el = window.frameElement && window.frameElement.parentElement;
      while (el) {
        const oy = window.parent.getComputedStyle(el).overflowY;
        if ((oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight) return el;
        el = el.parentElement;
      }
      return window.parent.document.scrollingElement;
    } catch (e) { return null; }
  };
  const innerCanScroll = (node, dy) => {
    for (let el = node; el && el !== document.body && el.nodeType === 1; el = el.parentElement) {
      const oy = getComputedStyle(el).overflowY;
      if ((oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight &&
          (dy > 0 ? el.scrollTop + el.clientHeight < el.scrollHeight - 1 : el.scrollTop > 0)) return true;
    }
    return false;
  };
  const scrollBy = (dy) => { const el = outer(); if (el) { el.scrollTop += dy; pin(); } };
  Element.prototype.scrollIntoView = function (arg) {
    fit();
    setTimeout(() => reveal(this, arg), 120);  // let Streamlit apply the new frame height first
  };
  function reveal(target, arg) {
    const el = outer();
    if (!el || !window.frameElement) return;
    const block = (arg && typeof arg === 'object' && arg.block) || (arg === false ? 'end' : 'start');
    const r = target.getBoundingClientRect(), frame = window.frameElement.getBoundingClientRect();
    const box = el === window.parent.document.scrollingElement ? { top: 0, height: window.parent.innerHeight } : el.getBoundingClientRect();
    const top = frame.top + r.top - box.top;
    const offset = block === 'center' ? top - (box.height - r.height)/2 : block === 'end' ? top - box.height + r.height : top;
    el.scrollTo({ top: el.scrollTop + offset, behavior: (arg && arg.behavior) || 'auto' });
    pin();
  }
  window.addEventListener('wheel', (e) => {
    if (e.defaultPrevented || e.ctrlKey || innerCanScroll(e.target, e.deltaY)) return;
    scrollBy(e.deltaMode === 1 ? e.deltaY*40 : e.deltaMode === 2 ? e.deltaY*window.parent.innerHeight : e.deltaY);
  });
  let touchY = null;
  window.addEventListener('touchstart', (e) => { touchY = e.touches.length === 1 ? e.touches[0].clientY : null; }, { passive: true });
  window.addEventListener('touchmove', (e) => {
    if (touchY === null || e.defaultPrevented) return;
    const y = e.touches[0].clientY; scrollBy(touchY - y); touchY = y;
  }, { passive: true });
  window.addEventListener('keydown', (e) => {
    if (e.defaultPrevented || /INPUT|SELECT|TEXTAREA/.test((e.target && e.target.tagName) || '')) return;
    const page = window.parent.innerHeight*0.85;
    const step = { ArrowDown: 60, ArrowUp: -60, PageDown: page, PageUp: -page, ' ': e.shiftKey ? -page : page }[e.key];
    if (step) { e.preventDefault(); scrollBy(step); }
    else if (e.key === 'Home' || e.key === 'End') { const el = outer(); if (el) { e.preventDefault(); el.scrollTop = e.key === 'Home' ? 0 : el.scrollHeight; } }
  });

  const json = (body, status) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, options = {}) => {
    const url = String(input);
    if (url.startsWith('/api/predict') && String(options.method || 'GET').toUpperCase() === 'POST') {
      const id = 'p' + Date.now().toString(36) + '-' + (seq++);
      const answer = await new Promise((resolve) => {
        pending.set(id, resolve);
        send('streamlit:setComponentValue', { value: { id, body: JSON.parse(options.body || '{}') }, dataType: 'json' });
      });
      return json(answer.body, answer.status);
    }
    if (url.startsWith('/api/')) {
      const response = await realFetch('data' + url.split('?')[0] + '.json');
      return response.ok ? response : json({ detail: 'This view is not available in the hosted app.' }, 404);
    }
    return realFetch(input, options);
  };
})();
