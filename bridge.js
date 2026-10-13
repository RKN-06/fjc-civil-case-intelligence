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
    const responses = (msg.args && msg.args.responses) || {};
    for (const [id, answer] of Object.entries(responses)) {
      const resolve = pending.get(id);
      if (resolve) { pending.delete(id); resolve(answer); }
    }
  });
  send('streamlit:componentReady', { apiVersion: 1 });
  send('streamlit:setFrameHeight', { height: window.parent === window ? 0 : 900 });

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
