(function () {
  var cfg = window.SITE_CONFIG || {};
  var form = document.getElementById("f");
  var err = document.getElementById("err");
  var btn = document.getElementById("go");
  var QUEUE_KEY = "pendingLead";

  function go() { window.location.href = cfg.nextUrl || "index.html"; }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    err.textContent = "";
    var email = form.email.value.trim();
    if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) {
      err.textContent = "Please enter a valid email, or tap Skip.";
      form.email.focus();
      return;
    }
    // Honeypot: bots fill the hidden field. Pretend success and send nothing.
    if (form.website.value) { go(); return; }

    var lead = { email: email, name: form.name.value.trim(), source: cfg.source || "", ts: new Date().toISOString() };
    btn.disabled = true;
    btn.textContent = "Saving...";

    var done = false;
    function finish() { if (!done) { done = true; go(); } }
    function queue() {
      try { localStorage.setItem(QUEUE_KEY, JSON.stringify(lead)); } catch (x) {}
    }

    if (!cfg.endpoint) { queue(); finish(); return; }

    // Conference wifi can be poor: never trap the visitor here.
    setTimeout(function () { queue(); finish(); }, 4000);
    fetch(cfg.endpoint, {
      method: "POST",
      mode: "no-cors",
      headers: { "Content-Type": "text/plain;charset=utf-8" },
      body: JSON.stringify(lead),
      keepalive: true
    }).then(function () {
      try { localStorage.removeItem(QUEUE_KEY); } catch (x) {}
      finish();
    }).catch(function () { queue(); finish(); });
  });
})();
