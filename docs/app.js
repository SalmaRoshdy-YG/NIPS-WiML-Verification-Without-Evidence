(function () {
  var cfg = window.SITE_CONFIG || {};
  var KEY = "pendingLead";
  // Greeting banner after the connect form.
  if (/[?&]welcome=1/.test(location.search)) {
    var w = document.getElementById("welcome");
    if (w) w.hidden = false;
  }
  // If the connect form could not reach the server (poor wifi), retry quietly now.
  try {
    var raw = localStorage.getItem(KEY);
    if (raw && cfg.endpoint) {
      fetch(cfg.endpoint, { method: "POST", mode: "no-cors",
        headers: { "Content-Type": "text/plain;charset=utf-8" }, body: raw })
        .then(function () { localStorage.removeItem(KEY); }).catch(function () {});
    }
  } catch (x) {}
})();
