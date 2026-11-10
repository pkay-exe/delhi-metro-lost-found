// Shows the metro lines at the chosen station and fills the example on the home page.
(function () {
  var input = document.getElementById("station");
  var out = document.getElementById("station-lines");
  var lines = window.STATION_LINES || {};

  function showLines() {
    if (!input || !out) return;
    var found = lines[input.value.trim()];
    out.textContent = "";
    (found || []).forEach(function (l) {
      var span = document.createElement("span");
      span.className = "pill";
      span.style.setProperty("--c", l.color);
      span.style.setProperty("--k", l.ink);
      span.textContent = l.name;
      out.appendChild(span);
    });
  }
  if (input) {
    input.addEventListener("input", showLines);
    input.addEventListener("change", showLines);
    showLines();
  }

  var example = document.getElementById("fill-example");
  if (example) {
    example.addEventListener("click", function () {
      var d = new Date();
      d.setDate(d.getDate() - 4);
      var iso = new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
      document.getElementById("description").value = "Black wallet with college ID and bank cards";
      document.getElementById("category").value = "Wallet";
      input.value = "Hauz Khas";
      document.getElementById("date").value = iso;
      showLines();
      document.getElementById("description").focus();
    });
  }
})();
