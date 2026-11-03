window.toast = (msg, kind) => {
  let box = document.getElementById("toasts");
  if (!box) { box = document.createElement("div"); box.id = "toasts"; box.setAttribute("aria-live", "polite"); document.body.appendChild(box); }
  const t = document.createElement("div"); t.className = "toast " + (kind || "info"); t.textContent = msg; box.appendChild(t);
  setTimeout(() => t.remove(), 5000);
};
window.api = (path, opts) => fetch((window.API_BASE || "") + path, opts);
