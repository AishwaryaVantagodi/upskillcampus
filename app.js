document.addEventListener("DOMContentLoaded", () => {
  const input = document.getElementById("image");
  const preview = document.getElementById("image-preview");
  const zone = document.getElementById("drop-zone");
  if (!input || !preview || !zone) return;
  input.addEventListener("change", () => {
    const file = input.files && input.files[0];
    if (!file) { preview.hidden = true; preview.removeAttribute("src"); return; }
    if (!file.type.startsWith("image/")) { input.value = ""; return; }
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    const title = zone.querySelector("strong");
    if (title) title.textContent = file.name;
  });
});
