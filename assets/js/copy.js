// A copy button on every code block. The page works without this file; a
// browser with no clipboard API simply gets no buttons.
(function () {
  if (!navigator.clipboard) return;
  var svg = '<svg viewBox="0 0 16 16" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">';
  var copyIcon = svg + '<rect x="5.5" y="5.5" width="8" height="8" rx="1.5"/><path d="M10.5 5.5V4A1.5 1.5 0 0 0 9 2.5H4A1.5 1.5 0 0 0 2.5 4v5A1.5 1.5 0 0 0 4 10.5h1.5"/></svg>';
  var doneIcon = svg + '<path d="M3 8.5l3.5 3.5L13 4.5"/></svg>';
  document.querySelectorAll('div.highlighter-rouge').forEach(function (block) {
    var pre = block.querySelector('pre');
    if (!pre) return;
    var button = document.createElement('button');
    button.type = 'button';
    button.className = 'copy';
    button.title = 'Copy';
    button.setAttribute('aria-label', 'Copy');
    button.innerHTML = copyIcon;
    button.addEventListener('click', function () {
      navigator.clipboard.writeText(pre.textContent.replace(/\s+$/, '')).then(function () {
        button.innerHTML = doneIcon;
        button.setAttribute('aria-label', 'Copied');
        setTimeout(function () {
          button.innerHTML = copyIcon;
          button.setAttribute('aria-label', 'Copy');
        }, 2000);
      });
    });
    block.appendChild(button);
  });
})();
