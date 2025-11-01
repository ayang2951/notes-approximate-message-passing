/* app.js – with max-height collapsibles + auto-numbering */

marked.setOptions({ gfm:true, mangle:false, headerIds:false });

const ORDERED_NOTES = [
  { file: "highdstats.md", title: "Introduction to High-D Stats" },
  { file: "amp.md", title: "AMP Algorithms" }
];

function $id(id){ return document.getElementById(id); }
const contentEl = () => $id('content');
const tocEl = () => $id('toc');

async function loadAll(){
  const parts = await Promise.all(ORDERED_NOTES.map(async({file,title},i)=>{
    try {
      const res = await fetch(`notes/${file}`);
      if(!res.ok) return `<section data-sec="${i+1}"><h1>${title}</h1><blockquote>⚠️ Missing ${file}</blockquote></section>`;
      const md = await res.text();
      const html = marked.parse(md);
      return `<section id="sec-${i+1}" class="note-section" data-sec="${i+1}"><h1>${title}</h1>${html}</section>`;
    } catch(e){
      return `<section data-sec="${i+1}"><h1>${title}</h1><blockquote>⚠️ Load error</blockquote></section>`;
    }
  }));
  contentEl().innerHTML = parts.join("\n");

  if(window.MathJax?.typesetPromise){ await MathJax.typesetPromise([contentEl()]); }

  removeEmptyPlaceholders();
  buildToc();
  autoNumberCallouts();
  addBookmarkButtons();
}

function removeEmptyPlaceholders() {
  // Remove empty paragraph/div/mjx nodes (including inside callouts)
  const nodes = document.querySelectorAll(
    '#content p, #content div, #content .MathJax_Display, #content mjx-container, #content .MathJax, .callout p, .callout mjx-container, .callout .MathJax_Display'
  );

  nodes.forEach(el => {
    const text = el.textContent?.trim() || '';
    const hasVisibleChild = Array.from(el.children).some(c =>
      c.tagName === 'IMG' || c.tagName === 'SVG' || c.querySelector('*') || c.textContent.trim() !== ''
    );
    if (!hasVisibleChild && text === '') {
      el.remove();
    }
  });
}


function buildToc(){
  const secs = Array.from(document.querySelectorAll('.note-section > h1'));
  tocEl().innerHTML = `<div class="toc-list">${secs.map(h=>`<a href="#${h.parentElement.id}">${h.textContent}</a>`).join('')}</div>`;
}

/* Auto-number callouts with colon */
/* Replace existing autoNumberCallouts() with this block */
function autoNumberCallouts(){
  document.querySelectorAll('.note-section').forEach(sec=>{
    const secIdx = sec.dataset.sec || (sec.id||'').replace(/^[^\d]*(\d+).*$/,'$1') || '0';
    const counters = {definition:0, proposition:0, lemma:0, theorem:0, remark:0, corollary:0, example:0};

    sec.querySelectorAll('.callout').forEach(c=>{
      for(const t in counters){
        if(c.classList.contains(t)){
          counters[t]++;
          const num = `${secIdx}.${counters[t]}`;
          const labelEl = c.querySelector('.label');
          if(labelEl){
            let raw = labelEl.textContent.trim();
            const afterType = raw.replace(/^[A-Za-z]+\s*[:(]?\s*/,'').trim();
            const titlePart = afterType ? `: ${afterType}` : '';
            const typeWord = t.charAt(0).toUpperCase() + t.slice(1);

            labelEl.innerHTML =
              '<span class="callout-type">' + typeWord + '</span> ' +
              '<span class="callout-num">' + num + '</span>' +
              '<span class="callout-title">' + titlePart + '</span>';
          }
          
          if (!c.id) c.id = `${t}-${secIdx}-${counters[t]}`;
          break;
        }
      }
    });
  });
}


/* Theme toggle */
function getTheme(){ return document.documentElement.getAttribute('data-theme')||'light'; }
function setTheme(t,b){ document.documentElement.setAttribute('data-theme',t); localStorage.setItem('theme',t); if(b){ b.textContent=(t==='dark'?'🌙':'☀️'); b.setAttribute('aria-pressed',t==='dark'); } }

document.addEventListener('DOMContentLoaded',()=>{
  const btn=$id('themeToggle');
  if(btn){ setTheme(getTheme(),btn); btn.addEventListener('click',()=>setTheme(getTheme()==='dark'?'light':'dark',btn)); }
  loadAll();
});

/* ----- Bookmark helpers (replace old versions) ----- */

function getTextExcluding(el, selectorToExclude = '.bookmark-btn') {
  const clone = el.cloneNode(true);
  clone.querySelectorAll(selectorToExclude).forEach(n => n.remove());
  return clone.textContent.trim().replace(/\s+/g, ' ');
}

function findNearestHeading(el) {
  let prev = el.previousElementSibling;
  while (prev) {
    if (['H1','H2','H3'].includes(prev.tagName)) return prev;
    prev = prev.previousElementSibling;
  }
  const sec = el.closest('section');
  if (sec) return sec.querySelector('h1');
  return null;
}

function addBookmarkButtons() {
  // clear any old ones
  document.querySelectorAll('.bookmark-btn').forEach(b => b.remove());

  const blocks = document.querySelectorAll('.callout, p');
  blocks.forEach((el, idx) => {
    if (!el.id) el.id = `bookmarkable-${idx}`;

    let container = el;
    if (el.classList.contains('callout')) {
      const label = el.querySelector('.label');
      if (label) container = label;
    }

    container.dataset.plain = getTextExcluding(container, '.bookmark-btn');
    if (container.querySelector('.bookmark-btn')) return;

    const btn = document.createElement('button');
    btn.className = 'bookmark-btn';
    btn.type = 'button';
    btn.title = 'Toggle bookmark';
    btn.setAttribute('aria-label', 'Toggle bookmark');
    btn.textContent = '🔖';

    btn.addEventListener('click', (ev) => {
      ev.stopPropagation();
      toggleBookmark(el.id);
    });

    container.appendChild(btn);
  });

  renderBookmarks();
}

function getBookmarks() {
  try {
    return JSON.parse(localStorage.getItem('bookmarks') || '[]');
  } catch {
    return [];
  }
}
function saveBookmarks(arr) {
  localStorage.setItem('bookmarks', JSON.stringify(arr));
}
function toggleBookmark(id) {
  const arr = getBookmarks();
  const i = arr.indexOf(id);
  if (i >= 0) arr.splice(i, 1);
  else arr.push(id);
  saveBookmarks(arr);
  renderBookmarks();
}
function removeBookmark(id) {
  const arr = getBookmarks().filter(x => x !== id);
  saveBookmarks(arr);
  renderBookmarks();
}

function renderBookmarks() {
  const bookmarks = getBookmarks();
  const panelList = document.getElementById('bookmarkList');
  if (!panelList) return;

  // reset counters every render
  let h1Count = 0;
  let h2Count = 0;
  let h3Count = 0;


  panelList.innerHTML = '';

  document.querySelectorAll('.bookmarkable, .callout').forEach(el => {
    el.classList.toggle('bookmarked', bookmarks.includes(el.id));
  });

  let headings = Array.from(document.querySelectorAll('h1, h2, h3'))
  .filter(h => !h.closest('#bookmarkPanel'));


  // drop the very first h1 (the document title)
  if (headings.length && headings[0].tagName === 'H1') {
    headings = headings.slice(1);
  }

  const map = new Map();
  headings.forEach(h => map.set(h, []));

  bookmarks.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    const nearest = findNearestHeading(el);
    const headingKey = nearest || headings[0] || null;
    if (headingKey) {
      if (!map.has(headingKey)) map.set(headingKey, []);
      map.get(headingKey).push(el);
    }
  });

  headings.forEach(h => {
    const li = document.createElement('li');
    li.className = `heading-item level-${parseInt(h.tagName.slice(1), 10)}`;

    const headingText = (h.dataset && h.dataset.plain)
      ? h.dataset.plain
      : getTextExcluding(h, '.bookmark-btn');

    // compute hierarchical numbering
    let numberLabel = '';
    if (h.tagName === 'H1') {
      h1Count++;
      h2Count = 0;
      h3Count = 0;
      numberLabel = h1Count + '. ';
    }
    else if (h.tagName === 'H2') {
      h2Count++;
      h3Count = 0;
      numberLabel = h1Count + '.' + h2Count + ' ';
    }
    else if (h.tagName === 'H3') {
      h3Count++;
      numberLabel = h1Count + '.' + h2Count + '.' + h3Count + ' ';
    }


    const strong = document.createElement('strong');
    strong.textContent = numberLabel + (headingText || '(Untitled)');
    li.appendChild(strong);


    const childBookmarks = map.get(h) || [];
    if (childBookmarks.length) {
      const ul = document.createElement('ul');
      ul.className = 'heading-bookmark-list';
      childBookmarks.forEach(bookEl => {
        const subli = document.createElement('li');
        subli.className = 'bookmark-item';

        const a = document.createElement('a');
        a.href = '#' + bookEl.id;

        if (bookEl.classList.contains('callout')) {
          const lab = bookEl.querySelector('.label');
          a.textContent = lab
            ? (lab.dataset && lab.dataset.plain ? lab.dataset.plain : getTextExcluding(lab, '.bookmark-btn'))
            : (bookEl.dataset.plain || bookEl.textContent.slice(0, 50));
        } else {
          a.textContent = bookEl.dataset && bookEl.dataset.plain
            ? bookEl.dataset.plain
            : getTextExcluding(bookEl, '.bookmark-btn');
        }

        a.addEventListener('click', (ev) => {
          ev.preventDefault();
          bookEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });

        const rm = document.createElement('button');
        rm.type = 'button';
        rm.className = 'bookmark-remove';
        rm.title = 'Remove bookmark';
        rm.textContent = '🔖';
        rm.addEventListener('click', () => removeBookmark(bookEl.id));

        subli.appendChild(a);
        subli.appendChild(rm);
        ul.appendChild(subli);
      });
      li.appendChild(ul);
    }

    panelList.appendChild(li);
  });
}



/* --- Panel toggle --- */
document.addEventListener('DOMContentLoaded', () => {
  const toggleBtn = document.getElementById('toggleBookmarks');
  const panel = document.getElementById('bookmarkPanel');
  if(toggleBtn && panel){
    toggleBtn.addEventListener('click', () => {
      panel.style.display = (panel.style.display === 'block' ? 'none' : 'block');
    });
  }
});
