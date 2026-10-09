window.unsaveAll = async () => {
  const delay = 3000 + Math.random() * 2000;
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const tiles = () => document.querySelectorAll('main a[href^="/p/"], main a[href^="/reel/"]');
  let n = 0;
  for (tile of tiles()) {
    if (!tile) break;
    tile.click();                       // open post modal over the saved grid
    await sleep(delay);
    const svg = document.querySelector('svg[aria-label="Remove"]');
    if (!svg) { console.warn('no Remove button — stopped after', n); break; }
    (svg.closest('[role="button"]') || svg).click();
    await sleep(delay);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await sleep(delay);
    console.log('unsaved:', ++n, "left:", tiles().length - n , "slept:", (delay*3 / 1000).toFixed(1), "s");
  }
  return n;
};
unsaveAll().then(n => console.log('done unsaving', n, 'posts'));


window.unsaveAll = async () => {
  const sleep = () => {
    const ms = 2000 + Math.random() * 2000;
    console.log('sleeping:', (ms / 1000).toFixed(1), 's');
    return new Promise(r => setTimeout(r, ms));
  };
  const tiles = () => document.querySelectorAll('main a[href^="/p/"], main a[href^="/reel/"]');
  let alltiles = tiles();
  let n = 0;
  for (tile of alltiles) {
    if (!tile) break;
    tile.click();
    await sleep();
    const svg = document.querySelector('svg[aria-label="Remove"]');
    if (!svg) { console.warn('no Remove button — stopped after', n); continue; }
    (svg.closest('[role="button"]') || svg).click();
    await sleep();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    await sleep();
    console.log('unsaved:', ++n, 'left:', alltiles.length - n);
  }
  return n;
};
unsaveAll().then(n => console.log('done unsaving', n, 'posts'));

clickUnsaveAll = async () => {
    const svg = document.querySelector('svg[aria-label="Remove"]');
    if (!svg) { console.warn('no Remove button — stopped after', n);  }
    (svg.closest('[role="button"]') || svg).click();
}
clickUnsaveAll().then(n => console.log('done unsaving', n, 'posts'));



window.unsaveAll = async () => {
  const sleep = () => {
    const ms = 2000 + Math.random() * 2000;
    console.log('[unsaveAll] sleeping:', (ms / 1000).toFixed(1), 's');
    return new Promise(r => setTimeout(r, ms));
  };

  const tiles = () => Array.from(document.querySelectorAll('main a[href^="/p/"], main a[href^="/reel/"]'));
  const tile = tiles()[0];
  if (!tile) {
    console.warn('[unsaveAll] no tile found');
    return 0;
  }

  console.log('[unsaveAll] starting with tile:', tile.href || tile.outerHTML.slice(0, 120));
  // click the tile to open the post modal over the saved grid
  tile.click();
  console.log('[unsaveAll] clicked tile to open modal');

  let saved = 0;
  for (let n = 0; n < 10; n++) {
    await sleep();
    const svg = document.querySelector('svg[aria-label="Remove"]');
    console.log('[unsaveAll] iteration', n + 1, 'remove button present:', !!svg);

    if (!svg) {
      console.warn('[unsaveAll] no Remove button — stopped after', saved, 'items');
    console.log('[unsaveAll] dispatching ArrowRight to advance to next post');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
      continue;
    }

    const button = svg.closest('[role="button"]') || svg;
    console.log('[unsaveAll] clicking Remove button:', button);
    button.click();

    await sleep();
    console.log('[unsaveAll] dispatching Escape to close modal');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

    await sleep();
    console.log('[unsaveAll] dispatching ArrowRight to advance to next post');
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));

    await sleep();
    saved += 1;
    console.log('[unsaveAll] unsaved:', saved, 'current index:', n + 1);
  }

  console.log('[unsaveAll] finished; total unsaved:', saved);
  return saved;
}

unsaveAll().then(n => console.log('done unsaving', n, 'posts'));


righArrowAndUnsave = async () => {
  const sleep = () => {
    const ms = 1000 + Math.random() * 1000;
    console.log('[righArrowAndUnsave] sleeping:', (ms / 1000).toFixed(1), 's');
    return new Promise(r => setTimeout(r, ms));
  };
  console.log('[righArrowAndUnsave] dispatching ArrowRight to advance to next post');
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
  await sleep();

    const svg = document.querySelector('svg[aria-label="Remove"]');
    console.log('[righArrowAndUnsave] remove button present:', !!svg);
    if (!svg) {
      console.warn('[righArrowAndUnsave] no Remove button found');
      return false;
    }
    
    const button = svg.closest('[role="button"]') || svg;
    console.log('[righArrowAndUnsave] clicking Remove button:', button);
    button.click();
}

const unsaveAllWithArrow = async () => {
    for (let n = 0; n < 1000; n++) {
        console.log('[unsaveAllWithArrow] iteration', n + 1);
        await righArrowAndUnsave();
    }
}
unsaveAllWithArrow().then(n => console.log('done unsaving', n, 'posts'));
