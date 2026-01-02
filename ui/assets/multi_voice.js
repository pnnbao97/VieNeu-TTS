async () => {
  const getRoot = () => {
    const app = document.querySelector('gradio-app');
    if (!app) return document;
    return app.shadowRoot || app;
  };

  const getOrderInput = () => {
    const root = getRoot();
    return root.querySelector('#multi-line-order textarea, #multi-line-order input');
  };

  const getCards = (container) => {
    if (!container) return [];
    return Array.from(container.querySelectorAll('.multi-line-card'))
      .filter(el => el.offsetParent !== null);
  };

  const setTitleIfNeeded = (el, title) => {
    if (!el) return;
    if (el.getAttribute('title') === title) return;
    el.setAttribute('title', title);
  };

  const setTooltips = (root) => {
    const tooltipMap = [
      ['#multi-add-line, #multi-add-line button', 'Add speaker'],
      ['#multi-expand-all, #multi-expand-all button', 'Expand all'],
      ['#multi-remove-line, #multi-remove-line button', 'Remove last speaker'],
      ['#multi-clear-lines, #multi-clear-lines button', 'Clear all'],
    ];
    tooltipMap.forEach(([selector, title]) => {
      const el = root.querySelector(selector);
      setTitleIfNeeded(el, title);
    });

    const deleteButtons = root.querySelectorAll(
      '.multi-line-card .icon-danger, .multi-line-card .icon-danger button'
    );
    deleteButtons.forEach(btn => setTitleIfNeeded(btn, 'Remove speaker'));
  };

  const syncSingleTextVisibility = () => {
    const root = getRoot();
    const tabButton = root.querySelector('#tab-multi-button');
    const textInput = root.querySelector('#single-text-input');
    if (!tabButton || !textInput) return;
    const isActive = tabButton.getAttribute('aria-selected') === 'true';
    textInput.style.display = isActive ? 'none' : '';
  };

  const setOrder = (container) => {
    const orderInput = getOrderInput();
    if (!container || !orderInput) return;
    const order = getCards(container)
      .map(el => el.id || '')
      .filter(id => id.startsWith('multi-line-'))
      .map(id => id.replace('multi-line-', ''));
    const next = order.join(',');
    if (container.dataset.multiOrder === next) return;
    container.dataset.multiOrder = next;
    if (orderInput.value === next) return;
    orderInput.value = next;
    orderInput.dispatchEvent(new Event('input', { bubbles: true }));
  };

  const getDragAfterElement = (container, y, draggingCard) => {
    const cards = getCards(container).filter(el => el !== draggingCard);
    let closest = null;
    let closestOffset = Number.NEGATIVE_INFINITY;
    cards.forEach(card => {
      const rect = card.getBoundingClientRect();
      const offset = y - rect.top - rect.height / 2;
      if (offset < 0 && offset > closestOffset) {
        closestOffset = offset;
        closest = card;
      }
    });
    return closest;
  };

  let activeDrag = null;
  let listenersBound = false;
  let rootObserver = null;
  let lastContainer = null;
  let pendingContainerRefresh = false;

  const getLineTextArea = (card) => {
    if (!card) return null;
    return card.querySelector('.multi-line-text textarea');
  };

  const normalizeSummary = (value) => {
    if (!value) return '';
    return value.replace(/\s+/g, ' ').trim();
  };

  const ensureSummaryElement = (card) => {
    if (!card) return null;
    let summary = card.querySelector('.collapsed-summary');
    if (summary) return summary;
    summary = document.createElement('div');
    summary.className = 'collapsed-summary';
    const header = card.querySelector('.multi-line-header');
    if (header) {
      header.insertAdjacentElement('afterend', summary);
    } else {
      card.appendChild(summary);
    }
    return summary;
  };

  const updateSummary = (card) => {
    const summary = ensureSummaryElement(card);
    if (!summary) return;
    const area = getLineTextArea(card);
    const text = normalizeSummary(area ? area.value : '');
    const next = text || 'Chua co noi dung';
    if (summary.textContent === next) return;
    summary.textContent = next;
  };

  const updateSummaries = (container) => {
    getCards(container).forEach(updateSummary);
  };

  const setAllActive = (container) => {
    if (!container) return;
    const cards = getCards(container);
    if (!cards.length) return;
    cards.forEach(card => {
      card.classList.remove('collapsed');
      card.classList.add('active');
      updateSummary(card);
    });
  };

  const setActiveCard = (container, nextCard) => {
    if (!container || !nextCard) return;
    container.dataset.multiExpandAll = 'false';
    const cards = getCards(container);
    if (!cards.length) return;
    cards.forEach(card => {
      const isActive = card === nextCard;
      if (isActive) {
        if (card.classList.contains('collapsed')) {
          card.classList.remove('collapsed');
        }
        if (!card.classList.contains('active')) {
          card.classList.add('active');
        }
      } else {
        if (!card.classList.contains('collapsed')) {
          card.classList.add('collapsed');
        }
        if (card.classList.contains('active')) {
          card.classList.remove('active');
        }
      }
      updateSummary(card);
    });
  };

  const ensureActiveCard = (container) => {
    const cards = getCards(container);
    if (!cards.length) return;
    let active = cards.find(card => card.classList.contains('active'));
    if (!active) {
      active = cards[0];
    }
    setActiveCard(container, active);
  };

  const applyActiveMode = (container) => {
    if (!container) return;
    if (container.dataset.multiExpandAll === 'true') {
      setAllActive(container);
      return;
    }
    ensureActiveCard(container);
  };

  const onCardClick = (e) => {
    const container = e.currentTarget;
    if (!container) return;
    if (e.target && e.target.closest && e.target.closest('.drag-handle')) return;
    const card = e.target.closest('.multi-line-card');
    if (!card || !container.contains(card)) return;
    if (container.dataset.multiExpandAll === 'true') {
      setActiveCard(container, card);
      return;
    }
    if (card.classList.contains('active')) return;
    setActiveCard(container, card);
  };

  const onLineInput = (e) => {
    const container = e.currentTarget;
    if (!container) return;
    const target = e.target;
    if (!target || !target.closest) return;
    if (!target.closest('.multi-line-text')) return;
    const card = target.closest('.multi-line-card');
    if (!card || !container.contains(card)) return;
    updateSummary(card);
  };

  const bindContainerListeners = (container) => {
    if (!container || container.dataset.multiCollapseBound === 'true') return;
    container.dataset.multiCollapseBound = 'true';
    container.addEventListener('click', onCardClick);
    container.addEventListener('input', onLineInput);
  };

  const onExpandAllClick = (e) => {
    e.preventDefault();
    e.stopPropagation();
    const root = getRoot();
    const container = root.querySelector('#multi-lines-container');
    if (!container) return;
    container.dataset.multiExpandAll = 'true';
    setAllActive(container);
  };

  const bindExpandAllButton = (root) => {
    const button = root.querySelector('#multi-expand-all, #multi-expand-all button');
    if (!button || button.dataset.multiExpandBound === 'true') return;
    button.dataset.multiExpandBound = 'true';
    button.addEventListener('click', onExpandAllClick);
  };

  const scheduleContainerRefresh = () => {
    if (pendingContainerRefresh) return;
    pendingContainerRefresh = true;
    requestAnimationFrame(() => {
      pendingContainerRefresh = false;
      const root = getRoot();
      const container = root.querySelector('#multi-lines-container');
      if (!container) return;
      setOrder(container);
      setTooltips(root);
      updateSummaries(container);
      applyActiveMode(container);
    });
  };

  const findHandleFromEvent = (event) => {
    if (!event) return null;
    if (event.target && event.target.closest) {
      const direct = event.target.closest('.drag-handle');
      if (direct) return direct;
    }
    if (typeof event.composedPath !== 'function') return null;
    const path = event.composedPath();
    for (const el of path) {
      if (el && el.classList && el.classList.contains('drag-handle')) {
        return el;
      }
    }
    return null;
  };

  const onPointerDown = (e) => {
    if (e.button !== 0) return;
    const handle = findHandleFromEvent(e);
    if (!handle) return;
    const card = handle.closest('.multi-line-card');
    const container = card ? card.closest('#multi-lines-container') : null;
    if (!card || !container) return;
    activeDrag = { container, card };
    card.classList.add('dragging');
    document.body.classList.add('multi-dragging');
    e.preventDefault();
  };

  const onPointerMove = (e) => {
    if (!activeDrag) return;
    const { container, card } = activeDrag;
    const after = getDragAfterElement(container, e.clientY, card);
    if (after === null) {
      container.appendChild(card);
    } else if (after !== card) {
      container.insertBefore(card, after);
    }
  };

  const onPointerUp = () => {
    if (!activeDrag) return;
    const { container, card } = activeDrag;
    card.classList.remove('dragging');
    activeDrag = null;
    document.body.classList.remove('multi-dragging');
    setOrder(container);
  };

  const bindPointerListeners = () => {
    if (listenersBound) return;
    listenersBound = true;
    window.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
  };

  const ensureRootObserver = () => {
    if (rootObserver) return;
    const root = getRoot();
    rootObserver = new MutationObserver(() => {
      const container = getRoot().querySelector('#multi-lines-container');
      if (container && container !== lastContainer) {
        lastContainer = container;
        bindContainerListeners(container);
        bindExpandAllButton(getRoot());
        syncSingleTextVisibility();
        scheduleContainerRefresh();
      }
    });
    rootObserver.observe(root, { childList: true, subtree: true });
  };

  const initMultiVoice = () => {
    const root = getRoot();
    const container = root.querySelector('#multi-lines-container');
    if (!container) {
      setTimeout(initMultiVoice, 500);
      return;
    }

    lastContainer = container;
    bindContainerListeners(container);
    bindPointerListeners();
    ensureRootObserver();
    bindExpandAllButton(root);
    syncSingleTextVisibility();
    scheduleContainerRefresh();

    const observer = new MutationObserver(() => {
      if (!container.isConnected) {
        observer.disconnect();
        setTimeout(initMultiVoice, 200);
        return;
      }
      scheduleContainerRefresh();
    });
    observer.observe(container, { childList: true, subtree: true, attributes: true });

    const tabButton = root.querySelector('#tab-multi-button');
    if (tabButton) {
      const tabObserver = new MutationObserver(syncSingleTextVisibility);
      tabObserver.observe(tabButton, { attributes: true, attributeFilter: ['aria-selected', 'class'] });
    }
  };

  try {
    if (!window.__multiVoiceInit) {
      window.__multiVoiceInit = true;
      if (document.readyState === 'loading') {
        window.addEventListener('load', initMultiVoice);
      } else {
        initMultiVoice();
      }
    }
  } catch (err) {
    console.error('multi_voice init failed', err);
  }
  return [];
}
