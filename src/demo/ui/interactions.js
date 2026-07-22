() => {
  if (window.__verifyHinglishInteractions) {
    return;
  }
  window.__verifyHinglishInteractions = true;

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  if (!reducedMotion.matches) {
    document.documentElement.classList.add("vh-motion-ready");
  }

  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 }
  );

  const bindReveals = () => {
    document.querySelectorAll(".vh-reveal:not([data-vh-bound])").forEach((element) => {
      element.dataset.vhBound = "true";
      if (reducedMotion.matches) {
        element.classList.add("is-visible");
      } else {
        revealObserver.observe(element);
      }
    });
  };

  let scrollFrame = 0;
  const syncNavbar = () => {
    scrollFrame = 0;
    document.querySelectorAll("[data-vh-navbar]").forEach((navbar) => {
      navbar.classList.toggle("is-scrolled", window.scrollY > 12);
    });
  };
  const onScroll = () => {
    if (!scrollFrame) {
      scrollFrame = window.requestAnimationFrame(syncNavbar);
    }
  };

  const routeObserver = new MutationObserver(() => {
    bindReveals();
    syncNavbar();
  });
  routeObserver.observe(document.body, { childList: true, subtree: true });
  window.addEventListener("scroll", onScroll, { passive: true });
  bindReveals();
  syncNavbar();
};
