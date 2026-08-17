(function () {
    const storageKey = 'scholarship-theme';

    function isLightTheme() {
        return document.body.classList.contains('light-mode');
    }

    function updateToggle(toggle) {
        const light = isLightTheme();
        toggle.innerHTML = light
            ? '<i class="fas fa-sun" aria-hidden="true"></i>'
            : '<i class="fas fa-moon" aria-hidden="true"></i>';
        toggle.title = light ? 'Dark mode u beddel' : 'Light mode u beddel';
        toggle.setAttribute('aria-label', toggle.title);
    }

    function setTheme(theme) {
        document.body.classList.toggle('light-mode', theme === 'light');
        document.documentElement.dataset.theme = theme;
        localStorage.setItem(storageKey, theme);
        document.querySelectorAll('[data-theme-toggle]').forEach(updateToggle);
        document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: theme } }));
    }

    function initializeTheme() {
        const savedTheme = localStorage.getItem(storageKey) || 'dark';
        document.body.classList.toggle('light-mode', savedTheme === 'light');
        document.documentElement.dataset.theme = savedTheme;

        let toggle = document.getElementById('theme-toggle');
        if (!toggle) {
            toggle = document.createElement('button');
            toggle.type = 'button';
            toggle.id = 'theme-toggle';
            document.body.appendChild(toggle);
        }
        toggle.classList.add('app-theme-toggle');
        toggle.setAttribute('data-theme-toggle', '');
        updateToggle(toggle);
        toggle.addEventListener('click', function () {
            setTheme(isLightTheme() ? 'dark' : 'light');
        });
        document.dispatchEvent(new CustomEvent('themechange', { detail: { theme: savedTheme } }));
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeTheme);
    } else {
        initializeTheme();
    }

    window.setScholarshipTheme = setTheme;
})();
