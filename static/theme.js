// Global Theme Management for Student Performance Prediction System
(function () {
    const savedTheme = localStorage.getItem("app_theme") || 
        (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    document.documentElement.setAttribute("data-theme", savedTheme);
})();

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    
    document.documentElement.setAttribute("data-theme", newTheme);
    localStorage.setItem("app_theme", newTheme);
    updateThemeButtons();
}

function updateThemeButtons() {
    const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
    const buttons = document.querySelectorAll(".theme-toggle-btn");
    
    buttons.forEach((btn) => {
        if (currentTheme === "dark") {
            btn.innerHTML = "☀️ Light";
            btn.setAttribute("aria-label", "Switch to light mode");
            btn.classList.remove("btn-outline-light");
            btn.classList.add("btn-light");
        } else {
            btn.innerHTML = "🌙 Dark";
            btn.setAttribute("aria-label", "Switch to dark mode");
            btn.classList.remove("btn-light");
            btn.classList.add("btn-outline-light");
        }
    });
}

// Ensure button label is synchronized once DOM is ready
if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", updateThemeButtons);
} else {
    updateThemeButtons();
}
