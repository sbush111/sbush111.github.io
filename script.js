const navMenuToggleButton = document.getElementById("nav-menu-toggle");
const navMenu = document.getElementById("nav-menu");
navMenuToggleButton.addEventListener("click", toggleMenu);

function toggleMenu(event) {
    console.log("toggleMenu() called.");
    event.preventDefault();
    if(navMenu.style.display == 'block') {
        navMenu.style.display = 'none';
    } else {
        navMenu.style.display = 'block';
    }
}

console.log("Script loaded.")