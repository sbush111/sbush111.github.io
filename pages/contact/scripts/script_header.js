const navMenuToggleButton = document.getElementById("nav-menu-toggle");
const navMenu = document.getElementById("nav-menu");

navMenuToggleButton.addEventListener("click", toggleMenu);
function toggleMenu(event) {
    event.preventDefault();
    navMenu.classList.toggle('hide');
    navMenu.classList.toggle('show');
}

document.addEventListener("click", clickOutOfMenu);
function clickOutOfMenu(event) {

    if(!navMenu.contains(event.target) && event.target != navMenuToggleButton) {
        navMenu.classList.remove('show');
        navMenu.classList.add('hide');
    }

}