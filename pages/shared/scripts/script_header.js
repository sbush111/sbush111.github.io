const body = document.querySelector('body');

const header_html = `
        <header id="page-header">
            <nav>
                <a id="nav-home" href="/pages/homepage">Sean Bush</a>
            </nav>
            <nav id="nav-bar">
                <a href="/pages/about">About</a>
                <a href="/pages/blog">Blog</a>
                <a href="/pages/portfolio">Portfolio</a>
                <a href="/pages/resume">Resume</a>
                <a href="/pages/contact">Contact</a>
            </nav>
            <div id="button-container">
                <button id="nav-menu-toggle">≡</button>
            </div>
            <menu id="nav-menu" class="hide">
                <li><a href="/pages/about">About</a></li>
                <li><a href="/pages/blog">Blog</a></li>
                <li><a href="/pages/portfolio">Portfolio</a></li>
                <li><a href="/pages/resume">Resume</a></li>
                <li><a href="/pages/contact">Contact</a></li>
            </menu>
        </header>
`

body.insertAdjacentHTML('afterbegin', header_html);

const navMenu = document.getElementById("nav-menu");
const navMenuToggleButton = document.getElementById("nav-menu-toggle");

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