const form = document.querySelector("form");
const textarea = document.querySelector("textarea");
const button = document.querySelector("button");

form.addEventListener("submit", function (event) {
    if (!textarea.value.trim()) {
        event.preventDefault();
        alert("Please enter a news headline.");
        return;
    }

    button.disabled = true;
    button.textContent = "Analyzing headline...";
});
