
document.addEventListener("DOMContentLoaded", function () {
    const boxesInput = document.getElementById("boxes_shipped");

    if (boxesInput) {
        boxesInput.addEventListener("input", function () {
            const boxes = Number(this.value);

            if (this.value !== "" && boxes < 0) {
                this.setCustomValidity(
                    "Boxes shipped cannot be negative."
                );
            } else {
                this.setCustomValidity("");
            }
        });
    }
});
