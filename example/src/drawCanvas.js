export class DrawCanvas {
    constructor(name) {
        this.canvas = document.getElementById(name);
        let canvasWidth = this.canvas.width;
        let canvasHeight = this.canvas.height;
        this.ctx = this.canvas.getContext("2d");
        this.data = this.ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    }
    drawCicle(x, y, color, size) {
        this.ctx.strokeStyle = color;
        this.ctx.beginPath();
        this.ctx.arc(x - 9, y - 9, size ? size : 5, 0, 2 * Math.PI, true);
        this.ctx.stroke();
        let canvasWidth = this.canvas.width;
        let canvasHeight = this.canvas.height;
        this.data = this.ctx.getImageData(0, 0, canvasWidth, canvasHeight);
    }
    drawImageByData(dataVector) {
        this.ctx.clearRect(0, 0, 500, 500);
        this.ctx.beginPath();
        for (var ix = 0; ix < this.canvas.width / 5; ix++) {
            for (var iy = 0; iy < this.canvas.height / 5; iy++) {
                let vector = dataVector[(ix + (iy * 500 / 5))];
                this.ctx.fillStyle = 'rgb(' + vector.r + ', ' + vector.g + ', ' + vector.b + ')';
                this.ctx.fillRect(ix * 5, iy * 5, 5, 5);
            }
        }
    }
    updateCanvas() {
        this.ctx.putImageData(this.data, 0, 0);
    }
}
