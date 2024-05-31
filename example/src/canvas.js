import { DrawCanvas } from './drawCanvas.js';
import { Keyboard } from './keyboard.js';
export class Canvas {
    init() {
        this.canvas = new DrawCanvas("area");
        this.keyboard = new Keyboard();
        this.dataForLearning = [];
        this.createNeuralNet();
    }
    createNeuralNet() {
        let layer_defs = [];
        layer_defs.push({ type: 'input', out_sx: 1, out_sy: 1, out_depth: 2 });
        layer_defs.push({ type: 'fc', num_neurons: 16, activation: 'leaky_relu' });
        layer_defs.push({ type: 'fc', num_neurons: 7, activation: 'leaky_relu' });
        layer_defs.push({ type: 'fc', num_neurons: 7, activation: 'leaky_relu' });
        layer_defs.push({ type: 'fc', num_neurons: 7, activation: 'leaky_relu' });
        layer_defs.push({ type: 'softmax', num_classes: 2 });
        this.net = new convnetjs.Net();
        this.net.makeLayers(layer_defs);
        this.trainer = new convnetjs.SGDTrainer(this.net, { learning_rate: 0.01, momentum: 0.1, batch_size: 10, l2_decay: 0.001 });
        this.drawImageWithNN();
        setInterval(() => {
            this.trainNN();
            this.drawImageWithNN();
        }, 100);
    }
    trainNN() {
        let traindata = new convnetjs.Vol(1, 1, 2);
        for (let index = 0; index < 20; index++) {
            this.dataForLearning.forEach(data => {
                traindata.w = [(data.x) / 10, (data.y) / 10];
                let trainer = this.trainer.train(traindata, data.v);
                console.log('loss = ' + trainer.loss);
            });
        }
    }
    drawImageWithNN() {
        let dataVector = [];
        for (let index = 0; index < 500 / 5; index++) {
            for (let indexy = 0; indexy < 500 / 5; indexy++) {
                let x = new convnetjs.Vol([(indexy) / 10, (index) / 10]);
                let prob = this.net.forward(x);
                dataVector.push({ r: 0, g: 100*prob.w[0], b: 100*prob.w[1] });
            }
        }
        this.canvas.drawImageByData(dataVector);
        this.dataForLearning.forEach(data => {
            if (data.v === 1) {
                this.canvas.drawCicle(data.x * 5, data.y * 5, '#0000ff');
            }
            else {
                this.canvas.drawCicle(data.x * 5, data.y * 5, '#00ff00');
            }
        });
    }
    onClickCanvas(event) {
        if (this.keyboard.isKeyOdd(67)) {
            this.dataForLearning.push({
                x: event.clientX / 5,
                y: event.clientY / 5,
                v: 0
            });
        }
        else {
            this.dataForLearning.push({
                x: event.clientX / 5,
                y: event.clientY / 5,
                v: 1
            });
        }
    }
    keyboardDown(event) {
        this.keyboard.onKeydown(event);
    }
    keyboardUp(event) {
        this.keyboard.onKeyup(event);
    }
}
