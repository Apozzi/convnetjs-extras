export class Keyboard {
    constructor() {
        this.keysData = new Array(300);
        this.defaultKey = {
            key: "empty",
            keyCounter: 0,
            pressed: true
        };
    }
    onKeydown(keyEvent) {
        let preObj = this.keysData[keyEvent.keyCode];
        if (preObj) {
            preObj.keyCounter++;
            preObj.pressed = true;
            this.keysData[keyEvent.keyCode] = preObj;
        }
        else {
            let keyObj = Object.assign({}, this.defaultKey);
            keyObj.key = keyEvent.key;
            keyObj.keyCounter++;
            this.keysData[keyEvent.keyCode] = keyObj;
        }
    }
    onKeyup(keyEvent) {
        let preObj = this.keysData[keyEvent.keyCode];
        if (preObj) {
            preObj.pressed = false;
            this.keysData[keyEvent.keyCode] = preObj;
        }
    }
    isKeyPressed(code) {
        return !!this.keysData[code].pressed;
    }
    isKeyOdd(code) {
        return this.keysData[code] ? this.keysData[code].keyCounter % 2 === 1 : false;
    }
}
