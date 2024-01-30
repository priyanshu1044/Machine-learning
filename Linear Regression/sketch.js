let x_vals = [];
let y_vals = [];

let m, c;

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);


function setup() {
    createCanvas(600, 600);
    m = tf.variable(tf.scalar(random(1)));
    c = tf.variable(tf.scalar(random(1)));

}

function loss(pred, labels) {
    // (gess-y)^2
    return pred.sub(labels).square().mean();
}

function predict(x) {
    const xs = tf.tensor1d(x);
    // y = mx + c
    const ys = xs.mul(m).add(c);
    return ys;
}

  
function mousePressed() {
    let x = map(mouseX, 0, width, 0, 1);
    let y = map(mouseY, 0, height, 1, 0);
    x_vals.push(x);
    y_vals.push(y);
}

function draw() {
    tf.tidy(() => {
        if(x_vals.length > 0){
            optimizer.minimize(() => {
                return loss(predict(x_vals), tf.tensor1d(y_vals));
            })
        }
        
        background(0);
        stroke(255);
        strokeWeight(8);
        for (let i = 0; i < x_vals.length; i++) {
            let px = map(x_vals[i], 0, 1, 0, width);
            let py = map(y_vals[i], 0, 1, height, 0);
            point(px, py);
        }

        let xs = [0, 1];

        let ys=predict(xs);
        let x1=map(xs[0],0,1,0,width);
        let x2=map(xs[1],0,1,0,width);

        let lineY=ys.dataSync();
        let y1=map(lineY[0],0,1,height,0);
        let y2=map(lineY[1],0,1,height,0);

        strokeWeight(2);
        line(x1,y1,x2,y2);
    });
}