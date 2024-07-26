

function draw() {}function setup() {
    createCanvas(1000,1000)
    background(220)

    fetch("/getFirst",{method: "GET"})
    .then(res => res.json())
    .then(data => {
        drawThing(data)
    })
}

function drawThing(data) {
    noFill()
    stroke(0)
    strokeWeight(3)
    beginShape()
    console.log(data.drawing)
    for (let i = 0; i < data.drawing.length; i++) {
        xs = data.drawing[i][0]
        ys = data.drawing[i][1]
        t = data.drawing[i][2]
        for (let j = 0; j < xs.length; j++) {
            console.log(xs[j])
            vertex(xs[j],ys[j])
        }
    }
    endShape()
}