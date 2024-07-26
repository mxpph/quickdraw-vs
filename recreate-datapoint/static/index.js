

function setup() {

    canvas = createCanvas(windowWidth * 0.9,windowHeight * 0.9)
    canvas.parent("canvasdiv")
    background(240)

    fetch("/get",{method: "GET"})
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