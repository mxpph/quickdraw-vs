import * as React from "react";
import { P5CanvasInstance, ReactP5Wrapper } from "@p5-wrapper/react"


// const { height, width } = x/x/();
function sketch(p5: P5CanvasInstance) {


    p5.setup = () => {
        // p5.createCanvas(window.innerWidth * 0.9,window.innerHeight * 0.9)
        p5.createCanvas(500,500,p5.WEBGL)
        p5.background(240)
        // fetchDrawing()
    }

    p5.draw = () => {
        p5.background(240);
        p5.normalMaterial();
        p5.push();
        p5.rotateZ(p5.frameCount * 0.01);
        p5.rotateX(p5.frameCount * 0.01);
        p5.rotateY(p5.frameCount * 0.01);
        p5.plane(100);
        p5.pop();
    };
}

// function newGame() {
//     fetch("/create_game",{method: "GET"})
//     .then(res => res.json())
//     .then(data => {
//         console.log(`New game with ID: ${data.game_id}`)
//     })
// }

// function fetchDrawing() {
//     fetch("/get",{method: "GET"})
//     .then(res => res.json())
//     .then(data => {
//         drawThing(data)
//         console.log(data.drawing[data.drawing.length-1][2][(data.drawing[data.drawing.length-1][2]).length-1])
//         setTimeout(fetchDrawing,data.drawing[data.drawing.length-1][2][(data.drawing[data.drawing.length-1][2]).length-1])
//     })
// }

// let curdrawing, curx, cury
// let curstroke = 0
// let curpoint = 0

// function draw() {
//     if (curdrawing) {
//         console.log(curdrawing)
//         // noFill()
//         stroke(0)
//         strokeWeight(3)
//         let x = curdrawing[curstroke][0][curpoint]
//         let y = curdrawing[curstroke][1][curpoint]
//         point(x,y)
//         curpoint+=1
//         if (curpoint >= curdrawing[curstroke][0].length) {
//             curstroke+=1
//             curpoint = 0
//         }
//         if (curstroke >= curdrawing.length) {
//             curdrawing = undefined
//         }
//         // beginShape()
//         // for (let i = 0; i < data.drawing.length; i++) {
//         //     xs = curdrawing[i][0]
//         //     ys = curdrawing[i][1]
//         //     t = curdrawing[i][2]
//         //     for (let j = 0; j < xs.length; j++) {
//         //         console.log(xs[j])
//         //         vertex(xs[j],ys[j])
//         //     }
//         // }
//         // endShape()
//     }
// }

// function drawThing(data) {
//     background(240)
//     curstroke = 0
//     curpoint = 0
//     curdrawing = data.drawing
// }


export default function DrawingCanvas() {
    return (
        <ReactP5Wrapper sketch={sketch} />
    );
}