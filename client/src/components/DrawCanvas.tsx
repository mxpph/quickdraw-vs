import React, { useRef, useState } from 'react';
import { Stage, Layer, Line } from 'react-konva';

interface Point {
  x: number;
  y: number;
}

const DrawCanvas: React.FC = () => {
  const [lines, setLines] = useState<Point[][]>([]);
  const isDrawing = useRef(false);

  const handleMouseDown = (e: any) => {
    isDrawing.current = true;
    const pos = e.target.getStage().getPointerPosition();
    setLines([...lines, [pos]]);
  };

  const handleMouseMove = (e: any) => {
    if (!isDrawing.current) return;

    const stage = e.target.getStage();
    const point = stage.getPointerPosition();
    const lastLine = lines[lines.length - 1].concat([point]);
    setLines(lines.slice(0, -1).concat([lastLine]));
  };

  const handleMouseUp = () => {
    isDrawing.current = false;
  };

  const toSVG = () => {
    const svgWidth = window.innerWidth*0.9;
    const svgHeight = window.innerHeight*0.9;

    let svgContent = `<svg width="${svgWidth}" height="${svgHeight}" xmlns="http://www.w3.org/2000/svg">`;

    lines.forEach((line) => {
      if (line.length < 2) return;

      let pathData = `M ${line[0].x} ${line[0].y}`;
      for (let i = 1; i < line.length; i++) {
        pathData += ` L ${line[i].x} ${line[i].y}`;
      }

      svgContent += `<path d="${pathData}" fill="none" stroke="black" stroke-width="2" />`;
    });

    svgContent += '</svg>';
    return svgContent;
  };

  return (
    <div>
      <Stage
        width={window.innerWidth*0.9}
        height={window.innerHeight*0.9}
        onMouseDown={handleMouseDown}
        onMousemove={handleMouseMove}
        onMouseup={handleMouseUp}
      >
        <Layer>
          {lines.map((line, i) => (
            <Line
              key={i}
              points={line.flatMap((p) => [p.x, p.y])}
              stroke="black"
              strokeWidth={7}
              tension={0.5}
              lineCap="round"
              lineJoin="round"
            />
          ))}
        </Layer>
      </Stage>
      <button onClick={() => console.log(toSVG())}>Convert to SVG</button>
    </div>
  );
};

export default DrawCanvas;





// export default function App() {
//   return (
//     <Stage width={window.innerWidth * 0.9} height={window.innerHeight * 0.9}>
//       <Layer>
//         <Rect width={100} height={100} fill="red" />
//       </Layer>
//     </Stage>
//   );
// }























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
