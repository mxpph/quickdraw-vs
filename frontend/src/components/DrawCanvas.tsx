import React, { useRef, useState } from "react";
import { Stage, Layer, Line } from "react-konva";

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

  const normalizeStrokes = (strokes: Point[][]): [number[], number[]][] => {
    const allPoints = strokes.flat();
    const minX = Math.min(...allPoints.map((p) => p.x));
    const minY = Math.min(...allPoints.map((p) => p.y));
    const maxX = Math.max(...allPoints.map((p) => p.x));
    const maxY = Math.max(...allPoints.map((p) => p.y));

    const width = maxX - minX;
    const height = maxY - minY;
    const maxDim = Math.max(width, height);
    const scale = 255 / maxDim;

    return strokes.map((stroke) => {
      const xCoords = stroke.map((p) => Math.round((p.x - minX) * scale));
      const yCoords = stroke.map((p) => Math.round((p.y - minY) * scale));
      return [xCoords, yCoords];
    });
  };

  const rasterizeStrokes = (
    normalizedStrokes: [number[], number[]][],
    side: number = 28,
    line_diameter: number = 16,
    padding: number = 16
  ): number[] => {
    const original_side = 256;
    const bg_color = [0, 0, 0];
    const fg_color = [1, 1, 1];

    const canvas = document.createElement("canvas");
    canvas.width = side;
    canvas.height = side;
    const ctx = canvas.getContext("2d");

    if (!ctx) {
      throw new Error("Could not get 2D context from canvas");
    }

    // Set up context to match Cairo settings
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = line_diameter;

    // Scale and translate
    const total_padding = padding * 2 + line_diameter;
    const new_scale = side / (original_side + total_padding);
    ctx.scale(new_scale, new_scale);
    ctx.translate(total_padding / 2, total_padding / 2);

    // Clear background
    ctx.fillStyle = `rgb(${bg_color[0] * 255},${bg_color[1] * 255},${
      bg_color[2] * 255
    })`;
    ctx.fillRect(0, 0, side / new_scale, side / new_scale);

    // Calculate bounding box and offset
    const allPoints = normalizedStrokes.flatMap((stroke) =>
      stroke[0].map((x, i) => [x, stroke[1][i]])
    );
    const bbox = [
      Math.max(...allPoints.map((p) => p[0])),
      Math.max(...allPoints.map((p) => p[1])),
    ];
    const offset = [
      (original_side - bbox[0]) / 2,
      (original_side - bbox[1]) / 2,
    ];

    // Draw strokes
    ctx.strokeStyle = `rgb(${fg_color[0] * 255},${fg_color[1] * 255},${
      fg_color[2] * 255
    })`;
    for (let i = 0; i < normalizedStrokes.length; i++) {
      const [xv, yv] = normalizedStrokes[i];
      ctx.beginPath();
      ctx.moveTo(xv[0] + offset[0], yv[0] + offset[1]);
      for (let j = 1; j < xv.length; j++) {
        ctx.lineTo(xv[j] + offset[0], yv[j] + offset[1]);
      }
      ctx.stroke();
    }

    // Get image data
    const imageData = ctx.getImageData(0, 0, side, side);

    // Convert to 1D array of grayscale values (0-255)
    const rasterImage = new Array(side * side);
    for (let i = 0; i < imageData.data.length; i += 4) {
      rasterImage[i / 4] = imageData.data[i]; // Invert colors (black on white background)
    }

    return rasterImage;
  };

  // const rasterizeStrokes = (normalizedStrokes: [number[], number[]][], size: number = 28): number[] => {
  //   const originalSize = 256;
  //   const padding = 16; // Adjust if needed
  //   const lineWidth = 16; // Adjust based on your needs

  //   const totalPadding = padding * 2 + lineWidth;
  //   const scale = size / (originalSize + totalPadding);

  //   const canvas = document.createElement('canvas');
  //   canvas.width = size;
  //   canvas.height = size;
  //   const ctx = canvas.getContext('2d');

  //   if (!ctx) {
  //     throw new Error('Could not get 2D context from canvas');
  //   }

  //   // Clear background
  //   ctx.fillStyle = 'white';
  //   ctx.fillRect(0, 0, size, size);

  //   // Set up stroke style
  //   ctx.strokeStyle = 'black';
  //   ctx.lineWidth = lineWidth * scale;
  //   ctx.lineCap = 'round';
  //   ctx.lineJoin = 'round';

  //   ctx.translate(totalPadding / 2 * scale, totalPadding / 2 * scale);

  //   // Draw strokes
  //   for (const [xCoords, yCoords] of normalizedStrokes) {
  //     ctx.beginPath();
  //     ctx.moveTo(xCoords[0] * scale, yCoords[0] * scale);
  //     for (let i = 1; i < xCoords.length; i++) {
  //       ctx.lineTo(xCoords[i] * scale, yCoords[i] * scale);
  //     }
  //     ctx.stroke();
  //   }

  //   // Get image data
  //   const imageData = ctx.getImageData(0, 0, size, size);

  //   // Convert to 1D array of grayscale values (0-255)
  //   const rasterImage = new Array(size * size);
  //   for (let i = 0; i < imageData.data.length; i += 4) {
  //     rasterImage[i / 4] = 255 - imageData.data[i]; // Invert colors (black on white background)
  //   }

  //   return rasterImage;
  // };

  const handleRasterize = () => {
    const normalizedStrokes = normalizeStrokes(lines);
    const rasterArray = rasterizeStrokes(normalizedStrokes);
    console.log("Rasterized array:", rasterArray);

    // Visualize the rasterized image (for debugging)
    const canvas = document.createElement("canvas");
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      const imageData = ctx.createImageData(28, 28);
      for (let i = 0; i < rasterArray.length; i++) {
        const value = rasterArray[i];
        imageData.data[i * 4] = value;
        imageData.data[i * 4 + 1] = value;
        imageData.data[i * 4 + 2] = value;
        imageData.data[i * 4 + 3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
      document.body.appendChild(canvas);
    }
  };

  const handleExportToSVG = () => {
    const svgHeader = `<svg width="${window.innerWidth * 0.9}" height="${
      window.innerHeight * 0.9
    }" xmlns="http://www.w3.org/2000/svg">`;
    const svgFooter = "</svg>";

    const svgPaths = lines
      .map((line, index) => {
        const pathData = line.map((p) => `${p.x},${p.y}`).join(" ");
        return `<path d="M ${pathData}" stroke="black" stroke-width="7" fill="none" />`;
      })
      .join("");

    const svgContent = svgHeader + svgPaths + svgFooter;

    // Log SVG content to the console
    console.log("SVG Content:", svgContent);
  };

  return (
    <div>
      <Stage
        width={window.innerWidth * 0.9}
        height={window.innerHeight * 0.9}
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
      <button
        className="my-2 mx-1 rounded-xl shadow shadow-neutral-400 px-2 bg-neutral-100 py-1"
        onClick={handleRasterize}
      >
        Rasterize Drawing
      </button>
      <button
        className="my-2 mx-1 rounded-xl shadow shadow-neutral-400 px-2 bg-neutral-100 py-1"
        onClick={handleExportToSVG}
      >
        Export to SVG
      </button>
    </div>
  );
};

export default DrawCanvas;
