import React, {
  useRef,
  useState,
  useEffect,
  useImperativeHandle,
  forwardRef,
} from "react";
import { Stage, Layer, Line } from "react-konva";
import { InferenceSession, Tensor } from "onnxruntime-web";
import { clear } from "console";
import { clearInterval } from "timers";

interface AnimateProps {
  children: React.ReactNode;
  on: string | undefined;
}

function AnimateText({ children, on }: AnimateProps) {
  return on === undefined ? (
    <p>{children}</p>
  ) : (
    <p className="text-xl font-semibold animate-color-fade" key={on}>
      {children}
    </p>
  );
}

interface Point {
  x: number;
  y: number;
}

interface DrawCanvasProps {
  dataPass: (data: string) => void;
  onParentClearCanvas: () => void;
  clearCanvas: boolean;
}

const DrawCanvas: React.FC<DrawCanvasProps> = ({
  dataPass,
  onParentClearCanvas,
  clearCanvas,
}) => {
  const [prediction, setPrediction] = useState("?");
  const [lines, setLines] = useState<Point[][]>([]);
  const [confidence, setConfidence] = useState(0);
  const isDrawing = useRef(false);
  const session = useRef<InferenceSession | null>(null);

  const predDebounce = 400;

  const modelCategories = [
    'airplane',
    'angel',
    'ant',
    'anvil',
    'apple',
    'banana',
    'basketball',
    'broom',
    'camera',
    'dog',
    'dresser',
    'hammer',
    'hat',
    'hexagon',
    'paper clip',
    'pencil'
  ];


  useEffect(() => {
    var evalTimer : NodeJS.Timeout
    (async () => {
      try {
        session.current = await InferenceSession.create("CNN_cat16_v6-0_large_gputrain.onnx");
        evalTimer = setInterval(handleEvaluate,predDebounce);
        console.log("evaltimer!",evalTimer)
      } catch (error) {
        // TODO: Handle this error properly
        console.error("Failed to load model", error);
      }
    })();

    return () => {
      clearInterval(evalTimer)
      session.current?.release();
    };
  }, []);

  const handleMouseDown = (e: any) => {
    isDrawing.current = true;
    const pos = e.target.getStage().getPointerPosition();
    setLines([...lines, [pos]]);
  };

  const handleMouseMove = (e: any) => {
    if (!isDrawing.current) return;

    const stage = e.target.getStage();
    const point = stage.getPointerPosition();
    if (lines[lines.length - 1] !== undefined) {
      const lastLine = lines[lines.length - 1].concat([point]);
      setLines(lines.slice(0, -1).concat([lastLine]));
    }
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

    // const rasterImage: number[][][][] = new Array(1).fill(null).map(() =>
    //   new Array(1).fill(null).map(() =>
    //     new Array(side).fill(null).map(() =>
    //       new Array(side).fill(0)
    //     )
    //   )
    // );
    
    // for (let y = 0; y < side; y++) {
    //   for (let x = 0; x < side; x++) {
    //     const i = (y * side + x) * 4;
    //     rasterImage[0][0][y][x] = imageData.data[i] / 255; // Normalize to 0-1
    //   }
    // }

    

    return rasterImage;
  };

  const softmax = (arr: Float32Array): Float32Array => {
    const max = Math.max(...arr);
    const exps = arr.map((x) => Math.exp(x - max)); // Subtract max for numerical stability
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((x) => x / sum);
  };

  const argMax = (arr: Float32Array): number => arr.indexOf(Math.max(...arr));

  async function ONNX(input: number[]) {
    if (session.current === null) {
      console.error(
        "Attempted to run inference while InferenceSession is null"
      );
      return;
    }
    try {
      const flattenedInput = input.flat(3);
      const tensor = new Tensor("float32", new Float32Array(flattenedInput), [1, 1, 28, 28]);

      // const tensor = new Tensor("float32", new Float32Array(input), [1, 784]);

      const inputMap = { input: tensor };

      const outputMap = await session.current.run(inputMap);

      const output = outputMap["output"].data as Float32Array;

      // console.log(output);
      return output;
    } catch (error) {
      console.error("Error running ONNX model:", error);
    }
  }

  const handleRasterize = () => {
    const normalizedStrokes = normalizeStrokes(lines);
    const rasterArray = rasterizeStrokes(normalizedStrokes);
    return rasterArray;
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

    return svgContent;
  };

  const handleEvaluate = () => {
    const normalizedStrokes = normalizeStrokes(lines);
    const rasterArray = rasterizeStrokes(normalizedStrokes);

    ONNX(rasterArray).then((res) => {
      // console.log(res);
      res = res as Float32Array;
      let i = argMax(res);
      setPrediction(modelCategories[i]);
      let prob = softmax(res)[i];
      let probPercent = Math.floor(prob * 1000) / 10;
      setConfidence(probPercent);
      if (probPercent > 70) {
        dataPass(prediction);
      } 
    });
  };

  useEffect(() => {
    // effect to check if clearCanvas is true
    if (clearCanvas) {
      setLines([]);
      onParentClearCanvas(); // call the callback function to reset the state in parent component
    }
  }, [clearCanvas, onParentClearCanvas]);

  const clearDrawing = () => {
    setLines([]);
  };

  return (
    <div>
      <div className="grid grid-cols-3 place-items-center">
        <div className="grid place-items-center">
          {prediction && (
            <AnimateText on={prediction}>
              I guess... {confidence > 70 ? prediction : "not sure"}!
            </AnimateText>
          )}
          {/* confidence > 70 ? (
            <p className="text-lg font-medium text-green-400">
              Confidence (dev): {confidence + "%"}
            </p>
          ) : (
            <p className="text-lg font-medium">
              Confidence (dev): {confidence + "%"}
            </p>
          )*/}
        </div>
        <button
          className="btn col-span-1 mt-1 btn-primary btn-outline"
          onClick={clearDrawing}
        >
          Clear Canvas
        </button>
      </div>
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
      <div className="flex justify-center items-center align-middle">
        {/* <button
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
        <button
          className="my-2 mx-1 rounded-xl shadow shadow-neutral-400 px-2 bg-neutral-100 py-1"
          onClick={handleEvaluate}
        >
          Evaluate drawing
        </button> */}
      </div>
    </div>
  );
};

export default DrawCanvas;
