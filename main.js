const IMAGE_SIZE = 224;
const CONV_SIZE = 112;
const THRESHOLD = 70.0;
const MODELPATH = "./christmas_clf.onnx";
const CLASSNAMES = ["christmas tree", "grinch", "reindeer", "santa", "snowman"];
const examplePath = ["example_image/ex(1).png", "example_image/ex(1).jpg", "example_image/ex(2).jpg", "example_image/ex(3).jpg", "example_image/ex(2).png"]
const containerDiv = document.getElementById("container");
const imageDisplay = document.getElementById("imageDisplay");
const predictionDiv = document.getElementById("prediction");
const predictProbas = document.getElementById("predictProbas");
const exampleA = document.getElementById("example");
const loadDiv = document.getElementsByClassName('load');
const imageContainer = document.getElementById("imageContainer")
const loadCir = document.getElementById("lds-facebook")
let pathIdx = 0;

async function loadModel() {
  try {
    const session = await ort.InferenceSession.create(MODELPATH);
    return session;
  } catch (error) {
    console.error("Error loading the model:", error);
    throw error;
  }
}

// Call the loadModel function once during web page load
(async () => {
  try {
    session = await loadModel();
    await exampleDisplay(pathIdx);
  } catch (error) {
    // Handle the error if needed
    console.error("Error setting up the model:", error);
  }
})();

function softmax(scores) {
  const maxScore = Math.max(...scores);
  const expScores = scores.map((score) => Math.exp(score - maxScore));
  const sumExpScores = expScores.reduce((sum, expScore) => sum + expScore, 0);
  const probabilities = expScores.map((expScore) => expScore / sumExpScores);
  return probabilities;
}

function mapValueToColor(value) {
  // Use a grayscale colormap for simplicity
  const grayscaleValue = Math.floor(value * 255);
  const colorMap = Array.from({ length: 7 }, (_, index) => Math.floor((255 / 7) * (index + 1)));
  if (grayscaleValue < colorMap[0]) {
    return `rgb(4, 0, ${grayscaleValue})`;
  } else if (grayscaleValue < colorMap[1]) {
    return `rgb(40, 1, ${grayscaleValue})`;
  } else if (grayscaleValue < colorMap[2]) {
    return `rgb(120, 4, ${grayscaleValue})`;
  } else if (grayscaleValue < colorMap[3]) {
    return `rgb(${grayscaleValue}, 45, 87)`;
  } else if (grayscaleValue < colorMap[4]) {
    return `rgb(${grayscaleValue}, 77, 61)`;
  } else if (grayscaleValue < colorMap[5]) {
    return `rgb(215, ${grayscaleValue}, 30)`;
  } else if (grayscaleValue < colorMap[6]) {
    return `rgb(210, ${grayscaleValue}, 8)`;
  }
}

async function exampleDisplay() {
  displayLoad("block")

  const path = examplePath[pathIdx%5];
  pathIdx += 1;
  const resizedImageData = await loadImageFromPath(path);
  const imageTensor = await imageDataToTensor(resizedImageData, [
    1,
    3,
    IMAGE_SIZE,
    IMAGE_SIZE,
  ]);

  const feeds = { input: imageTensor };
  const outputMap = await session.run(feeds);
  const logits = outputMap.probas.data;
  const conv1 = outputMap.conv1.data;
  const conv2 = outputMap.conv2.data;
  const probas = softmax(logits);
  const percent = [];
  for (prob of probas) {
    percent.push((prob * 100).toFixed(2));
  }
  const conv1r = reshapeArray(conv1, [8, 224, 224]);
  const conv2r = reshapeArray(conv2, [8, 110, 110]);

  for (let i = 0; i < conv1r.length; i++) {
    const canvasId = `conv1r${i}`;
    createCanvas(canvasId);
    resizeCanvasAndDrawImage(canvasId, conv1r[i], true);
  }

  for (let i = 0; i < conv2r.length; i++) {
    const canvasId = `conv2r${i}`;
    createCanvas(canvasId);
    resizeCanvasAndDrawImage(canvasId, conv2r[i], false);
  }

  displayLoad("none")
  displayImage(path)
    const idx = probas.map((_, index) => index);
  // [0, 1, 4, 3] idx order max-min -> [2, 3, 1, 0]
  const sortedIdx = idx.sort((a, b) => probas[b] - probas[a]);
  const prediction = [];
  for (let i=0; i<sortedIdx.length; i++) {
    prediction.push([CLASSNAMES[sortedIdx[i]], percent[sortedIdx[i]]])
  }

  if (Math.max(...percent) > THRESHOLD) {
    if (prediction[0][0] == "christmas tree"){
      predictionDiv.innerText = prediction[0][0].split("christmas")[1]; // christmas tree -> tree (too long)
    } else {
      predictionDiv.innerText = prediction[0][0];
    }
  } else {
    predictionDiv.innerText = "Not sure";
  }

  let resultHtml = ""
    for (let i=0; i<prediction.length; i++) {
      resultHtml +=  `<li>${prediction[i][0]}<span>${prediction[i][1]}%</span></li>`;
    }
  predictProbas.innerHTML = resultHtml;
}

function createCanvas(id) {
  const canvas = document.createElement("canvas");
  canvas.id = id;
  canvas.width = CONV_SIZE;
  canvas.height = CONV_SIZE;
  containerDiv.appendChild(canvas);
}

function resizeCanvasAndDrawImage(canvasId, dataArray, rescale) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");

  for (let y = 0; y < dataArray.length; y++) {
    for (let x = 0; x < dataArray[y].length; x++) {
      const color = mapValueToColor(dataArray[y][x]);
      ctx.fillStyle = color;

      if (rescale == true) {
        const scaledX = (x * CONV_SIZE) / dataArray[y].length;
        const scaledY = (y * CONV_SIZE) / dataArray.length;
        const scaledWidth = CONV_SIZE / dataArray[y].length;
        const scaledHeight = CONV_SIZE / dataArray.length;
        ctx.fillRect(scaledX, scaledY, scaledWidth, scaledHeight);
      } else {
        ctx.fillRect(x, y, 1, 1);
      }
    }
  }
}

async function uploadImage() {
  displayLoad("block")

  const fileInput = document.getElementById("fileInput");
  const file = fileInput.files[0];

  if (file) {
    const imageData = await readImageData(file);
    const resizedImageData = await loadImageFromDataUrl(imageData);
    const imageTensor = await imageDataToTensor(resizedImageData, [
      1,
      3,
      IMAGE_SIZE,
      IMAGE_SIZE,
    ]);

    const feeds = { input: imageTensor };
    const outputMap = await session.run(feeds);
    const logits = outputMap.probas.data;
    const conv1 = outputMap.conv1.data;
    const conv2 = outputMap.conv2.data;
    const probas = softmax(logits);
    const percent = [];
    for (prob of probas) {
      percent.push((prob * 100).toFixed(2));
    }
    const conv1r = reshapeArray(conv1, [8, 224, 224]);
    const conv2r = reshapeArray(conv2, [8, 110, 110]);

    for (let i = 0; i < conv1r.length; i++) {
      const canvasId = `conv1r${i}`;
      createCanvas(canvasId);
      resizeCanvasAndDrawImage(canvasId, conv1r[i], true);
    }

    for (let i = 0; i < conv2r.length; i++) {
      const canvasId = `conv2r${i}`;
      createCanvas(canvasId);
      resizeCanvasAndDrawImage(canvasId, conv2r[i], false);
    }

  displayLoad("none")
	displayImage(imageData)
    const idx = probas.map((_, index) => index);
	// [0, 1, 4, 3] idx order max-min -> [2, 3, 1, 0]
	const sortedIdx = idx.sort((a, b) => probas[b] - probas[a]);
	const prediction = [];
	for (let i=0; i<sortedIdx.length; i++) {
		prediction.push([CLASSNAMES[sortedIdx[i]], percent[sortedIdx[i]]])
	}

    if (Math.max(...percent) > THRESHOLD) {
      if (prediction[0][0] == "christmas tree"){
        predictionDiv.innerText = prediction[0][0].split("christmas")[1]; // christmas tree -> tree (too long)
      } else {
        predictionDiv.innerText = prediction[0][0];
      }
      let resultHtml = ""
      for (let i=0; i<prediction.length; i++) {
      resultHtml +=  `<li>${prediction[i][0]}<span>${prediction[i][1]}%</span></li>`;
      }
      predictProbas.innerHTML = resultHtml;
      } else {
        predictProbas.innerHTML =
          `NOTSURE ${percent}: ${predict}!`;
      }
  }
}

async function readImageData(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (event) => resolve(event.target.result);
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
}

async function loadImageFromDataUrl(dataUrl) {
  const buffer = Buffer.from(dataUrl.split(",")[1], "base64");
  return Jimp.read(buffer).then((image) =>
    image.resize(IMAGE_SIZE, IMAGE_SIZE),
  );
}

async function loadImageFromPath(path) {
  return Jimp.read(path).then((image) =>
    image.resize(IMAGE_SIZE, IMAGE_SIZE),
  );
}

function displayImage(imageData) {
  imageDisplay.src = imageData;
}

function displayLoad(state) {
  if (state == "block") {
      containerDiv.innerHTML = `<div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>
        <div class="load"></div>`
      imageContainer.style.display = "none";
      imageDisplay.style.display = "none";
      loadCir.style.display = "block"
  } else {
    for (let i = 0; i < loadDiv.length; i++) {
      loadDiv[i].style.display = "none";
    }
    imageContainer.style.display = "flex";
    imageDisplay.style.display = "block";
    loadCir.style.display = "none";
  }
}

function imageDataToTensor(image, dims) {
  // 1. Get buffer data from image and create R, G, and B arrays.
  const imageBufferData = image.bitmap.data;
  const redArray = [];
  const greenArray = [];
  const blueArray = [];

  // 2. Loop through the image buffer and extract the R, G, and B channels
  for (let i = 0; i < imageBufferData.length; i += 4) {
    redArray.push(imageBufferData[i]);
    greenArray.push(imageBufferData[i + 1]);
    blueArray.push(imageBufferData[i + 2]);
  }

  // 3. Concatenate RGB to transpose [IMAGE_SIZE, IMAGE_SIZE, 3] -> [3, IMAGE_SIZE, IMAGE_SIZE] to a number array
  const transposedData = redArray.concat(greenArray).concat(blueArray);

  // 4. convert to float32
  const float32Data = new Float32Array(dims[1] * dims[2] * dims[3]);
  for (let i = 0; i < transposedData.length; i++) {
    float32Data[i] = transposedData[i] / 255.0; // convert to float
  }
  // 5. create the tensor object from onnxruntime-web.
  const inputTensor = new ort.Tensor("float32", float32Data, dims);
  return inputTensor;
}

function reshapeArray(inputArray, shape) {
  if (inputArray.length !== shape.reduce((acc, dim) => acc * dim, 1)) {
    throw new Error("Invalid shape for reshaping the array");
  }

  const reshapedArray = [];

  let index = 0;
  for (let i = 0; i < shape[0]; i++) {
    const row = [];
    for (let j = 0; j < shape[1]; j++) {
      const col = [];
      for (let k = 0; k < shape[2]; k++) {
        col.push(inputArray[index]);
        index++;
      }
      row.push(col);
    }
    reshapedArray.push(row);
  }

  return reshapedArray;
}
