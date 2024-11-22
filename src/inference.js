const path = require('path');
const tfjs = require('@tensorflow/tfjs-node');

async function loadModel() {
  // Path relatif dari `src/inference.js` ke `models/model.json`
  const modelUrl = `file://${path.resolve(__dirname, '../models/model.json')}`;
  console.log('Model URL:', modelUrl); // Debug log
  try {
    const model = await tfjs.loadLayersModel(modelUrl);
    console.log('Model loaded successfully');
    return model;
  } catch (error) {
    console.error('Error loading model:', error);
    throw error;
  }
}

async function predict(model, imageBuffer) {
  if (!Buffer.isBuffer(imageBuffer)) {
    throw new Error('Invalid image buffer');
  }

  const tensor = tfjs.node
    .decodeJpeg(imageBuffer)
    .resizeNearestNeighbor([150, 150])
    .expandDims()
    .toFloat();

  try {
    const predictions = await model.predict(tensor).data(); // Await ditambahkan
    console.log('Predictions:', predictions); // Untuk debugging
    return predictions;
  } finally {
    tensor.dispose(); // Membersihkan tensor
  }
}

module.exports = { loadModel, predict };
