// src/layerSequence/trainModel.ts
import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import { buildTrainingData, encodeSequences, padSequences, vocab } from "./domain.js";
import { trainingData } from "./trainingData.js";
function createModel(vocabSize, embeddingDim, lstmUnits) {
  const model2 = tf.sequential();
  model2.add(tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: embeddingDim,
    maskZero: true
    // Ignore padding values which are set to 0
  }));
  model2.add(tf.layers.lstm({
    units: lstmUnits,
    returnSequences: false
  }));
  model2.add(tf.layers.dense({
    units: vocabSize,
    // Predict only valid layers, excluding PAD
    activation: "softmax"
    // Output probabilities
  }));
  model2.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });
  return model2;
}
function prepareTrainingData() {
  const encodedSequences = encodeSequences(trainingData);
  const { xTrain, yTrain } = buildTrainingData(encodedSequences);
  const maxLen2 = Math.max(...xTrain.map((x) => x.length), 2);
  const xTrainPadded = padSequences(xTrain, maxLen2);
  const yTrainOHE = yTrain.map((idx) => {
    const oneHot = new Array(vocab.length).fill(0);
    oneHot[idx] = 1;
    return oneHot;
  });
  return {
    maxLen: maxLen2,
    tfXTrain: tf.tensor2d(xTrainPadded),
    tfYTrain: tf.tensor2d(yTrainOHE)
  };
}
var { maxLen, tfXTrain, tfYTrain } = prepareTrainingData();
var model = createModel(vocab.length, 16, 32);
await model.fit(tfXTrain, tfYTrain, {
  epochs: 50,
  batchSize: 4,
  validationSplit: 0.2,
  verbose: 0
  // Suppress training output
});
console.log("Model training complete; saving model to disk");
await model.save("file://./build/model");
var metadata = { vocab, maxLen };
fs.writeFileSync("build/model/model-metadata.json", JSON.stringify(metadata));
