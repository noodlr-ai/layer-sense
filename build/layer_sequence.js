import * as tf from "@tensorflow/tfjs";
const vocab = ["PAD", "Conv2D", "MaxPooling", "Dense", "Dropout", "Flatten"];
const sequences = [
  ["Conv2D", "MaxPooling", "Dense"],
  ["Conv2D", "MaxPooling", "Dense"],
  ["Conv2D", "MaxPooling", "Dense"],
  ["Conv2D", "MaxPooling", "Dense"],
  ["Conv2D", "MaxPooling", "Dense"],
  ["Conv2D", "MaxPooling", "Dense"],
  ["Conv2D", "MaxPooling", "Dense"],
  ["Conv2D", "MaxPooling", "Flatten", "Dense"],
  ["Conv2D", "MaxPooling", "Flatten", "Dense"],
  ["Conv2D", "MaxPooling", "Flatten", "Dense"],
  ["Conv2D", "MaxPooling", "Flatten", "Dense"],
  ["Conv2D", "MaxPooling", "Flatten", "Dense"],
  ["Conv2D", "MaxPooling", "Flatten", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dense", "Dropout", "Dense"],
  ["Dropout", "Dense", "Flatten"],
  ["Dropout", "Dense", "Flatten"],
  ["Dropout", "Dense", "Flatten"],
  ["Dropout", "Dense", "Flatten"],
  ["Dropout", "Dense", "Flatten"],
  ["Dropout", "Dense", "Flatten"],
  ["Dropout", "Dense", "Flatten"],
  ["Dropout", "Dense", "Flatten"],
  ["Dropout", "Dense"],
  ["Dropout", "Dense"],
  ["Dropout", "Dense"],
  ["Dropout", "Dense"],
  ["Dropout", "Dense"],
  ["Dropout", "Dense"],
  ["Dropout", "Dense"],
  ["Dropout", "Dense"],
  ["Flatten", "Dense"],
  ["Flatten", "Dense"],
  ["Flatten", "Dense"],
  ["Flatten", "Dense"],
  ["Flatten", "Dense"],
  ["Flatten", "Dense"],
  ["Flatten", "Dense"],
  ["Flatten", "Dense"]
];
function encodeSequences(sequences2, vocab2) {
  const layerToIdx2 = Object.fromEntries(vocab2.map((layer, idx) => [layer, idx]));
  return sequences2.map((seq) => seq.map((layer) => layerToIdx2[layer]));
}
function buildTrainingData(encodedSequences2) {
  let xTrain2 = [];
  let yTrain2 = [];
  encodedSequences2.forEach((seq) => {
    for (let t = 0; t < seq.length - 1; t++) {
      xTrain2.push(seq.slice(0, t + 1));
      yTrain2.push(seq[t + 1]);
    }
  });
  return { xTrain: xTrain2, yTrain: yTrain2 };
}
function buildIndices(vocab2) {
  const layerToIdx2 = Object.fromEntries(vocab2.map((layer, idx) => [layer, idx]));
  const idxToLayer2 = Object.fromEntries(vocab2.map((layer, idx) => [idx, layer]));
  console.log(layerToIdx2);
  return { layerToIdx: layerToIdx2, idxToLayer: idxToLayer2 };
}
function padSequences(sequences2) {
  const maxLen2 = Math.max(...xTrain.map((x) => x.length), 2);
  const paddedSequences = sequences2.map((seq) => {
    const pad = new Array(maxLen2 - seq.length).fill(layerToIdx["PAD"]);
    return [...seq, ...pad];
  });
  return { maxLen: maxLen2, paddedSequences };
}
const { layerToIdx, idxToLayer } = buildIndices(vocab);
const encodedSequences = encodeSequences(sequences, vocab);
const { xTrain, yTrain } = buildTrainingData(encodedSequences);
const { maxLen, paddedSequences: xTrainPadded } = padSequences(xTrain);
console.log(`XTrainPadded: `, xTrainPadded);
console.log(`yTrain: `, yTrain);
const yTrainOHE = yTrain.map((idx) => {
  const oneHot = new Array(vocab.length).fill(0);
  oneHot[idx] = 1;
  return oneHot;
});
const tfXTrain = tf.tensor2d(xTrainPadded);
const tfYTrain = tf.tensor2d(yTrainOHE);
console.log(tfXTrain.print());
console.log(tfYTrain.print());
console.log(tfXTrain.shape);
console.log(tfYTrain.shape);
function createModel(vocabSize, embeddingDim, lstmUnits) {
  const model2 = tf.sequential();
  model2.add(tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: embeddingDim,
    maskZero: true
    // Ignore padding values
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
const model = createModel(vocab.length, 16, 32);
console.log("before fit");
await model.fit(tfXTrain, tfYTrain, {
  epochs: 50,
  batchSize: 2,
  validationSplit: 0.2
});
console.log("after fit");
const testSeq = [layerToIdx["MaxPooling"], layerToIdx["Flatten"]];
const paddedSeq = new Array(maxLen).fill(layerToIdx["PAD"]);
paddedSeq.splice(0, testSeq.length, ...testSeq);
console.log(paddedSeq);
const tfTestSeq = tf.tensor2d([paddedSeq]);
function displayProbabilities(probs2) {
  console.log("Predicted Probabilities");
  if (Array.isArray(probs2)) {
    const p = probs2[0].squeeze().arraySync();
    console.log(p);
  } else {
    const p = probs2.squeeze().arraySync();
    console.log(p);
    console.log(p.map((prob, idx) => `${idxToLayer[idx]}: ${prob.toFixed(3)}`).join("\n"));
    console.log(p);
  }
}
const probs = model.predict(tfTestSeq);
displayProbabilities(probs);
