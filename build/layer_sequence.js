import * as tf from "@tensorflow/tfjs";
const vocab = ["PAD", "Dataset", "RowSplit", "ColumnSplit", "ColumnSplice", "Dense", "Dropout", "BatchNormalization", "Training", "Launcher", "ManualDataEntry", "DataViewer", "Transformer"];
const sequences = [
  ["Launcher", "ManualDataEntry", "Transformer", "DataViewer"],
  ["Launcher", "ManualDataEntry", "Transformer", "DataViewer"],
  ["Launcher", "ManualDataEntry", "Transformer", "Transformer", "DataViewer"],
  ["Dataset", "RowSplit", "ColumnSplit", "Training"],
  ["Dataset", "RowSplit", "ColumnSplit", "Dense", "Dense", "Dense", "Training"],
  ["Dataset", "RowSplit", "ColumnSplit", "Dense", "Dense", "Dense", "Training"],
  ["Dataset", "RowSplit", "ColumnSplit", "Dense", "Dense", "Dense", "Training"],
  ["Dataset", "RowSplit", "ColumnSplit", "Dense", "Dense", "Dense", "Training"],
  ["Dataset", "RowSplit", "ColumnSplice", "Training"],
  ["Dataset", "RowSplit", "ColumnSplice", "Dense", "Dense", "Dense", "Training"],
  ["Dataset", "RowSplit", "ColumnSplice", "Dense", "Dense", "Dense", "Training"],
  ["Dataset", "RowSplit", "ColumnSplice", "Dense", "Dense", "Dense", "Training"],
  ["Dataset", "RowSplit", "ColumnSplice", "Dense", "Dense", "Dense", "Training"]
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
  return { layerToIdx: layerToIdx2, idxToLayer: idxToLayer2 };
}
function padSequence(sequence, maxLen2) {
  const pad = new Array(maxLen2 - sequence.length).fill(layerToIdx["PAD"]);
  return [...sequence, ...pad];
}
function padSequences(sequences2, maxLen2) {
  return sequences2.map((seq) => padSequence(seq, maxLen2));
}
const { layerToIdx, idxToLayer } = buildIndices(vocab);
const encodedSequences = encodeSequences(sequences, vocab);
const { xTrain, yTrain } = buildTrainingData(encodedSequences);
const maxLen = Math.max(...xTrain.map((x) => x.length), 2);
const xTrainPadded = padSequences(xTrain, maxLen);
const yTrainOHE = yTrain.map((idx) => {
  const oneHot = new Array(vocab.length).fill(0);
  oneHot[idx] = 1;
  return oneHot;
});
const tfXTrain = tf.tensor2d(xTrainPadded);
const tfYTrain = tf.tensor2d(yTrainOHE);
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
console.log("Training model");
await model.fit(tfXTrain, tfYTrain, {
  epochs: 50,
  batchSize: 2,
  validationSplit: 0.2
});
console.log("Model training complete");
const testSeq = [layerToIdx["Launcher"], layerToIdx["ManualDataEntry"]];
const paddedSeq = padSequence(testSeq, maxLen);
const tfTestSeq = tf.tensor2d([paddedSeq]);
function displayProbabilities(probs2) {
  console.log("Predicted Probabilities");
  if (Array.isArray(probs2)) {
    const p = probs2[0].squeeze().arraySync();
    console.log(p);
  } else {
    const p = probs2.squeeze().arraySync();
    console.log(p.map((prob, idx) => `${idxToLayer[idx]}: ${prob.toFixed(3)}`).join("\n"));
  }
}
const probs = model.predict(tfTestSeq);
displayProbabilities(probs);
async function treeSearch(model2, startSequence2, maxDepth2, vocab2, layerToIdx2, idxToLayer2) {
  const priorityQueue = [];
  priorityQueue.push({ sequence: startSequence2, score: 1 });
  let bestSequence2 = null;
  let bestScore2 = 0;
  while (priorityQueue.length > 0) {
    const current = priorityQueue.shift();
    const { sequence, score } = current;
    if (sequence.length >= maxDepth2) {
      if (score > bestScore2) {
        bestScore2 = score;
        bestSequence2 = sequence;
      }
      continue;
    }
    const inputSeq = sequence.map((layer) => layerToIdx2[layer]);
    const paddedInput = padSequence(inputSeq, maxDepth2);
    const prediction = model2.predict(tf.tensor2d([paddedInput]));
    const probs2 = (await prediction.array())[0];
    probs2.forEach((prob, idx) => {
      if (prob > 0.1) {
        priorityQueue.push({
          sequence: [...sequence, idxToLayer2[idx]],
          score: score * prob
          // Calculate the new score as a cumulative probability that punishes low-probability paths
        });
      }
    });
    priorityQueue.sort((a, b) => b.score - a.score);
  }
  return { bestSequence: bestSequence2, bestScore: bestScore2 };
}
const startSequence = ["Dataset"];
const maxDepth = 4;
const { bestSequence, bestScore } = await treeSearch(
  model,
  startSequence,
  maxDepth,
  vocab,
  layerToIdx,
  idxToLayer
);
console.log("Best Sequence:", bestSequence);
console.log("Best Score:", bestScore);
