// src/layerSequence/domain.ts
import * as tf from "@tensorflow/tfjs-node";
var vocab = [
  "PAD",
  "dataset",
  "rowSplit",
  "columnSplit",
  "columnSplice",
  "dense",
  "dropout",
  "batchNormalization",
  "training",
  "launcher",
  "manualDataEntry",
  "dataViewer",
  "transformerFillMask",
  "transformerSentimentAnalysis",
  "transformerSummarization"
];
function layerToIdx(layer) {
  const layerToIdx2 = Object.fromEntries(vocab.map((layer2, idx) => [layer2, idx]));
  return layerToIdx2[layer];
}
function idxToLayer(idx) {
  const idxToLayer2 = Object.fromEntries(vocab.map((layer, idx2) => [idx2, layer]));
  return idxToLayer2[idx];
}
function padSequence(sequence, maxLen) {
  const pad = new Array(maxLen - sequence.length).fill(layerToIdx("PAD"));
  return [...sequence, ...pad];
}
function padSequences(sequences, maxLen) {
  return sequences.map((seq) => padSequence(seq, maxLen));
}
function encodeSequences(sequences) {
  return sequences.map((seq) => seq.map((layer) => layerToIdx(layer)));
}
function buildTrainingData(encodedSequences) {
  let xTrain = [];
  let yTrain = [];
  encodedSequences.forEach((seq) => {
    for (let t = 0; t < seq.length - 1; t++) {
      xTrain.push(seq.slice(0, t + 1));
      yTrain.push(seq[t + 1]);
    }
  });
  return { xTrain, yTrain };
}
function convertSequenceIntoTensors(sequence) {
  const inputSeq = sequence.map((layer) => layerToIdx(layer));
  return tf.tensor2d([inputSeq]);
}
function displayProbabilities(probs) {
  if (Array.isArray(probs)) {
    const p = probs[0].squeeze().arraySync();
    console.log(p);
  } else {
    const p = probs.squeeze().arraySync();
    console.log(p.map((prob, idx) => `${idxToLayer(idx)}: ${prob.toFixed(3)}`).join("\n"));
  }
}
function getProbabilities(pTensor) {
  const probs = Array.isArray(pTensor) ? pTensor[0].squeeze().arraySync() : pTensor.squeeze().arraySync();
  return probs.map((prob, idx) => ({ layer: idxToLayer(idx), prob })).filter((d) => d.layer !== "PAD").sort((a, b) => b.prob - a.prob);
}
export {
  buildTrainingData,
  convertSequenceIntoTensors,
  displayProbabilities,
  encodeSequences,
  getProbabilities,
  idxToLayer,
  layerToIdx,
  padSequence,
  padSequences,
  vocab
};
