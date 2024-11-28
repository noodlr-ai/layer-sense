// src/layerSequence/treeSearch.ts
import { convertSequenceIntoTensors, idxToLayer } from "./domain.js";
async function treeSearch(model, startSequence, maxDepth) {
  const priorityQueue = [];
  priorityQueue.push({ sequence: startSequence, score: 1 });
  let bestSequence = [];
  let bestScore = 0;
  while (priorityQueue.length > 0) {
    const current = priorityQueue.shift();
    const { sequence, score } = current;
    const appendedDepth = sequence.length - startSequence.length;
    if (appendedDepth >= maxDepth) {
      if (score > bestScore) {
        bestScore = score;
        bestSequence = sequence;
      }
      continue;
    }
    const inputSeq = convertSequenceIntoTensors(sequence);
    const prediction = model.predict(inputSeq);
    const probs = (await prediction.array())[0];
    probs.forEach((prob, idx) => {
      if (prob > 0.1) {
        const layer = idxToLayer(idx);
        if (layer !== "PAD") {
          priorityQueue.push({
            sequence: [...sequence, layer],
            score: score * prob
            // Calculate the new score as a cumulative probability that punishes low-probability paths
          });
        }
      }
    });
    priorityQueue.sort((a, b) => b.score - a.score);
  }
  return { bestSequence, bestScore };
}
export {
  treeSearch
};
