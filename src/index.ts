import * as tf from '@tensorflow/tfjs';

// Vocabulary of layers
const vocab = ["PAD", "Conv2D", "MaxPooling", "Dense", "Dropout", "Flatten"];
// Example training sequences
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
    ["Flatten", "Dense"],
];

// Note: the model input will be the maximum sequences.length - 1

// Turns a sequence of layers into a sequence of indices
function encodeSequences(sequences: string[][], vocab: string[]) {
    const layerToIdx = Object.fromEntries(vocab.map((layer, idx) => [layer, idx]));
    return sequences.map(seq => seq.map(layer => layerToIdx[layer]));
}

// This function splits a sequence into multiple input-output pairs for testing based
// Each sequence will result it in multiple input-output pairs
function buildTrainingData(encodedSequences: number[][]) {
    // Prepare input-output pairs
    let xTrain: number[][] = [];
    let yTrain: number[] = [];
    encodedSequences.forEach(seq => {
        for (let t = 0; t < seq.length - 1; t++) {
            xTrain.push(seq.slice(0, t + 1)); // Input sequence
            yTrain.push(seq[t + 1]);         // Target layer
        }
    });
    return { xTrain, yTrain };
}

function buildIndices(vocab: string[]) {
    const layerToIdx = Object.fromEntries(vocab.map((layer, idx) => [layer, idx]));
    const idxToLayer = Object.fromEntries(vocab.map((layer, idx) => [idx, layer]));
    console.log(layerToIdx);
    return { layerToIdx, idxToLayer };
}

// padSequences ensures all of the x-values are the same length
function padSequences(sequences: number[][]) {
    const maxLen = Math.max(...xTrain.map(x => x.length), 2); // Note: this is very important, the sequence input needs to be at least, even if the second element is PAD !important!
    const paddedSequences = sequences.map(seq => {
        const pad = new Array(maxLen - seq.length).fill(layerToIdx["PAD"]);
        return [...seq, ...pad];
    });
    return { maxLen, paddedSequences };
}

// Build indices for encoding and decoding
const { layerToIdx, idxToLayer } = buildIndices(vocab);

// Encode sequences into indices
const encodedSequences = encodeSequences(sequences, vocab);

const { xTrain, yTrain } = buildTrainingData(encodedSequences);

// Pad sequences to the same length
const { maxLen, paddedSequences: xTrainPadded } = padSequences(xTrain);
console.log(`XTrainPadded: `, xTrainPadded);

// One-hot encode targets
console.log(`yTrain: `, yTrain);
const yTrainOHE: number[][] = yTrain.map<number[]>(idx => {
    const oneHot: number[] = new Array(vocab.length).fill(0);
    // const oneHot: number[] = new Array(vocab.length - 1).fill(0);
    oneHot[idx] = 1;
    // oneHot[idx - 1] = 1;
    return oneHot;
});

// Convert to tensors
const tfXTrain = tf.tensor2d(xTrainPadded);
const tfYTrain = tf.tensor2d(yTrainOHE);
console.log(tfXTrain.print());
console.log(tfYTrain.print());
console.log(tfXTrain.shape);
console.log(tfYTrain.shape);

// Build and train the model
function createModel(vocabSize: number, embeddingDim: number, lstmUnits: number) {
    const model = tf.sequential();
    model.add(tf.layers.embedding({
        inputDim: vocabSize,
        outputDim: embeddingDim,
        maskZero: true // Ignore padding values
    }));
    model.add(tf.layers.lstm({
        units: lstmUnits,
        returnSequences: false
    }));
    model.add(tf.layers.dense({
        units: vocabSize, // Predict only valid layers, excluding PAD
        activation: 'softmax' // Output probabilities
    }));
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
    return model;
}

// Create and train the model
const model = createModel(vocab.length, 16, 32);
console.log('before fit');
await model.fit(tfXTrain, tfYTrain, {
    epochs: 50,
    batchSize: 2,
    validationSplit: 0.2
});

console.log('after fit');

// const testSeq = [layerToIdx["Conv2D"], layerToIdx["MaxPooling"], layerToIdx["Flatten"]];
// const testSeq = [layerToIdx["Dense"]];
// const testSeq = [layerToIdx["Dropout"]];
// const testSeq = [layerToIdx["Conv2D"], layerToIdx["MaxPooling"]];
// const testSeq = [layerToIdx["Conv2D"]];
// const testSeq = [layerToIdx["Flatten"]];
// const testSeq = [layerToIdx["Dense"], layerToIdx["Dropout"]];
const testSeq = [layerToIdx["Dropout"], layerToIdx["Dense"]];
// const inputSeq = testSeq.map(layer => layerToIdx[layer]);
const paddedSeq = new Array(maxLen).fill(layerToIdx["PAD"]); // Pad with 0
paddedSeq.splice(0, testSeq.length, ...testSeq);
console.log(paddedSeq);
const tfTestSeq = tf.tensor2d([paddedSeq]);


function displayProbabilities(probs: tf.Tensor | tf.Tensor[]) {
    console.log('Predicted Probabilities');
    if (Array.isArray(probs)) {
        const p = probs[0].squeeze().arraySync();
        console.log(p);

    } else {
        const p = probs.squeeze().arraySync() as number[];
        console.log(p);
        // We add 1 to skip the padding index
        console.log(p.map((prob, idx) => `${idxToLayer[idx]}: ${prob.toFixed(3)}`).join('\n'));
        console.log(p);
    }
}

const probs = model.predict(tfTestSeq);
displayProbabilities(probs);

// OK, this is working, but not for a single sequence...it needs at least two for some reason...



// const paddedInput = tf.pad(tf.tensor2d([inputSeq], [1, inputSeq.length]), [[0, 0], [0, maxDepth - inputSeq.length]], -1);
// const probs = (await model.predict(paddedInput).array())[0];
// console.log("Predicted Probabilities:", probs);

// Implement Tree Search for Sequence Prediction
// async function treeSearch(model, startSequence, maxDepth, vocab, layerToIdx, idxToLayer) {
//     const priorityQueue = []; // Priority queue for tree search
//     priorityQueue.push({ sequence: startSequence, score: 1.0 });

//     let bestSequence = null;
//     let bestScore = 0;

//     while (priorityQueue.length > 0) {
//         // Dequeue the highest-priority sequence
//         const current = priorityQueue.shift();
//         const { sequence, score } = current;

//         // Stop if max depth is reached
//         if (sequence.length >= maxDepth) {
//             if (score > bestScore) {
//                 bestScore = score;
//                 bestSequence = sequence;
//             }
//             continue;
//         }

//         // Predict probabilities for the next layer
//         const inputSeq = sequence.map(layer => layerToIdx[layer]);
//         const paddedInput = tf.pad(tf.tensor2d([inputSeq], [1, inputSeq.length]), [[0, 0], [0, maxDepth - inputSeq.length]], -1);
//         const probs = (await model.predict(paddedInput).array())[0];

//         // Expand the tree for top-k predictions
//         probs.forEach((prob, idx) => {
//             if (prob > 0.1) { // Ignore low-probability paths
//                 priorityQueue.push({
//                     sequence: [...sequence, idxToLayer[idx]],
//                     score: score * prob
//                 });
//             }
//         });

//         // Sort the queue by score (descending)
//         priorityQueue.sort((a, b) => b.score - a.score);
//     }

//     return { bestSequence, bestScore };
// }

// Testing
// Define starting sequence and max depth
// const startSequence = ["Conv2D", "MaxPooling"];
// const maxDepth = 4;

// // Run tree search
// const { bestSequence, bestScore } = await treeSearch(
//     model,
//     startSequence,
//     maxDepth,
//     vocab,
//     layerToIdx,
//     idxToLayer
// );

// console.log("Best Sequence:", bestSequence);
// console.log("Best Score:", bestScore);


