import * as tf from '@tensorflow/tfjs';
import { LayerType, convertSequenceIntoTensors, idxToLayer, layerToIdx, padSequence } from './domain';

// treeSearch performs a tree search to find the best sequence of layers doesn to the maxDepth.
// For instance, if the initial sequence is ['launcher', 'manualDataEntry'], the tree search will find the best sequence of layers that maximizes the probability of the next layer,
// up to a total length of initial sequence + maxDepth.
export async function treeSearch(model: tf.LayersModel, startSequence: LayerType[], maxDepth: number) {
    type QueueItem = { sequence: LayerType[], score: number };
    const priorityQueue: QueueItem[] = []; // Priority queue for tree search
    priorityQueue.push({ sequence: startSequence, score: 1.0 }); // Use 1.0 as the base score

    let bestSequence: QueueItem['sequence'] = [];
    let bestScore = 0;

    // While loop avoids recursion to reduce memory footprint
    while (priorityQueue.length > 0) {
        // Dequeue the highest-priority sequence
        const current = priorityQueue.shift()!;
        const { sequence, score } = current;

        // Calculate the appended depth (difference from the initial sequence length)
        const appendedDepth = sequence.length - startSequence.length;

        // If we have reached the maximum appended depth, update the best score and sequence if applicable
        if (appendedDepth >= maxDepth) {
            if (score > bestScore) {
                bestScore = score;
                bestSequence = sequence;
            }
            continue;
        }
        // Predict probabilities for the next layer
        const inputSeq = convertSequenceIntoTensors(sequence);
        const prediction = model.predict(inputSeq) as tf.Tensor;
        const probs = (await prediction.array())[0] as number[];

        // Expand the tree for top-k predictions
        probs.forEach((prob, idx) => {
            // Each probability greater than 0.1 creates a new path to explore in the tree
            if (prob > 0.1) { // Ignore low-probability paths
                const layer = idxToLayer(idx);
                // We don't want to use in our search frontier
                if (layer !== 'PAD') {
                    priorityQueue.push({
                        sequence: [...sequence, layer],
                        score: score * prob // Calculate the new score as a cumulative probability that punishes low-probability paths
                    });
                }
            }
        });

        // Sort the queue by score (descending)
        priorityQueue.sort((a, b) => b.score - a.score);
    }

    return { bestSequence, bestScore };
}