import * as tf from '@tensorflow/tfjs';
import { LayerType, idxToLayer, layerToIdx, padSequence } from './domain';

// Implement Tree Search for Sequence Prediction
export async function treeSearch(model: tf.Sequential, startSequence: LayerType[], maxDepth: number) {
    type QueueItem = { sequence: LayerType[], score: number };
    const priorityQueue: QueueItem[] = []; // Priority queue for tree search
    priorityQueue.push({ sequence: startSequence, score: 1.0 }); // Use 1.0 as the base score

    let bestSequence: QueueItem['sequence'] | null = null;
    let bestScore = 0;

    // While loop avoids recursion to reduce memory footprint
    while (priorityQueue.length > 0) {
        // Dequeue the highest-priority sequence
        const current = priorityQueue.shift()!;
        const { sequence, score } = current;

        // Once we reach out max depth, see if the score is better than the current best score
        if (sequence.length >= maxDepth) {
            if (score > bestScore) {
                bestScore = score;
                bestSequence = sequence;
            }
            continue;
        }

        // Predict probabilities for the next layer
        const inputSeq = sequence.map(layer => layerToIdx(layer));
        const paddedInput = padSequence(inputSeq, maxDepth);
        const prediction = model.predict(tf.tensor2d([paddedInput])) as tf.Tensor;
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