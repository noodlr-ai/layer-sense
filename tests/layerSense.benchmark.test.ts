import * as tf from '@tensorflow/tfjs-node';
import { performance } from 'perf_hooks';
import fs from 'fs';
import path from 'path';
import { Parser } from '@json2csv/plainjs';
import { LayerType, convertSequenceIntoTensors, getProbabilities } from '../src/layerSequence/domain';
import { treeSearch } from '../build/treeSearch';

type InferenceResults = {
    iteration: number;
    timeMs: string;
}

type TreeSearchResults = {
    maxDepth: number;
    averageTimeMs: string;
}

// Helper function to write results to CSV
function writeResultsToCSV(filename: string, data: (InferenceResults | TreeSearchResults)[]) {
    const fields = Object.keys(data[0]);
    const parser = new Parser({ fields });
    const csv = parser.parse(data);
    const filePath = path.join(__dirname, filename);
    fs.writeFileSync(filePath, csv);
    console.log(`Results written to ${filePath}`);
}

// Benchmarking tests
describe('Benchmarking inference and tree search', () => {
    let model: tf.LayersModel;
    it('Loads the model', async () => {
        model = await tf.loadLayersModel('file://build/model/model.json');
        expect(model).toBeDefined
    });

    it('Benchmark: Predict next layer multiple times and save results', async () => {
        const testSeq: LayerType[] = ['dataset', 'rowSplit', 'columnSplit', 'dense', 'dense'];
        const inputSeq = convertSequenceIntoTensors(testSeq);

        const iterations = 100; // Number of iterations to benchmark
        const results: InferenceResults[] = []; // Store results here

        for (let i = 0; i < iterations; i++) {
            const start = performance.now();

            const pTensor = model.predict(inputSeq);
            const probs = getProbabilities(pTensor);

            const end = performance.now();
            results.push({ iteration: i + 1, timeMs: (end - start).toFixed(2) });

            // Validate during benchmark
            expect(probs[0].layer).toEqual('dense');
        }

        writeResultsToCSV('predict_benchmark.csv', results);
    });

    it('Benchmark: Tree search with different max depths and average over 5 runs', async () => {
        const inputSeq: LayerType[] = ['dataset', 'rowSplit'];
        const depths = [2, 4, 6, 8, 10, 12]; // Different maximum depths for testing
        const runsPerDepth = 5; // Number of runs for each depth
        const results: TreeSearchResults[] = []; // Store results here

        for (const maxDepth of depths) {
            let totalTime = 0;

            for (let run = 0; run < runsPerDepth; run++) {
                const start = performance.now();

                const seq = await treeSearch(model, inputSeq, maxDepth);

                const end = performance.now();
                totalTime += (end - start);
            }

            // Calculate average time and store the result
            const averageTime = (totalTime / runsPerDepth).toFixed(2);
            results.push({
                maxDepth,
                averageTimeMs: averageTime,
            });
        }

        writeResultsToCSV('tree_search_benchmark.csv', results);
    });
});