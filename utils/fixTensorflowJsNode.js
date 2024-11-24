import fs from 'fs';
import path from 'path';

const source = path.resolve(__dirname, 'node_modules/@tensorflow/tfjs-node/deps/lib/tensorflow.dll');
const destination = path.resolve(__dirname, 'node_modules/@tensorflow/tfjs-node/lib/napi-v8/tensorflow.dll');

fs.copyFile(source, destination, (err) => {
    if (err) {
        console.error('Error copying tensorflow.dll:', err);
    } else {
        console.log('tensorflow.dll copied successfully!');
    }
});
