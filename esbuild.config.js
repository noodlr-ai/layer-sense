import { context } from 'esbuild';
import { esbuildPluginFilePathExtensions } from 'esbuild-plugin-file-path-extensions';

let build = await context({
    entryPoints: ['src/layerSequence/*.ts'],
    outdir: 'build',
    bundle: true,
    format: 'esm',
    logLevel: 'info',
    platform: 'node',
    plugins: [
        esbuildPluginFilePathExtensions({
            esm: true, // Set to true for ESM format
            esmExtension: 'js', // Specify 'js' for the extension
            filter: /^\.\//, // Match only relative paths
        }),
    ],
    external: ['@tensorflow/tfjs', '@tensorflow/tfjs-node'],
});


await build.watch();
console.log('build..');