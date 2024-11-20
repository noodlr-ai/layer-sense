import { context } from 'esbuild';

let build = await context({
    entryPoints: ['src/layer_sequence.ts', 'src/feature_sequence.ts'],
    outdir: 'build',
    // bundle: true,
    format: 'esm',
    logLevel: 'info',
});


await build.watch();
console.log('build..');