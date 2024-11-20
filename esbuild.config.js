import { context } from 'esbuild';

let build = await context({
    entryPoints: ['src/index.ts'],
    outdir: 'build',
    // bundle: true,
    format: 'esm',
    logLevel: 'info',
});


await build.watch();
console.log('build..');