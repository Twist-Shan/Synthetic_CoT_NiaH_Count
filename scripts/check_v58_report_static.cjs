// Parse only: this does not execute the report or emulate a blocked browser.
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const report = path.resolve(process.argv[2]);
const text = fs.readFileSync(report, 'utf8');
const scripts = [...text.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)];
for (let i = 0; i < scripts.length; i++) new vm.Script(scripts[i][1], {filename: 'inline-' + i});
const match = text.match(/const V58_GEOMETRY_DATA=(\[[\s\S]*?\]);<\/script>/);
if (!match) throw Error('Missing geometry payload');
const data = JSON.parse(match[1]);
const endpoints = ['nonthinking_prompt_occurrence','thinking_item_end','nonthinking_answer_query','thinking_answer_query'];
for (const endpoint of endpoints) for (const layer of [1,2,3,4]) {
  const rows = data.filter(x => x.endpoint === endpoint && x.layer === layer);
  const n = endpoint.includes('answer_query') ? 100 : 550;
  if (rows.length !== n || new Set(rows.map(x => x.sample)).size !== 100) throw Error('Bad geometry coverage');
  if (rows.some(x => ![x.pc1,x.pc2,x.pc3].every(Number.isFinite))) throw Error('Nonfinite geometry');
}
for (const id of ['geometry-layer-2d','geometry-layer-3d','geometry-2d','geometry-3d']) {
  if (text.split('id="' + id + '"').length !== 2) throw Error('Missing/duplicate control ' + id);
}
if (!text.includes("type:'scatter3d'") || !text.includes("dragmode:'orbit'")) throw Error('Missing 3D configuration');
console.log(JSON.stringify({status:'passed',inlineScripts: scripts.length, geometryRows:data.length,
  geometryPanels:16,querySelectorIdsUnique:true,javascriptSyntax:'valid',
  interactiveBrowserTest:'not performed: local file URL blocked by browser policy'},null,2));
