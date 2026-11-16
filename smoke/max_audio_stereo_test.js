/* Run with node smoke/max_audio_stereo_test.js.
 * Exercises the production legacy-Max script against a strict patcher API mock
 * and checks the fixed AMXD graph. This is not a substitute for a Max DSP run.
 */
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const root = path.resolve(__dirname, '..');
const scriptFile = process.argv[2] || path.join(root, 'inference/after.m4l/after.audio.js');
let source = fs.readFileSync(scriptFile, 'utf8');
// Max's legacy JS accepts a reserved-word message handler that Node rejects.
// Adapt that identifier only in this test harness; production dispatch stays
// unchanged. No channel/layout expressions or function bodies are rewritten.
assert.strictEqual((source.match(/function delete\(\)/g) || []).length, 1);
source = source.replace('function delete()', 'function after_test_delete()');

const device = fs.readFileSync(path.join(root, 'inference/after.m4l/after.audio.amxd'));
assert.strictEqual(device.subarray(0, 4).toString(), 'ampf');
assert.strictEqual(device.readUInt32LE(16), device.length - 20);
assert.strictEqual(device[device.length - 1], 0);
const patcher = JSON.parse(device.subarray(20, device.length - 1)).patcher;
const boxes = new Map(patcher.boxes.map(row => [row.box.id, row.box]));
assert.strictEqual(boxes.size, patcher.boxes.length, 'duplicate top-level IDs');
for (const {patchline: line} of patcher.lines) {
    assert(boxes.has(line.source[0]) && boxes.has(line.destination[0]), 'dangling patch cord');
}
function edge(a, outlet, b, inlet) {
    assert(patcher.lines.some(({patchline: p}) => p.source[0] === a && p.source[1] === outlet && p.destination[0] === b && p.destination[1] === inlet), `missing ${a}:${outlet} -> ${b}:${inlet}`);
}
const added = name => 'obj-after-audio-' + name;
// The existing mono average stays before feedback and structure filtering.
edge('obj-19', 0, 'obj-272', 0); edge('obj-19', 1, 'obj-272', 1);
edge('obj-272', 0, 'obj-518', 0);
for (const [channel, outlet] of [['left', 0], ['right', 1]]) {
    edge('obj-518', 0, added('input-' + channel), 1);
    edge('obj-19', outlet, added('input-' + channel), 2);
    assert.strictEqual(boxes.get(added('input-' + channel)).text, 'selector~ 2 1');
    edge(added('channels'), 0, added('input-' + channel), 0);
}
edge(added('input-left'), 0, 'obj-463', 0);
edge(added('input-right'), 0, added('feedback-sum-right'), 0);
// The right branch owns all histories; it shares only parameter signals.
edge(added('feedback-receive-right'), 0, added('feedback-sum-right'), 1);
edge(added('feedback-sum-right'), 0, added('filter-right'), 0);
edge('obj-245', 0, added('filter-right'), 0);
edge('obj-243', 0, added('filter-select-right'), 0);
edge(added('feedback-sum-right'), 0, added('filter-select-right'), 1);
edge(added('filter-right'), 0, added('filter-select-right'), 2);
edge(added('filter-select-right'), 0, added('input-send-right'), 0);
edge(added('feedback-model-right'), 0, added('feedback-gain-right'), 0);
edge('obj-256', 0, added('feedback-gain-right'), 1);
edge(added('feedback-gain-right'), 0, added('feedback-send-right'), 0);
edge(added('dry-receive-right'), 0, added('dry-gain-right'), 0);
edge('obj-269', 0, added('dry-gain-right'), 1);
edge(added('wet-receive-right'), 0, added('delay-right'), 0);
edge('obj-1426', 0, added('delay-right'), 1);
edge(added('delay-right'), 0, added('wet-gain-right'), 0);
edge('obj-268', 0, added('wet-gain-right'), 1);
edge(added('dry-gain-right'), 0, added('mix-right'), 0);
edge(added('wet-gain-right'), 0, added('mix-right'), 1);
edge(added('mix-right'), 0, 'obj-259', 1);
assert.deepStrictEqual(boxes.get(added('delay-right')).patcher, boxes.get('obj-1422').patcher);
assert.strictEqual(boxes.get(added('feedback-send-right')).text, 'send~ ---feedback_r');
assert.strictEqual(boxes.get(added('feedback-receive-right')).text, 'receive~ ---feedback_r');
assert.strictEqual(boxes.get(added('input-send-right')).text, 'send~ ---plug_in_r');
assert.strictEqual(boxes.get(added('dry-receive-right')).text, 'receive~ ---plug_in_r');
// Explicit final duplication preserves mono even after stereo histories differ.
edge('obj-259', 0, 'obj-24', 0);
edge('obj-259', 0, added('output-right'), 1);
edge('obj-259', 1, added('output-right'), 2);
edge(added('channels'), 0, added('output-right'), 0);
edge(added('output-right'), 0, 'obj-24', 1);
for (const {box} of patcher.boxes) {
    if (box.id.startsWith('obj-after-audio-'))
        assert(!box.parameter_enable && !box.presentation, 'stereo wiring added a host/UI parameter');
}

function environment(config) {
    let serial = 0;
    const created = [], connections = [], posts = [], output = [];
    function object(ni, no, props = {}) {
        return Object.assign({id: 'dynamic-' + serial++, numinlets: ni, numoutlets: no, maxclass: 'newobj', removed: false,
            getboxattr(name) { return this[name]; },
            message(...args) { this.messages = (this.messages || []).concat([args]); },
            setattr(...args) { this.attrs = (this.attrs || []).concat([args]); }
        }, props);
    }
    const named = {};
    for (const {box} of patcher.boxes) {
        if (box.varname) named[box.varname] = object(box.numinlets, box.numoutlets, {id: box.id, maxclass: box.maxclass, text: box.text});
    }
    const p = {
        getnamed(name) { return named[name] || null; },
        newdefault(x, y, kind, model, method) {
            let obj;
            if (kind === 'nn~') {
                let shape;
                if (method === 'diffuse_timbre_modulate' && config.legacy)
                    obj = object(0, 0, {maxclass: config.missingMethodClass || 'newobj', method});
                else {
                    const t = config.timbres || 6, n = config.channels;
                    shape = {
                        diffuse_timbre_modulate: [config.inputs || n + t + 2 * (config.structures || 8), config.codec || 16],
                        decode: [config.decoderInputs || config.codec || 16, n],
                        generate_timbre: [n + t, n],
                        map2latent: [2, config.mapWidth || t],
                        latent2map: [config.mapWidth || t, 2]
                    }[method];
                    assert(shape, 'unknown method ' + method);
                    obj = object(...shape, {maxclass: 'nn~', method});
                }
            } else if (kind === 'mc.unpack~') obj = object(1, model, {kind});
            else if (kind === 'r~' || kind === 'sig~') obj = object(1, 1, {kind, argument: model});
            else throw new Error('unexpected dynamically created class: ' + kind);
            created.push(obj); return obj;
        },
        connect(a, outlet, b, inlet) {
            assert(a && b && !a.removed && !b.removed, 'missing/removed object');
            assert(Number.isInteger(outlet) && outlet >= 0 && outlet < a.numoutlets, `invalid ${a.method || a.id} outlet ${outlet}`);
            assert(Number.isInteger(inlet) && inlet >= 0 && inlet < b.numinlets, `invalid ${b.method || b.id} inlet ${inlet}`);
            connections.push({a, outlet, b, inlet});
        },
        remove(obj) {
            assert(obj && !obj.removed, 'object removed twice');
            obj.removed = true;
            for (let i = connections.length - 1; i >= 0; --i) if (connections[i].a === obj || connections[i].b === obj) connections.splice(i, 1);
        }
    };
    const context = {patcher: p, post: (...args) => posts.push(args.join(' ')), outlet: (...args) => output.push(args)};
    vm.createContext(context); vm.runInContext(source, context, {filename: scriptFile});
    context.delete = context.after_test_delete;
    context.prefix('timbre', 'projection', 'structure');
    return {context, config, created, connections, named, posts, output,
        mode() { return named.audio_channels.messages.slice(-1)[0][1]; },
        alive() { return created.filter(x => !x.removed); }};
}
function dynamicEdge(e, from, outlet, to, inlet) {
    assert(e.connections.some(c => c.a === from && c.outlet === outlet && c.b === to && c.inlet === inlet), 'missing dynamic patch cord');
}
function checkLoad(e, legacy) {
    e.context.create('fixture.ts', 8192, 0);
    assert(e.output.some(x => x[0] === 1 && x[1] === 'create'), e.posts.join('\n'));
    assert.strictEqual(e.mode(), e.config.channels);
    const generator = e.alive().find(x => x.method === (legacy ? 'generate_timbre' : 'diffuse_timbre_modulate'));
    const decoder = legacy ? generator : e.alive().find(x => x.method === 'decode');
    dynamicEdge(e, e.named.plug_in, 0, generator, 0);
    if (e.config.channels === 2) dynamicEdge(e, e.named.plug_in_r, 0, generator, 1);
    dynamicEdge(e, decoder, 0, e.named.rave_out, 0);
    dynamicEdge(e, decoder, e.config.channels - 1, e.named.rave_out_r, 0);
    const unpack = e.alive().find(x => x.kind === 'mc.unpack~');
    for (let i = 0; i < (e.config.timbres || 6); ++i)
        dynamicEdge(e, unpack, i, generator, e.config.channels + i);
    if (!legacy) {
        for (let i = 0; i < (e.config.structures || 8); ++i) {
            const scale = e.alive().find(x => x.kind === 'r~' && x.argument === 'structure' + i + '_s');
            const bias = e.alive().find(x => x.kind === 'r~' && x.argument === 'structure' + i + '_b');
            dynamicEdge(e, scale, 0, generator, e.config.channels + 6 + 2 * i);
            dynamicEdge(e, bias, 0, generator, e.config.channels + 6 + 2 * i + 1);
        }
        for (let i = 0; i < (e.config.codec || 16); ++i) dynamicEdge(e, generator, i, decoder, i);
    }
    const m2l = e.alive().find(x => x.method === 'map2latent');
    const l2m = e.alive().find(x => x.method === 'latent2map');
    assert.strictEqual(e.connections.filter(c => c.a === m2l).length, e.config.timbres || 6);
    assert.strictEqual(e.connections.filter(c => c.b === l2m).length, e.config.timbres || 6);
    e.context.delete();
    assert.strictEqual(e.alive().length, 0, 'delete leaked dynamic objects');
    assert.strictEqual(e.mode(), 1);
}
for (const config of [{channels: 2}, {channels: 1, codec: 64}, {channels: 1, legacy: true}, {channels: 2, legacy: true}, {channels: 2, legacy: true, missingMethodClass: 'nn~'}, {channels: 1, legacy: true, timbres: 8}]) {
    checkLoad(environment(config), config.legacy);
}
const switched = environment({channels: 1});
for (const channels of [1, 2, 1, 2]) {
    switched.config.channels = channels;
    const before = switched.created.length;
    switched.context.create('fixture.ts', 8192, 0);
    assert.strictEqual(switched.mode(), channels);
    assert(switched.created.slice(0, before).every(x => x.removed), 'switch leaked old objects');
}
switched.context.create('off', 8192, 0);
assert.strictEqual(switched.alive().length, 0);
assert.strictEqual(switched.mode(), 1);
for (const config of [{channels: 3}, {channels: 2, inputs: 25}, {channels: 2, decoderInputs: 32}, {channels: 2, mapWidth: 8}, {channels: 3, legacy: true}]) {
    const e = environment(config);
    e.context.create('fixture.ts', 8192, 0);
    assert.strictEqual(e.output.length, 0, 'invalid layout reported success');
    assert.strictEqual(e.alive().length, 0, 'invalid layout leaked objects');
    assert.strictEqual(e.mode(), 1);
    assert(e.posts.some(x => x.includes('[ERROR]')));
}
console.log('PASS: audio mono/stereo + legacy dynamic wiring, reload/teardown/rejection, independent fixed DSP and mono compatibility selectors');
