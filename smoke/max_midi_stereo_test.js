/* Headless contract/patch-graph regression; run `node smoke/max_midi_stereo_test.js`.
 * This simulates Max patcher calls and the fixed routing graph. It is not a Max
 * runtime/audio capture or an inference-quality test. No models are required.
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { execFileSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const devicePath = 'inference/after.m4l/after.midi.amxd';
const binary = fs.readFileSync(path.join(root, devicePath));
const decode = b => JSON.parse(b.subarray(20).toString().replace(/\0+$/, '')).patcher;
const patch = decode(binary);
// Optional contribution-review check; ordinary tests also run in shallow or
// source-archive checkouts without historical Git objects.
const baselineBinary = process.env.AFTER_MAX_BASELINE_GIT_REF ? execFileSync('git',
    ['show', process.env.AFTER_MAX_BASELINE_GIT_REF + ':' + devicePath], { cwd: root }) : null;
const baseline = baselineBinary ? decode(baselineBinary) : null;
const byId = Object.fromEntries(patch.boxes.map(x => [x.box.id, x.box]));
const source = fs.readFileSync(path.join(root, 'inference/after.m4l/after.midi.js'), 'utf8');

// The bundled legacy script uses this handler spelling; Node reserves `delete`.
// Adapt only that identifier for the mock, leaving the body and routing intact.
assert.equal((source.match(/function delete\(\)/g) || []).length, 1);
const nodeSource = source.replace('function delete()', 'function delete_message()');
function fixture(timbres, audio, overrides = {}) {
    return { diffuse: [16 + timbres, 16], decode: [16, audio],
        map2latent: [2, timbres], latent2map: [timbres, 2], ...overrides };
}
function environment(attributeType = 'number') {
    let serial = 0;
    const boxes = new Set(), links = [], posts = [];
    const staticBoxes = patch.boxes.map(x => ({ ...x.box,
        getboxattr(name) { return this[name]; },
        setattr(name, value) { this[name] = value; },
        message(selector, value) { this.lastMessage = [selector, value]; } }));
    const names = Object.fromEntries(staticBoxes.filter(x => x.varname).map(x => [x.varname, x]));
    const models = { mono6: fixture(6, 1), stereo6: fixture(6, 2), mono8: fixture(8, 1), stereo8: fixture(8, 2),
        badChannels: fixture(6, 3), badCodec: fixture(6, 2, { decode: [32, 2] }),
        badTimbre: fixture(9, 2), badMap: fixture(6, 2, { map2latent: [2, 5] }) };
    const p = {
        getnamed(name) { return names[name] || null; },
        newdefault(x, y, type, ...args) {
            let input = 1, output = 1;
            if (type === 'nn~') [input, output] = models[args[0]][args[1]];
            else if (type === 'unpack') output = args.length;
            else if (type === 'mc.unpack~') output = args[0];
            const b = { id: 'dynamic-' + (++serial), maxclass: type, args, numinlets: input,
                numoutlets: output, getboxattr(name) { return attributeType === 'array' ? [this[name]] : attributeType === 'string' ? String(this[name]) : this[name]; } };
            boxes.add(b); return b;
        },
        connect(a, out, b, input) {
            assert(a && b, 'Connection endpoint must exist');
            assert(out >= 0 && out < a.numoutlets, `Invalid outlet${out} on ${a.id}`);
            assert(input >= 0 && input < b.numinlets, `Invalid inlet${input} on ${b.id}`);
            links.push({ a, out, b, input });
        },
        remove(b) {
            assert(boxes.delete(b), 'Dynamic object removed exactly once');
            for (let i = links.length - 1; i >= 0; --i)
                if (links[i].a === b || links[i].b === b) links.splice(i, 1);
        }
    };
    const context = { patcher: p, post(...args) { posts.push(args.join(' ')); }, outlet() {} };
    vm.createContext(context); vm.runInContext(nodeSource, context);
    const neural = method => [...boxes].find(x => x.maxclass === 'nn~' && x.args[1] === method);
    return { context, names, boxes, links, posts, neural };
}
function checkCreate(env, name, timbres, channels) {
    env.context.create(name, 8192, 0);
    const diffuse = env.neural('diffuse'), decoder = env.neural('decode');
    assert(diffuse && decoder);
    assert.equal([...env.boxes].filter(x => x.maxclass === 'nn~').length, 4);
    const output = env.links.filter(x => x.a === decoder);
    assert.equal(output.length, 2);
    assert.equal(output.find(x => x.b === env.names.rave_out).out, 0);
    assert.equal(output.find(x => x.b === env.names.rave_out_r).out, channels - 1);
    assert.deepEqual(env.names.audio_channels.lastMessage, ['int', channels]);
    const controls = env.links.filter(x => x.b === diffuse && x.a.maxclass === 'mc.unpack~');
    assert.equal(controls.length, timbres);
    controls.forEach((x, i) => { assert.equal(x.out, i); assert.equal(x.input, 16 + i); });
    for (let i = 0; i < 16; ++i)
        assert(env.links.some(x => x.b === diffuse && x.input === i && x.a.maxclass === 'sig~'));
    const latentLinks = env.links.filter(x => x.a === diffuse && x.b === decoder);
    assert.equal(latentLinks.length, 16);
    latentLinks.forEach((x, i) => { assert.equal(x.out, i); assert.equal(x.input, i); });
    assert.equal(env.links.filter(x => x.a === env.neural('map2latent')).length, timbres);
    assert.equal(env.links.filter(x => x.b === env.neural('latent2map')).length, timbres);
    assert(!env.posts.some(x => x.includes('[ERROR]')));
}
for (const [timbres, channels] of [[6, 1], [6, 2], [8, 1], [8, 2]]) {
    const env = environment(); checkCreate(env, (channels === 1 ? 'mono' : 'stereo') + timbres, timbres, channels);
    const firstObjects = [...env.boxes];
    checkCreate(env, (channels === 1 ? 'stereo' : 'mono') + timbres, timbres, 3 - channels);
    firstObjects.forEach(b => assert(!env.boxes.has(b), 'Replacing a model removes all prior dynamic boxes'));
    env.context.delete_message();
    assert.equal(env.boxes.size, 0); assert.equal(env.links.length, 0);
    assert.deepEqual(env.names.audio_channels.lastMessage, ['int', 1]);
    env.context.delete_message(); // Repeated deletion must not remove stale boxes twice.
    checkCreate(env, (channels === 1 ? 'mono' : 'stereo') + timbres, timbres, channels);
    env.context.create('off', 8192, 0);
    assert.equal(env.boxes.size, 0); assert.equal(env.links.length, 0);
}
for (const type of ['array', 'string']) checkCreate(environment(type), 'stereo6', 6, 2);
for (const bad of ['badChannels', 'badCodec', 'badTimbre', 'badMap']) {
    const env = environment(); env.context.create(bad, 8192, 0);
    assert.equal(env.boxes.size, 0); assert.equal(env.links.length, 0);
    assert(env.posts.some(x => x.includes('[ERROR]')));
}
// A newer script remains usable by a legacy mono device; stereo needs both buses.
{
    const env = environment(); delete env.names.rave_out_r;
    env.context.create('mono6', 8192, 0); assert(env.neural('decode'));
    env.context.create('stereo6', 8192, 0);
    assert.equal(env.boxes.size, 0); assert(env.posts.some(x => x.includes('updated MIDI device')));
}

assert.equal(binary.subarray(0, 4).toString(), 'ampf');
assert.equal(binary.subarray(8, 16).toString(), 'iiiiptch');
assert.equal(binary.readUInt32LE(16), binary.length - 20);
assert.equal(binary[binary.length - 1], 0);
const addedIds = patch.boxes.filter(x => x.box.id.startsWith('obj-after-midi-')).map(x => x.box.id);
assert.equal(addedIds.length, 7);
for (const id of addedIds) assert(!(id in patch.parameters), 'New routing boxes add no Live parameters');
if (baseline) {
    assert.deepEqual(binary.subarray(0, 16), baselineBinary.subarray(0, 16));
    assert.deepEqual(patch.parameters, baseline.parameters, 'Live parameter identities/storage must not change');
    for (const { box } of baseline.boxes) assert.deepEqual(byId[box.id], box, 'Existing Max box unchanged: ' + box.id);
    assert.equal(patch.boxes.length, baseline.boxes.length + 7);
    const edgeKey = l => JSON.stringify([l.source, l.destination]);
    const currentEdges = new Set(patch.lines.map(x => edgeKey(x.patchline)));
    for (const { patchline: l } of baseline.lines) {
        const removed = l.source[0] === 'obj-213' && l.destination[0] === 'obj-24' && l.destination[1] === 1;
        assert.equal(currentEdges.has(edgeKey(l)), !removed, 'Only old mono-to-right duplicate edge is removed');
    }
}
assert.deepEqual(byId['obj-after-midi-delay-r'].patcher, byId['obj-1422'].patcher,
    'The new audio delay uses the same Gen graph with independent state');
// Evaluate actual fixed patch edges with isolated L/R impulses, both selector
// states and audio/gate delay. This checks bus routing plus shared control flow;
// it intentionally does not pretend to execute live.adsr~ or the Max scheduler.
function renderFixed(audioChannels, left, right, audioDelay, gateDelay, envelopeOn, histories = {}) {
    const signals = { '---rave_out': left, '---rave_out_r': audioChannels === 1 ? left : right };
    const gate = left.map((_, i) => i % 4 / 3);
    const values = new Map();
    const controls = { 'obj-1426': audioDelay, 'obj-210': gateDelay, 'obj-212': envelopeOn ? 2 : 1, 'obj-after-midi-channels': audioChannels };
    function input(id, inlet) {
        const sources = patch.lines.map(x => x.patchline).filter(x => x.destination[0] === id && x.destination[1] === inlet);
        assert.equal(sources.length, 1, 'One source per evaluated signal/control inlet: ' + id + ':' + inlet);
        return output(sources[0].source[0]);
    }
    function output(id) {
        if (values.has(id)) return values.get(id);
        if (id in controls) return controls[id];
        const b = byId[id]; let v;
        if (b.text === 'r~ ---gate') v = gate;
        else if (b.text.startsWith('receive~ ')) v = signals[b.text.slice(9)];
        else if (b.text === 'gen~') {
            assert(b.patcher.boxes.some(x => x.box.text === 'delay 480000'));
            const audio = input(id, 0), delay = input(id, 1);
            v = audio.map((_, i) => i < delay ? ((histories[id] || [])[i] || 0) : audio[i - delay]);
        } else if (b.text === '*~ 1') {
            const a = input(id, 0), g = input(id, 1); v = a.map((x, i) => x * g[i]);
        } else if (b.text.startsWith('selector~ 2')) v = input(id, input(id, 0));
        else throw new Error('Unsupported evaluated object: ' + id + ' ' + b.text);
        values.set(id, v); return v;
    }
    return [input('obj-24', 0), input('obj-24', 1)];
}
for (const channels of [1, 2]) for (const audioDelay of [0, 3]) for (const envelope of [false, true]) {
    const left = [1, 0, -2, 0, 3, 0, -4, 0, 5, 0, -6, 0];
    const right = [0, 7, 0, -8, 0, 9, 0, -10, 0, 11, 0, -12];
    const actual = renderFixed(channels, left, right, audioDelay, 2, envelope);
    const expected = [left, channels === 1 ? left : right].map(signal => signal.map((_, i) =>
        (i < audioDelay ? 0 : signal[i - audioDelay]) * (envelope ? (i < 2 ? 0 : (i - 2) % 4 / 3) : 1)));
    assert.deepEqual(actual, expected, 'Independent L/R audio delay and shared optional gate');
}
// Model replacement does not erase fixed Gen histories. Mono must choose the
// final left branch, including retained tails, rather than a second delay.
for (const envelope of [false, true]) {
    const zeros = Array(12).fill(0);
    const histories = { 'obj-1422': [.81, .82, .83], 'obj-after-midi-delay-r': [-.91, -.92, -.93] };
    const mono = renderFixed(1, zeros, zeros, 3, 0, envelope, histories);
    assert.deepEqual(mono[1], mono[0], 'Mono replacement duplicates the final left output despite unequal retained histories');
    const stereo = renderFixed(2, zeros, zeros, 3, 0, envelope, histories);
    assert.notDeepEqual(stereo[1], stereo[0], 'Stereo retains independent delay history');
}
console.log('PASS: MIDI mono/stereo and six/eight timbre dynamic wiring; invalid layouts; replacement/teardown; no new routing parameters; fixed L/R/envelope/delay graph.' + (baseline ? ' Original boxes and Live parameters match the baseline.' : '') + ' Max runtime not exercised.');
