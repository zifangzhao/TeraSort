const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../src/terasort/webui/index.html'), 'utf8');
const handler = html.slice(html.indexOf('function renderPicker()'), html.indexOf('async function browse('));

for (const field of ['filenames', 'results_dir', 'stage_dir', 'lfp_output', 'neuroscope_xml', 'settings', 'probe_json']) {
    for (const [kind, target] of [['drive', 'T:\\'], ['directory', '\\\\server\\share\\folder']]) {
        test(`${field}: clicking ${kind} navigates without selecting or closing`, () => {
            const row = {dataset: {kind, path: target}};
            const nodes = new Map();
            const calls = [];
            let closed = false;
            const context = {
                lastBrowse: {entries: [{kind, path: target, name: target}]},
                pickField: field,
                selectedRecordings: new Set(),
                escapeHtml: s => String(s),
                document: {querySelectorAll: () => [row]},
                browse: p => calls.push(p),
                $: selector => {
                    if (!nodes.has(selector)) nodes.set(selector, {value: '', close: () => {closed = true;}});
                    return nodes.get(selector);
                },
            };
            vm.runInNewContext(handler + '\nrenderPicker();', context);
            row.onclick();
            assert.deepEqual(calls, [target]);
            assert.equal(context.selectedRecordings.size, 0);
            assert.equal(closed, false);
            assert.match(nodes.get('#picker-list').innerHTML, /<button type="button"/);
        });
    }
}

test('recording file click toggles selection without navigation', () => {
    const target = 'T:\\data\\amplifier.dat';
    const row = {dataset: {kind: 'file', path: target}};
    const nodes = new Map();
    const context = {
        lastBrowse: {entries: [{kind: 'file', path: target, name: 'amplifier.dat', bytes: 1024}]},
        pickField: 'filenames', selectedRecordings: new Set(), escapeHtml: String,
        document: {querySelectorAll: () => [row]},
        browse: () => assert.fail('File click must not navigate'),
        $: s => {if (!nodes.has(s)) nodes.set(s, {}); return nodes.get(s);},
    };
    vm.runInNewContext(handler + '\nrenderPicker();', context);
    row.onclick();
    assert.equal(context.selectedRecordings.has(target), true);
    row.onclick();
    assert.equal(context.selectedRecordings.size, 0);
});

const startPathHelper = html.slice(html.indexOf('function pickerStartPath('), html.indexOf('function renderPicker()'));
for (const [source, expected] of [
    ['T:\\data\\day1\\amplifier.dat', 'T:\\data\\day1\\'],
    ['\\\\server\\share\\day1\\amplifier.dat', '\\\\server\\share\\day1\\'],
    ['C:\\amplifier.dat', 'C:\\'],
    ['/data/day1/amplifier.dat', '/data/day1/'],
    ['/amplifier.dat', '/'],
]) {
    test(`XML browser starts in first recording folder: ${source}`, () => {
        const context = {sourcePaths: () => [source, '/other/later.dat'],
            $: () => ({value: '/old/location/session.xml'})};
        assert.equal(vm.runInNewContext(startPathHelper + "\npickerStartPath('neuroscope_xml')", context), expected);
    });
}
test('XML without a recording starts beside existing XML', () => {
    const context = {sourcePaths: () => [], $: () => ({value: '/data/session.xml'})};
    assert.equal(vm.runInNewContext(startPathHelper + "\npickerStartPath('neuroscope_xml')", context), '/data/');
});
test('empty output browser starts in recording folder', () => {
    const context = {sourcePaths: () => ['T:\\day1\\amplifier.dat'], $: () => ({value: ''})};
    assert.equal(vm.runInNewContext(startPathHelper + "\npickerStartPath('results_dir')", context), 'T:\\day1\\');
});

const phaseHelper = html.slice(html.indexOf('function phaseProgressMarkup('), html.indexOf('function showJob('));
test('three progress bars include phase ETAs and measured task progress', () => {
    const job = {status: 'running', stage: 'Final clustering', estimate_basis: 'estimated',
        phases: [
            {name: 'Preprocessing', percent: 100, status: 'completed', eta_seconds: 0},
            {name: 'Sorting', percent: 100, status: 'completed', eta_seconds: 0},
            {name: 'Post-processing', percent: 25, status: 'running', eta_seconds: 90},
        ], task_progress: {label: 'Clustering regions', completed: 4, total: 16, percent: 25, eta_seconds: 60}};
    const markup = vm.runInNewContext(phaseHelper + '\nphaseProgressMarkup(job)', {
        job, escapeHtml: String, duration: seconds => `${seconds}s`});
    assert.equal((markup.match(/class="phase-progress"/g) || []).length, 3);
    assert.match(markup, /ETA ~90s/);
    assert.match(markup, /4 \/ 16 \(25%\)/);
    assert.match(markup, /Task ETA ~60s/);
});
test('legacy worker does not invent measured percentages', () => {
    const markup = vm.runInNewContext(phaseHelper + '\nphaseProgressMarkup(job)', {
        job: {status: 'running', stage: 'Preprocessing'}, escapeHtml: String, duration: String});
    assert.match(markup, /Measuring/);
    assert.match(markup, /Detailed counters require a newly started worker/);
    assert.equal((markup.match(/class="phase-progress"/g) || []).length, 3);
});


test('legacy batch counter and ETA survive a page refresh', () => {
    const job = {status: 'running', stage: 'Batch processing', percent: 1.3,
        eta_seconds: 24540, log: '1118/89024 [03:38<6:49:00, 11.9it/s]'};
    const markup = vm.runInNewContext(phaseHelper + '\nphaseProgressMarkup(job)', {
        job, escapeHtml: String, duration: seconds => `${seconds}s`});
    assert.match(markup, /Current Kilosort pass/);
    assert.match(markup, /1\.3%/);
    assert.match(markup, /ETA ~24540s/);
    assert.equal((markup.match(/class="phase-progress"/g) || []).length, 4);
});
