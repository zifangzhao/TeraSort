import json
import pytest
from terasort.neuroscope import read_xml, configure_probe, generate_probe
from terasort.web import validate_request, JobManager


XML = '''<parameters><acquisitionSystem><nBits>16</nBits><nChannels>4</nChannels>
<samplingRate>20000</samplingRate><offset>0</offset></acquisitionSystem>
<anatomicalDescription><channelGroups><group><channel>0</channel><channel skip="1">1</channel></group>
<group><channel>2</channel><channel>3</channel></group></channelGroups></anatomicalDescription>
<spikeDetection><channelGroups><group><channels><channel>0</channel><channel>1</channel></channels></group>
<group><channels><channel>2</channel></channels></group></channelGroups></spikeDetection></parameters>'''


def fixture(tmp_path):
    path = tmp_path / 'amplifier.xml'
    path.write_text(XML)
    (tmp_path / 'amplifier.dat').write_bytes(bytes(80))
    probe = dict(chanMap=[2,0,3,1], xc=[20,0,30,10], yc=[2,0,3,1])
    p = tmp_path / 'probe.json'
    p.write_text(json.dumps(probe))
    return path, p, probe


def test_xml_settings_groups_skips_and_physical_mapping(tmp_path):
    path, _, probe = fixture(tmp_path)
    meta = read_xml(path,.195)
    assert meta['settings'] == dict(n_chan_bin=4,fs=20000.,scale=.195)
    assert meta['bad_channels'] == [1,3]
    mapped = configure_probe(meta,probe)
    assert mapped['chanMap'] == [2,0]
    assert mapped['xc'] == [20,0] and mapped['yc'] == [2,0]
    assert mapped['kcoords'] == [1,0]
    assert read_xml(path)['settings'].get('scale') is None
    assert meta['recording_candidates'] == [str(tmp_path/'amplifier.dat')]


@pytest.mark.parametrize('source', [XML.replace('<nBits>16','<nBits>32'),
    XML.replace('<channel>3','<channel>4'), XML.replace('<channel>3','<channel>2'),
    XML.replace('<samplingRate>20000','<samplingRate>nan'), '<IntanRHX/>',
    '<!DOCTYPE parameters [<!ENTITY x "foo">]>'+XML])
def test_invalid_metadata_rejected(tmp_path,source):
    path=tmp_path/'bad.xml'; path.write_text(source)
    with pytest.raises(ValueError): read_xml(path)


def test_xml_queue_snapshots_settings_and_probe(tmp_path,monkeypatch):
    path,p,_=fixture(tmp_path)
    request=dict(neuroscope_xml=str(path),gain_uv_per_count=.195,
                 filename=str(tmp_path/'amplifier.dat'),probe_json=str(p),
                 results_dir=str(tmp_path/'sort'))
    before=path.read_bytes()
    monkeypatch.setattr(JobManager,'_loop',lambda self: None)
    manager=JobManager(tmp_path/'state')
    job=manager.add(request)
    assert job['status']=='queued'
    saved=job['request']
    assert json.loads(open(saved['settings']).read())['fs']==20000
    assert json.loads(open(saved['probe_json']).read())['chanMap']==[2,0]
    assert path.read_bytes()==before
    assert saved['xml_import']['xml_sha256']==read_xml(path)['xml_sha256']
    assert not any(k.startswith('_xml') for k in saved)
    request['lfp_output']=str(tmp_path/'out.lfp')
    request.pop('gain_uv_per_count')
    with pytest.raises(ValueError,match='gain'): validate_request(request)


def test_probe_coordinates_required(tmp_path):
    path,_,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='coordinates'):
        configure_probe(read_xml(path),{'chanMap':[0,1,2,3]})


def test_automatic_probe_preserves_skipped_positions_and_adc_order(tmp_path,monkeypatch):
    path,_,_=fixture(tmp_path)
    # Anatomical order is not acquisition order; skipped slots still occupy space.
    path.write_text(XML.replace('<channel>0</channel><channel skip="1">1</channel>',
                                '<channel skip="1">1</channel><channel>0</channel>'))
    mapped=generate_probe(read_xml(path))
    assert mapped == dict(chanMap=[0,2],xc=[220,380],yc=[-40,-20],kcoords=[0,1],n_chan=2)
    monkeypatch.setattr(JobManager,'_loop',lambda self: None)
    manager=JobManager(tmp_path/'state')
    job=manager.add(dict(neuroscope_xml=str(path), filename=str(tmp_path/'amplifier.dat'),
                         results_dir=str(tmp_path/'sort')))
    assert json.loads(open(job['request']['probe_json']).read()) == mapped
    assert job['request']['xml_import']['probe_layout'].startswith('staggered')
