"""Render retained experiment records without running or changing a sorter."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--output-root',type=Path,required=True)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for root in a.roots:
        if not (root/'complete.json').is_file():raise ValueError(f'Incomplete experiment: {root}')
        for line in (root/'results.jsonl').read_text().splitlines():
            row=json.loads(line);row['root']=str(root)
            row['start']=row.get('start',60 if row.get('phase')=='screen' else 420)
            rows.append(row)
    (a.output_root/'results.json').write_text(json.dumps(rows,indent=2))
    lines=['# Accuracy search results','',f'{len(rows)} sorting runs; {sum(bool(r["exit_code"]) for r in rows)} failed.','']
    for start in sorted({r['start'] for r in rows}):
        lines += [f'## {start}–{start+30} seconds','',
                  '| Configuration | Recovered units | Precision % | Recall % | F1 % | Splits / merges |',
                  '|---|---:|---:|---:|---:|---:|']
        for r in [r for r in rows if r['start']==start]:
            if r['exit_code']:lines.append(f'| {r["name"]} | FAILED | | | | |');continue
            lines.append(f'| {r["name"]} | {r["recovered_units_iou_0p8"]} | {100*r["spike_precision"]:.3f} | {100*r["spike_recall"]:.3f} | {100*r["f1"]:.3f} | {r["split_gt_units"]} / {r["merged_predicted_units"]} |')
        lines.append('')
    (a.output_root/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for ax,start in zip(axes,(60,420,480)):
        selected=[r for r in rows if r['start']==start and not r['exit_code']]
        scatter=ax.scatter([100*r['spike_precision'] for r in selected],[100*r['spike_recall'] for r in selected],
                   c=[r['recovered_units_iou_0p8'] for r in selected],cmap='viridis',vmin=120,vmax=180,s=55)
        labels={'reference':'Reference','smooth_score_0.75':'Smooth + score .75','score_0.8':'Score .80',
                'mask97_score07':'Mask97 + score .70','best_f1_score_floor_0.75':'Score .75',
                'pca_validation':'PCA split','fresh_reference':'Reference',
                'fresh_matched_score':'Raw, matched score','fresh_smooth':'Smooth, matched score'}
        for r in selected:
            if r['name'] in labels:
                offsets={'reference':(-40,-18),'pca_validation':(-35,24),
                         'mask97_score07':(5,34 if start==60 else 15),
                         'smooth_score_0.75':(-20,-26),'score_0.8':(-45,-18) if start==420 else (4,-18)}
                ax.annotate(labels[r['name']],(100*r['spike_precision'],100*r['spike_recall']),
                            xytext=offsets.get(r['name'],(4,7)),textcoords='offset points',fontsize=8,
                            arrowprops=dict(arrowstyle='-',color='.6',linewidth=.5))
        ax.set_title(f'{start}–{start+30} s'+(' development' if start==60 else ' validation'))
        ax.set_xlabel('Spike precision (%)');ax.set_ylabel('Spike recall (%)')
        ax.grid(alpha=.2);ax.margins(.25)
    fig.suptitle('TeraSort accuracy search: precision–recall tradeoffs\nColor represents recovered units (IoU ≥ 0.8); one 250-unit recording')
    fig.colorbar(scatter,ax=axes,label='Recovered units',shrink=.7)
    fig.savefig(a.output_root/'precision_recall.png',dpi=160)
    plt.close(fig)
    print(json.dumps(dict(runs=len(rows),development=sum(r['start']==60 for r in rows),
                         validation=sum(r['start']!=60 for r in rows),output=str(a.output_root))))


if __name__=='__main__':main()
