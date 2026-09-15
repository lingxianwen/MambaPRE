"""Read-only result audit and post-hoc unseen-label macro F1 aggregation."""
import json
from pathlib import Path
from aggregate_batch_invariant_final import SEEDS, read, summarize, guided_path, transformer_path

ROOT = Path(__file__).resolve().parents[1]
groups = {'unseen_protocol_labels': ['iec104', 'lon', 'modbus_delta'],
          'unseen_families_excluding_modbus_variant': ['iec104', 'lon']}
out = {'note': 'Post-hoc analysis of saved five-seed protocol summaries. Modbus-Delta is an unseen label but a related Modbus variant.', 'groups': {}, 'fins_counts': {}, 'long_envelope': {}}
for name, protocols in groups.items():
    result = {'protocols': protocols, 'models': {}}
    for model, fn in [('mambapre', guided_path), ('transformer', transformer_path)]:
        reports = [read(fn(seed, 'external_full')) for seed in SEEDS]
        values = [sum(r['by_protocol'][p]['boundary_f1'] for p in protocols)/len(protocols) for r in reports]
        result['models'][model] = summarize(values)
        result['message_counts'] = {p: reports[0]['by_protocol'][p]['n'] for p in protocols}
    a,b = result['models']['mambapre']['values'],result['models']['transformer']['values']
    result['paired_delta'] = summarize([x-y for x,y in zip(a,b)])
    out['groups'][name] = result
for split in ['calibration', 'test']:
    p=ROOT / f'data/processed/long_full_strict/{split}_long_full_capture_disjoint_novel.jsonl'
    rows=[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    out['fins_counts'][split]={'n':len(rows), 'groups':sorted(set(str(r.get('capture_id')) for r in rows))}
for model,fn in [('mambapre',guided_path),('transformer',transformer_path)]:
    reports=[read(fn(seed,'long_envelope'))['overall'] for seed in SEEDS]
    out['long_envelope'][model]={k:summarize([r[k] for r in reports]) for k in ['boundary_precision','boundary_recall','boundary_f1']}
p=ROOT/'results/revision/submission_final_checks.json'
p.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
