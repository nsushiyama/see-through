"""Update fields in AGENT/STATE.md.  Usage:
python AGENT/tools/set_state.py KEY "value" [KEY "value" ...]
KEY in STATUS CURRENT_PHASE LAST_COMPLETED_TASK CURRENT_TASK NEXT_TASK BLOCKERS LAST_GOOD_COMMIT TEST_STATUS.
A field's value is everything between 'KEY:' and the next blank line."""
import re, sys, os.path as osp
p = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'STATE.md')
s = open(p, encoding='utf-8').read()
a = sys.argv[1:]
for k, v in zip(a[::2], a[1::2]):
    pat = re.compile(rf'^{k}:.*?(?=\n\n)', re.S | re.M)
    if k == 'STATUS':
        rep = f'STATUS: {v}'
    else:
        rep = f'{k}:\n{v}'
    s, n = pat.subn(lambda m: rep, s, count=1)
    assert n == 1, k
open(p, 'w', encoding='utf-8').write(s)
