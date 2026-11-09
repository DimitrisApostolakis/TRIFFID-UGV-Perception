"""Make MASA honour its `with_cats` flag during association (upstream computes but never applies it).

With external detections from several classes, a person must not be matched to a vehicle track just
because their appearance embeddings are close. The patch masks cross-class match scores before the
greedy assignment. Idempotent; fails loudly if the upstream source no longer matches.
"""
from pathlib import Path
import sys

MARKER = '# TRIFFID: respect the category gate for heterogeneous objects.'
ANCHOR = '            # track according to match_scores\n'
GATE = (
    f'            {MARKER}\n'
    '            if self.with_cats:\n'
    '                match_scores = match_scores.masked_fill(\n'
    '                    labels[:, None] != memo_labels[None, :], -1.0\n'
    '                )\n\n'
)


def main(repo):
    path = Path(repo) / 'masa/models/tracker/masa_tao_tracker.py'
    source = path.read_text()
    if MARKER in source:
        print('MASA category gate already applied')
        return
    if source.count(ANCHOR) != 1:
        raise SystemExit(f'Unexpected MASA source layout in {path}; refusing to patch')
    path.write_text(source.replace(ANCHOR, GATE + ANCHOR))
    print('MASA category gate applied')


if __name__ == '__main__':
    main(sys.argv[1])
