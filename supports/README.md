Mount or copy a support bank here (`bank.json` + the images/masks it references), or point
`SUPPORTS_DIR` in `.env` at one. Support images are site imagery and are never committed.

YOLOE (default) turns every class's examples into one visual prompt; `detector.classes: auto` uses every
class with at least one usable example. SAM3 needs supports only for `visual` / `text+visual` prompts.

Manifest (`triffid.ugv.support-bank.v1`; the UAV `triffid.uav.support-bank.v1` layout also works),
paths relative to `bank.json`. Empty slots are skipped, so classes may have any number of examples:

    {"schema_version": "triffid.ugv.support-bank.v1",
     "classes": {"citizen": [{"id": "task_5", "image": "images/a.jpg", "mask": "masks/a_citizen.png"}, ...]}}

Masks are single-channel; foreground is any non-zero pixel, or `foreground_value` for indexed masks.
The visual-prompt cache in `state/` is keyed by the support files, so editing the bank re-embeds it.
