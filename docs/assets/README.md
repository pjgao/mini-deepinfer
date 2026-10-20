# README diagrams

`model-architecture.png` was generated with GPT Image and checked against
`mini_deepinfer/model.py` and `config.py`. Its prompt is in
`model-architecture.prompt.txt`. The boxes represent operations, not tensor shapes;
the residual formulas describe the two pre-norm sublayers.

`scheduler.svg` is rendered from `scheduler.mmd`. Regenerate it from the repository root:

```bash
npx --yes --package @mermaid-js/mermaid-cli@12.0.0 mmdc -i docs/assets/scheduler.mmd -o docs/assets/scheduler.svg -c docs/assets/mermaid-config.json -b white --size 1400 --no-font-embed
```

The configuration uses native SVG text instead of HTML labels so the diagram can
be displayed as a regular README image. The exported SVG does not embed font files.
