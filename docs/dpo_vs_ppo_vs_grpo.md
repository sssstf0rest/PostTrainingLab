# DPO vs PPO vs GRPO

Placeholder — this document is written in **M19**, after all three have been
implemented and measured in M9, M10 and M13. It must compare, with measured numbers:

- models required in memory at once (policy / reference / reward / value)
- offline vs online data
- reward-model requirement
- value-network requirement
- stability and observed failure modes
- peak VRAM and tokens/sec
- KL behaviour
- benchmark deltas from a shared SFT checkpoint

Nothing is written here until those experiments exist. See Rule 10.
