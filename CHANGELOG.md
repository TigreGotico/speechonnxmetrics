# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses semantic
versioning driven by conventional commits.

## [Unreleased]

### Added
- Repo skeleton: packaging, CI wiring, and shared numpy DSP primitives
  (`_dsp.stft`, `_dsp.fbank`, `_dsp.resample`) reused across the planned metric
  families (no-reference MOS, intrusive, ASR, speaker similarity).
