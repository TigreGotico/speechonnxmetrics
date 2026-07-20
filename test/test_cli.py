"""CLI: `speechonnxmetrics list` / `speechonnxmetrics score`."""
from __future__ import annotations

import json
import os

from speechonnxmetrics.cli import main

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "audio")
SOURCE_WAV = os.path.join(FIXTURES, "source.wav")


def test_list_table(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "stoi" in out
    assert "wer" in out


def test_list_json(capsys):
    assert main(["list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    names = {row["name"] for row in data}
    assert "stoi" in names


def test_score_happy_path_table(capsys):
    rc = main(["score", SOURCE_WAV, "--ref", SOURCE_WAV, "--metrics", "stoi"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stoi" in out
    assert SOURCE_WAV in out


def test_score_happy_path_json(capsys):
    rc = main(["score", SOURCE_WAV, "--ref", SOURCE_WAV, "--metrics", "stoi,estoi", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert set(data[SOURCE_WAV]) == {"stoi", "estoi"}


def test_score_unknown_metric_exits_nonzero(capsys):
    rc = main(["score", SOURCE_WAV, "--metrics", "not-a-metric"])
    assert rc != 0
    assert "error" in capsys.readouterr().err


def test_score_missing_ref_for_intrusive_exits_nonzero(capsys):
    rc = main(["score", SOURCE_WAV, "--metrics", "stoi"])
    assert rc != 0
    assert "error" in capsys.readouterr().err


def test_score_bad_audio_path_reports_error_but_exits_nonzero(capsys):
    rc = main(["score", "/no/such/file.wav", "--ref", SOURCE_WAV, "--metrics", "stoi"])
    assert rc != 0
    assert "error" in capsys.readouterr().err
