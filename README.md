# Agenthon 2026 — Track 1 (Coding) agent

Submission image for the [Agenthon 2026](https://www.agenthon.net/) NeurIPS competition, Track 1 (quantitative-finance coding tasks).

## How it works

`agent.py` implements the Track 1 CLI contract (`solve --task-dir <dir> --out <dir>`):

1. Reads the task's `instruction.md` and previews the mounted input files, mapping the paths named in the task (for example `/app/data/...`) to the files actually mounted under `/input`.
2. Asks the organizer-hosted House model (`$MODEL_ENDPOINT/v1/chat/completions`, bearer `$MODEL_TOKEN`, model `$MODEL_NAME`) for one self-contained Python script, continuing replies that hit the per-request output cap.
3. Runs the script with a timeout, checks that the deliverables exist and are well formed, and sends errors back for a corrected script, within the per-unit request and time budgets.
4. When the first version succeeds, asks for one audit against the specification and keeps the revised outputs only if they run cleanly.

The agent always exits 0 and logs to stderr. It makes no other network calls and ships no model weights.

## Image

Built for `linux/amd64` by the GitHub Actions workflow in `.github/workflows/publish.yml` and published to `ghcr.io/emamira02/agenthon-2026-t1`. The image carries the label `qfbench2.interface_version="2.0"` and declares no `VOLUME`.

## License

Apache-2.0. See [LICENSE](LICENSE).
