# Contributing

Thanks for looking. This is a small hobby project, so keep changes small and easy to review.

## Setup

```
uv sync            # or: pip install -r requirements.txt
uv run python -m pytest
```

Run pytest as `python -m pytest` from the repo root so the `spotwatch` package is importable.

## Before you open a pull request

- Tests pass, and new behaviour has a test. Tests use saved or synthetic frames; they never touch a real camera.
- One change per pull request, with a short description of what it does and why.
- Don't commit anything personal: `.env`, `blink_session.json`, your `config.yaml`, or photos of your street
  (`runs/`, `calibration/`, `frames/`, `recordings/`). They are gitignored; keep it that way.
- If you add a screenshot to the docs, blur license plates, faces and house numbers first.

## License

spotwatch is licensed under the GNU AGPL-3.0 (see `LICENSE`), mainly because the YOLO detector it uses,
`ultralytics`, is AGPL-3.0. By opening a pull request you agree that your contribution is released under the same
license, and that the maintainer may relicense it if the project later moves to a different detector.

## Reporting a problem

Say which camera type you use, what you expected, and what happened. Logs help; remove tokens and webhook URLs
before pasting them. A saved frame helps a lot for detection problems, but only share it if you're comfortable
with what's in it.
