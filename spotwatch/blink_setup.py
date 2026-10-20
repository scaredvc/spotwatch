"""One-time Blink login.

    python -m spotwatch.blink_setup

Asks for your Blink email, password, and the 2FA PIN Blink sends you. Then it
lists the cameras on your account and saves a session file (tokens only, no
password) so spotwatch can log in by itself from then on. Optionally grabs a
test snapshot from each camera so you can see what it sees.

Nothing here prints or logs your password or tokens.
"""

import argparse
import asyncio
import logging
import re
from getpass import getpass
from pathlib import Path

from spotwatch.blink_source import save_session, take_snapshot


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "camera"


async def login(email: str, password: str):
    from aiohttp import ClientSession
    from blinkpy.auth import Auth, BlinkTwoFARequiredError
    from blinkpy.blinkpy import Blink

    session = ClientSession()
    blink = Blink(session=session)
    blink.auth = Auth({"username": email, "password": password}, no_prompt=True, session=session)
    try:
        try:
            ok = await blink.start()
        except BlinkTwoFARequiredError:
            pin = input("Blink sent you a verification PIN (email or text). Enter it: ").strip()
            ok = await blink.send_2fa_code(pin)
    except BaseException:
        await session.close()
        raise
    if not ok:
        await session.close()
        return None
    return blink


def print_cameras(blink) -> None:
    print(f"\nFound {len(blink.cameras)} camera(s):\n")
    for name, camera in blink.cameras.items():
        attrs = camera.attributes
        print(f"  {name}")
        print(f"    model/type:      {attrs.get('type')}")
        print(f"    online:          {camera.online}")
        print(f"    sync module:     {attrs.get('sync_module')}")
        print(f"    battery:         {attrs.get('battery')}  (level {attrs.get('battery_level')}, "
              f"voltage {attrs.get('battery_voltage')})")
        print(f"    wifi strength:   {attrs.get('wifi_strength')}")
        print(f"    sync signal:     {attrs.get('sync_signal_strength')}")
        print(f"    temperature (C): {attrs.get('temperature_c')}")
    print(
        "\nBlink doesn't report USB vs battery power directly; the battery fields\n"
        "above are what the API gives. Signal values are Blink's own 0-5 style scale.\n"
    )


async def grab_test_snapshots(blink, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in blink.cameras.keys():
        print(f"Snapping {name} (can take 10-30 seconds)...")
        image_bytes, reason, captured_at, _ = await take_snapshot(blink, name)
        if image_bytes is None:
            print(f"  no fresh image: {reason}")
            continue
        path = out_dir / f"test-{slug(name)}.jpg"
        path.write_bytes(image_bytes)
        when = f" taken {captured_at:%H:%M:%S}" if captured_at else ""
        print(f"  saved {path}{when}")


async def main_async(session_file: Path, out_dir: Path) -> int:
    print("Blink login. Your password is only used for this login and is not saved.\n")
    email = input("Blink email: ").strip()
    password = getpass("Blink password: ")

    from aiohttp import ClientError

    try:
        blink = await login(email, password)
    except (OSError, ClientError) as error:  # DNS, connection refused, timeouts
        print(f"\nCouldn't reach Blink's servers ({type(error).__name__}). Check your internet and try again.")
        return 1
    if blink is None:
        print("\nLogin failed. Check the email/password/PIN and try again.")
        return 1

    try:
        save_session(blink, session_file)
        print(f"\nSaved session to {session_file} (tokens only, readable only by you).")
        print_cameras(blink)

        print("Add a camera to config.yaml like this, using the exact name above:\n")
        print("  - name: window-left")
        print("    source: blink")
        print('    blink_camera: "<camera name>"')
        print("    interval_seconds: 60\n")

        answer = input("Grab a test snapshot from each camera now? [y/N] ").strip().lower()
        if answer == "y":
            await grab_test_snapshots(blink, out_dir)
            save_session(blink, session_file)  # tokens may have rotated
    finally:
        await blink.auth.session.close()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Log in to Blink once and save a session file.")
    parser.add_argument("--session-file", default="blink_session.json")
    parser.add_argument("--out", default="frames", help="where test snapshots go")
    args = parser.parse_args()

    # Keep blinkpy quiet: its debug logs can include request details.
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main_async(Path(args.session_file), Path(args.out))))


if __name__ == "__main__":
    main()
