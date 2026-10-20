"""Optional dedicated local Ollama server. Works on Windows, macOS and Linux."""
import argparse
import os
import platform
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=11435)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    executable = shutil.which('ollama')
    if not executable:
        parser.error('Ollama executable is not on PATH')
    env = dict(os.environ, OLLAMA_HOST=f'127.0.0.1:{args.port}')
    process = subprocess.Popen([executable, 'serve'], env=env,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    url = f'http://127.0.0.1:{args.port}'
    print('Use these settings in a second terminal before running the experiment:', flush=True)
    if platform.system() == 'Windows':
        print(f"$env:OLLAMA_URL='{url}'\n$env:OLLAMA_SERVER_PID='{process.pid}'", flush=True)
    else:
        print(f"export OLLAMA_URL='{url}'\nexport OLLAMA_SERVER_PID='{process.pid}'", flush=True)
    try:
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if process.returncode and process.returncode > 0:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    main()
