import os

import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("LIVE_DEV_PORT", "8091"))
    print(f"the live worker will listen at http://127.0.0.1:{port} (ws /api/live)")
    uvicorn.run("voiceya.live.app:app", host="127.0.0.1", port=port, ws_max_size=1 << 20)
