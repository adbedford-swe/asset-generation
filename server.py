"""Start the assetgen server: the HTTP API and the web UI on one port.

    python server.py                       # http://127.0.0.1:7860
    python server.py --host 0.0.0.0 --image-backend sdcpp --image-url http://127.0.0.1:1234

Same as `assetgen serve`; `python server.py --help` lists every option.
"""
from assetgen.server import main

if __name__ == "__main__":
    main()
