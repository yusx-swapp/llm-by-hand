# Local editor assets

`monaco/` contains Monaco Editor **0.56.0**, copied from the project's previously installed `monaco-editor` package (`min/vs/`). Its original `LICENSE` and `ThirdPartyNotices.txt` are included without modification.

The AMD loader, Python tokenizer, CSS, fonts and web workers are all served by the local FastAPI server. No CDN or npm process is involved at runtime.

To upgrade, replace this directory with the matching distribution and licenses from a single Monaco version, then verify editor startup, Python highlighting, worker requests and the textarea fallback. Do not mix assets from different versions.
