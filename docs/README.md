# docs/

Written material that explains the project, rather than code that runs it.

This folder holds:
- An architecture diagram of the full system (frontend, gateway, inference service, data stores, training cluster).
- A decisions log — short write-ups of the "why" behind non-obvious choices (e.g. why SAR instead of optical imagery, why flood extent requires differencing against a permanent-water layer, why ONNX for serving).
- The results narrative that ties the benchmark numbers in `benchmarks/RESULTS.md` into a written story, rather than leaving them as a bare table.

This is where "what did we build and why" lives, separate from "how does it run."
