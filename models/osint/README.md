# OSINT Modules

This directory contains the OSINT (Open Source Intelligence) modules of the
sofia-forensics prototype. These modules are **architecturally separate** from
the deepfake detectors in `models/image/`, `models/video/`, and (eventually)
`models/audio/` and `models/text/`.

## Design principle

The separation between OSINT modules and deepfake detectors is a deliberate
architectural choice, requested by the thesis supervisor and formalized in
ADR-003 (`docs/adr/0003-tier1-osint-and-sofia-mapper.md`).

OSINT modules and deepfake detectors:

- run in separate containers
- expose the same interface (`GET /health` and `POST /predict`)
- never call each other directly — the API gateway orchestrates them
- never share Python code or libraries beyond the standard SDK base classes

This separation ensures:

- single responsibility for each module
- independent evolution and testing
- clear contribution boundaries between inherited DeepSafe components
  and original thesis work

## Modules

| Module | Purpose | Status |
|---|---|---|
| `ingestion/` | File hashing (MD5, SHA-256, perceptual), keyframe extraction from videos | Planned |
| `metadata_forensics/` | EXIF/XMP extraction, ELA analysis, JPEG quantization analysis, generator fingerprint detection | Planned |
| `reverse_search/` | Reverse image search across public engines, normalized results | Planned |
| `first_appearance/` | Timeline ordering of reverse search results, identification of earliest known appearance | Planned |

## Contribution provenance

All code under `models/osint/` is original thesis work, not inherited from
DeepSafe. The DeepSafe project (MIT license) provides the orchestration
infrastructure (API gateway, microservices architecture, meta-learner) but
contains no OSINT components.
