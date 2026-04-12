# Security Policy

## Supported Versions

Security fixes are provided for the latest released version line of HoloVec.

| Version | Supported |
| ------- | --------- |
| 0.3.x   | :white_check_mark: |
| < 0.3.0 | :x: |

## Reporting a Vulnerability

Please avoid posting exploit details in a public issue.

Preferred path:
- Use GitHub's private vulnerability reporting flow from the repository Security tab when it is
  available.

Fallback path:
- If private reporting is not available, open a regular issue with only enough detail to identify
  the affected area and request a private follow-up before sharing reproduction details.

Include:
- A short description of the issue
- Expected impact
- Affected versions or commits, if known
- Reproduction notes or proof of concept, if you have them
- Any suggested mitigation, if you have one

## Response Expectations

HoloVec is maintained on a best-effort basis. When possible, we will:
- acknowledge reports
- assess whether the issue is in scope for the project
- work toward a fix or mitigation
- coordinate public disclosure after a fix is available when that makes sense

## Scope

This policy covers:
- the `holovec/` library code
- project build and packaging configuration
- the documentation and example code shipped in this repository

Out of scope:
- third-party dependencies themselves
- user applications built on top of HoloVec
- deployment environments outside this repository

## General Guidance

HoloVec is a local computational library. It does not provide a network service or hosted control
plane, but callers should still treat untrusted input carefully and validate dimensions, file
paths, and resource usage in their own applications.

## Updates

Security-relevant fixes will be documented through normal release notes and, when appropriate,
GitHub Security Advisories.
