# Security Policy

## Supported Versions

We release security updates for the following versions of HoloVec:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

We take the security of HoloVec seriously. If you discover a security vulnerability, please follow these steps:

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please email security details to:

**security@twistient.com**

Include:

- **Description**: Clear description of the vulnerability
- **Impact**: What could an attacker accomplish?
- **Reproduction**: Steps to reproduce the issue
- **Affected versions**: Which versions are affected
- **Suggested fix**: If you have one (optional)
- **Credit preferences**: How you'd like to be credited (if desired)

### What to Expect

1. **Acknowledgment**: We will acknowledge receipt within 48 hours
2. **Assessment**: We will assess the vulnerability and determine severity
3. **Updates**: We will keep you informed of our progress
4. **Resolution**: We will work on a fix and coordinate disclosure
5. **Credit**: We will credit you in the security advisory (if you wish)

### Disclosure Timeline

- **Day 0**: You report the vulnerability
- **Day 1-2**: We acknowledge and begin assessment
- **Day 3-7**: We develop and test a fix
- **Day 7-14**: We prepare a security release
- **Day 14+**: Public disclosure after fix is released

We aim to disclose vulnerabilities within 90 days of the initial report.

## Security Considerations

### Data Sensitivity

HoloVec is a computational library and does not:
- Collect user data
- Make network requests
- Store credentials
- Execute arbitrary code from untrusted sources

### Dependency Security

HoloVec has minimal dependencies:
- **Core**: Only NumPy (required)
- **Optional**: PyTorch, JAX (for GPU/JIT support)
- **Development**: pytest, black, ruff, mypy

We monitor dependencies for security advisories and update promptly.

### Input Validation

When using HoloVec:
- Validate numeric inputs to prevent numeric overflow
- Be cautious with user-supplied dimensions (memory consumption)
- Sanitize file paths when loading/saving codebooks

### Best Practices

1. **Keep Updated**: Use the latest version of HoloVec
2. **Pin Dependencies**: Use exact versions in production
3. **Audit Dependencies**: Regularly check dependency security
4. **Validate Input**: Always validate user input before encoding
5. **Limit Resources**: Set reasonable limits on vector dimensions

## Scope

This security policy covers:
- The HoloVec library code (holovec/ directory)
- Example code (examples/ directory)
- Documentation (docs/ directory)
- Build and distribution (setup, pyproject.toml)

Out of scope:
- Third-party dependencies (report to respective projects)
- User code that uses HoloVec
- Deployment environments

## Security Updates

Security updates will be announced via:
- GitHub Security Advisories
- Release notes in CHANGELOG.md
- Tagged releases with security fixes

Subscribe to repository notifications to stay informed.

## Contact

- **Security issues**: security@twistient.com
- **General issues**: https://github.com/Twistient/HoloVec/issues
- **Private concerns**: brodie@twistient.com

## Acknowledgments

We appreciate security researchers who responsibly disclose vulnerabilities.

Past security contributors:
- (None yet - be the first!)

---

Thank you for helping keep HoloVec and its users safe!
