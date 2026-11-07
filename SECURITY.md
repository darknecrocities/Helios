# Security Policy

The Helios project is committed to delivering a secure, robust foundation model ecosystem. We treat vulnerability management with the highest priority and appreciate the responsible disclosure of any security concerns.

---

## 1. Supported Versions

Security fixes and patches are applied to active minor releases. Older unsupported branches will not receive backported security patches.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | Yes                |
| < 0.1.0 | No (Pre-release)   |

---

## 2. Reporting a Vulnerability

**Please do not report security vulnerabilities via public GitHub issues, pull requests, or public discussions.**

If you discover a vulnerability in Helios, please report it via one of the following secure channels:

1. **GitHub Private Vulnerability Reporting**: Submit an advisory directly via the **Security** tab of the [Helios repository](https://github.com/darknecrocities/Helios/security/advisories/new).
2. **Email**: Send an encrypted report to **`security@helios.ai`**.

### Report Details
To assist in rapid triage and remediation, please include:
- A descriptive summary of the vulnerability.
- Steps to reproduce or a minimal Proof-of-Concept (PoC) script.
- The component affected (`helios.model`, `helios.memory`, `helios.training`, `helios.tokenizer`, CLI).
- Assessment of potential impact and vector of exploitation.

---

## 3. Response Service Level Agreements (SLAs)

The Helios core security team adheres to the following timelines:

- **Initial Acknowledgment**: Within **48 hours** of report receipt.
- **Triage & Preliminary Assessment**: Within **7 calendar days**, confirming exploitability and severity rating (CVSS v3.1).
- **Remediation & Coordinated Disclosure**: Security patches are developed, tested, and released within **90 calendar days**, coordinated with the reporter prior to public disclosure.

---

## 4. Security Scope

### 4.1 In-Scope Components & Vulnerability Classes
- **Model Weight Deserialization**: Arbitrary code execution or unsafe memory access during model checkpoint or safetensors loading.
- **Hierarchical Memory Architecture (HMA) Isolation**: Cross-tenant data leaks, SQL injection in Episodic SQLite storage, or prompt-injection attacks traversing semantic retrieval buffers into execution contexts.
- **Tool Execution & Procedural Sandboxing**: Sandbox escapes, path traversals, or privilege escalation during procedural tool invocation.
- **Training Controller Integrity**: Tampering with atomic checkpointing (`.pt.tmp` -> `.pt`) or bypassing SHA-256 integrity checks during resumption.

### 4.2 Out-of-Scope Vulnerabilities
- Attacks requiring physical access to the host or root-level compromise of the operating system.
- Volumetric Distributed Denial of Service (DDoS) against public demonstration endpoints.
- Vulnerabilities in third-party dependencies unless directly exploitable through Helios standard configurations.
- Theoretical attacks without practical proof-of-concept or actionable impact demonstration.

---

## 5. Safe Harbor

Helios supports responsible security research conducted in good faith. If you conduct vulnerability research in compliance with this policy:
- We will not pursue legal action against you or request law enforcement intervention.
- We will work collaboratively with you to understand and resolve the vulnerability.
- We will publicly credit your contribution in our security advisory (unless you request anonymity).
