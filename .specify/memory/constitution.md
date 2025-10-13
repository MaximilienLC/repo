<!--
Sync Impact Report
==================
Version change: [none] → 1.0.0
Modified principles: N/A (initial constitution)
Added sections: Core Principles (1 principle), Governance
Templates status:
  - ✅ plan-template.md: Constitution Check section ready
  - ✅ spec-template.md: Compatible with principle structure
  - ✅ tasks-template.md: Ready for principle-driven tasks
Follow-up TODOs: Additional principles to be added as needed
-->

# Project Constitution

## Core Principles

### I. PyTorch-First Computation

All numeral values (rank 1+ tensors) MUST be processed in PyTorch, with all operations being performed on the select GPUs.

**Rationale**: Ensures consistent GPU-accelerated computation across all tensor operations, preventing performance bottlenecks from mixed CPU/GPU execution.

## Governance

### Amendment Process
Constitution amendments require:
1. Written proposal documenting the change
2. Review of impact on existing code
3. Version bump following semantic versioning

### Versioning
- **MAJOR**: Backward incompatible changes to principles
- **MINOR**: New principles added
- **PATCH**: Clarifications and wording improvements

**Version**: 1.0.0 | **Ratified**: 2025-10-13 | **Last Amended**: 2025-10-13
