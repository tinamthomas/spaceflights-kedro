<!-- SPECKIT START -->
**Current Feature**: Aircraft Range Prediction Neural Network Pipeline (001-aircraft-range-prediction)

**Key Documents**:
- Implementation Plan: [specs/001-aircraft-range-prediction/plan.md](../specs/001-aircraft-range-prediction/plan.md)
- Feature Specification: [specs/001-aircraft-range-prediction/spec.md](../specs/001-aircraft-range-prediction/spec.md)
- Data Model: [specs/001-aircraft-range-prediction/data-model.md](../specs/001-aircraft-range-prediction/data-model.md)
- Model I/O Contract: [specs/001-aircraft-range-prediction/contracts/model_io_contract.md](../specs/001-aircraft-range-prediction/contracts/model_io_contract.md)
- Quickstart Guide: [specs/001-aircraft-range-prediction/quickstart.md](../specs/001-aircraft-range-prediction/quickstart.md)

**Constitution**: [.specify/memory/constitution.md](./.specify/memory/constitution.md) (Code Quality & Data Excellence)
- Test-First Development (TDD mandatory, ≥80% coverage)
- Kedro Pipeline Architecture (pure functions, declarative)
- Data Quality & Lineage Standards (schema versioning, validation)

**Tech Stack**: Python 3.10+, PyTorch 2.0+, Kedro 1.4.0, scikit-learn, pytest

**Next Step**: Run `/speckit.tasks` to generate implementation task breakdown (Phase 2)
<!-- SPECKIT END -->

