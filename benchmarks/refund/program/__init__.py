"""Compiler-to-runtime helpers for the refund benchmark."""

from .claude_cli_teacher import (
    CLAUDE_CLI_DATA_CLASSIFICATION,
    CLAUDE_CLI_MODEL,
    CLAUDE_CLI_PROTOCOL,
    CLAUDE_CLI_VERSION,
    ClaudeCliTeacherConfig,
    ClaudeCliTeacherProvenance,
    ClaudeCliTrainingTeacher,
)
from .pipeline import (
    CompiledRefundProgram,
    FinalBenchmarkDatasetIdentity,
    RefundPipelineError,
    RefundPipelineResult,
    ReleaseVerificationRecord,
    build_release_verification_record,
    compile_refund_program,
    derive_refund_artifact_training_key_sha256,
    derive_training_input_ledger,
    parse_release_verification_record,
    run_refund_pipeline,
    run_refund_runtime,
)

__all__ = [
    "CLAUDE_CLI_DATA_CLASSIFICATION",
    "CLAUDE_CLI_MODEL",
    "CLAUDE_CLI_PROTOCOL",
    "CLAUDE_CLI_VERSION",
    "ClaudeCliTeacherConfig",
    "ClaudeCliTeacherProvenance",
    "ClaudeCliTrainingTeacher",
    "CompiledRefundProgram",
    "FinalBenchmarkDatasetIdentity",
    "RefundPipelineError",
    "RefundPipelineResult",
    "ReleaseVerificationRecord",
    "build_release_verification_record",
    "compile_refund_program",
    "derive_refund_artifact_training_key_sha256",
    "derive_training_input_ledger",
    "parse_release_verification_record",
    "run_refund_pipeline",
    "run_refund_runtime",
]
