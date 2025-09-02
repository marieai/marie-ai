import os

from marie.extract.validator.base import (
    ValidationContext,
    ValidationResult,
    ValidationSummary,
)
from marie.logging_core.predefined import default_logger as logger


def generate_validation_report(
    validation_summaries: list[ValidationSummary], output_dir: str
):
    """Generate a comprehensive validation report"""
    import json
    from datetime import datetime

    report = {
        "timestamp": datetime.now().isoformat(),
        "total_parsers": len(validation_summaries),
        "summary": {
            "parsers_passed": sum(1 for vs in validation_summaries if vs.overall_valid),
            "parsers_failed": sum(
                1 for vs in validation_summaries if not vs.overall_valid
            ),
            "total_errors": sum(vs.total_errors for vs in validation_summaries),
            "total_warnings": sum(vs.total_warnings for vs in validation_summaries),
            "total_execution_time": sum(
                vs.execution_time for vs in validation_summaries
            ),
        },
        "parser_results": [],
    }

    for vs in validation_summaries:
        parser_result = {
            "parser_name": vs.parser_name,
            "valid": vs.overall_valid,
            "errors": vs.total_errors,
            "warnings": vs.total_warnings,
            "execution_time": vs.execution_time,
            "validator_results": [],
        }

        for result in vs.results:
            validator_result = {
                "validator_name": result.validator_name,
                "valid": result.valid,
                "execution_time": result.execution_time,
                "errors": [
                    {"code": e.code, "message": e.message, "severity": e.severity}
                    for e in result.errors
                ],
                "warnings": [
                    {"code": w.code, "message": w.message} for w in result.warnings
                ],
                "metadata": result.metadata,
            }
            parser_result["validator_results"].append(validator_result)

        report["parser_results"].append(parser_result)

    report_file = os.path.join(output_dir, "validation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    markdown_report = generate_markdown_validation_report(report)
    with open(os.path.join(output_dir, "validation_report.md"), "w") as f:
        f.write(markdown_report)

    logger.info(f"Validation report saved to {report_file}")


def generate_markdown_validation_report(report: dict) -> str:
    """Generate a human-readable markdown validation report"""
    lines = [
        "# Validation Report",
        f"Generated: {report['timestamp']}\n",
        "## Summary",
        f"- **Total Parsers**: {report['total_parsers']}",
        f"- **Parsers Passed**: {report['summary']['parsers_passed']}",
        f"- **Parsers Failed**: {report['summary']['parsers_failed']}",
        f"- **Total Errors**: {report['summary']['total_errors']}",
        f"- **Total Warnings**: {report['summary']['total_warnings']}",
        f"- **Total Execution Time**: {report['summary']['total_execution_time']:.3f}s\n",
    ]

    # Status indicators
    if report['summary']['parsers_failed'] == 0:
        lines.append("🟢 **All parsers passed validation**\n")
    else:
        lines.append("🔴 **Some parsers failed validation**\n")

    lines.append("## Parser Results\n")

    for parser_result in report['parser_results']:
        status = "✅" if parser_result['valid'] else "❌"
        lines.append(f"### {status} {parser_result['parser_name']}")
        lines.append(f"- **Status**: {'PASS' if parser_result['valid'] else 'FAIL'}")
        lines.append(f"- **Errors**: {parser_result['errors']}")
        lines.append(f"- **Warnings**: {parser_result['warnings']}")
        lines.append(f"- **Execution Time**: {parser_result['execution_time']:.3f}s\n")

        if parser_result['validator_results']:
            lines.append("#### Validator Details")
            for vr in parser_result['validator_results']:
                v_status = "✅" if vr['valid'] else "❌"
                lines.append(
                    f"- **{v_status} {vr['validator_name']}**: {len(vr['errors'])} errors, {len(vr['warnings'])} warnings"
                )

                # Show errors
                for error in vr['errors']:
                    lines.append(f"  - ❌ `{error['code']}`: {error['message']}")

                # Show warnings
                for warning in vr['warnings']:
                    lines.append(f"  - ⚠️ `{warning['code']}`: {warning['message']}")

        lines.append("")

    return "\n".join(lines)
