"""Derive de-identified refund inputs from UCI Online Retail II cancellations.

Source: "Online Retail II" (UCI Machine Learning Repository, dataset 502),
real transactions of a UK online retailer from 2009-12-01 to 2011-12-09,
licensed CC BY 4.0. Every cancellation invoice (invoice numbers starting with
``C``) is treated as one real refund request:

- ``order.ageDays``   whole days from the customer's most recent earlier
                      positive invoice to the cancellation invoice;
- ``order.total``     that earlier invoice's total (quantity times price,
                      rounded to cents);
- ``customer.priorRefunds``  the customer's cancellation invoices strictly
                      before this one;
- ``customer.tier``   ``enterprise`` when the customer's total positive spend
                      ranks in the top fifth of customers, else ``standard``;
- ``order.status``    ``paid`` for every extracted row; a later, documented
                      judge step may set ``fraudulent`` on a hash-selected subset.

No customer identifier, invoice number, country, or product appears in the
output. Rows are keyed by a salted digest of the cancellation invoice number so
the extraction is reproducible without being reversible from the output alone.

Usage::

    PYTHONPATH=.:trainer/src:model/src:.python-packages python -m \\
        benchmarks.refund.program.extract_uci_refund_inputs \\
        --archive online_retail_ii.zip --output candidates.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SOURCE = {
    "name": "Online Retail II",
    "publisher": "UCI Machine Learning Repository",
    "url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
    "license": "CC BY 4.0",
    "citation": "Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository.",
}
EXTRACTION_VERSION = 1
KEY_SALT = "semantscript.refund-benchmark.uci-online-retail-ii/v1"
ENTERPRISE_SPEND_QUANTILE = 0.8
SHEETS = ("Year 2009-2010", "Year 2010-2011")


def extract_candidates(archive_path: str | Path) -> dict[str, Any]:
    """Return the candidate rows and the source provenance for one archive."""

    import pandas as pd

    archive = Path(archive_path)
    archive_bytes = archive.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    with zipfile.ZipFile(archive) as bundle:
        members = [member for member in bundle.namelist() if member.lower().endswith(".xlsx")]
        if len(members) != 1:
            raise ValueError("archive must contain exactly one workbook")
        with bundle.open(members[0]) as workbook:
            frames = [
                pd.read_excel(workbook, sheet_name=sheet, engine="openpyxl") for sheet in SHEETS
            ]
    rows = pd.concat(frames, ignore_index=True)
    rows = rows.dropna(subset=["Customer ID"])
    rows["Invoice"] = rows["Invoice"].astype(str)
    rows["amount"] = rows["Quantity"].astype(float) * rows["Price"].astype(float)
    invoices = (
        rows.groupby("Invoice")
        .agg(
            customer=("Customer ID", "first"),
            date=("InvoiceDate", "min"),
            amount=("amount", "sum"),
        )
        .reset_index()
    )
    invoices["cancel"] = invoices["Invoice"].str.startswith("C")
    invoices["customer"] = invoices["customer"].astype(int)
    invoices = invoices.sort_values(["customer", "date", "Invoice"]).reset_index(drop=True)

    spend = invoices[~invoices["cancel"]].groupby("customer")["amount"].sum()
    threshold = float(spend.quantile(ENTERPRISE_SPEND_QUANTILE))
    enterprise = {int(customer) for customer, total in spend.items() if total >= threshold}

    candidates: list[dict[str, Any]] = []
    skipped_without_prior_order = 0
    for customer, group in invoices.groupby("customer", sort=True):
        positives: list[tuple[Any, float]] = []
        prior_refunds = 0
        for record in group.itertuples(index=False):
            if not record.cancel:
                if record.amount > 0:
                    positives.append((record.date, float(record.amount)))
                continue
            if not positives:
                skipped_without_prior_order += 1
                prior_refunds += 1
                continue
            order_date, order_total = positives[-1]
            age_days = int((record.date - order_date).days)
            candidates.append(
                {
                    "key": hashlib.sha256(f"{KEY_SALT}:{record.Invoice}".encode()).hexdigest()[:24],
                    "inputs": {
                        "customer": {
                            "priorRefunds": prior_refunds,
                            "tier": "enterprise" if int(customer) in enterprise else "standard",
                        },
                        "order": {
                            "ageDays": max(age_days, 0),
                            "status": "paid",
                            "total": round(order_total, 2),
                        },
                    },
                    "refundedAmount": round(abs(float(record.amount)), 2),
                }
            )
            prior_refunds += 1
    candidates.sort(key=lambda row: row["key"])
    return {
        "kind": "semantscript.refund-benchmark-uci-candidates",
        "extractionVersion": EXTRACTION_VERSION,
        "source": {**SOURCE, "archiveSha256": archive_sha256, "archiveBytes": len(archive_bytes)},
        "derivation": {
            "ageDays": "whole days from the customer's latest earlier positive invoice to the cancellation invoice",
            "total": "that earlier positive invoice's quantity times price, rounded to cents",
            "priorRefunds": "the customer's cancellation invoices strictly before this one",
            "tier": f"enterprise when total positive spend is at or above the {ENTERPRISE_SPEND_QUANTILE:g} customer quantile ({threshold:.2f})",
            "status": "paid for every extracted row",
            "key": "sha256 of a fixed salt plus the cancellation invoice number, first 24 hex characters",
        },
        "counts": {
            "invoices": len(invoices),
            "customers": int(invoices["customer"].nunique()),
            "cancellations": int(invoices["cancel"].sum()),
            "candidates": len(candidates),
            "skippedWithoutPriorOrder": skipped_without_prior_order,
        },
        "candidates": candidates,
    }


def summarize(document: dict[str, Any]) -> dict[str, Any]:
    rows = document["candidates"]
    ages = sorted(row["inputs"]["order"]["ageDays"] for row in rows)
    totals = sorted(row["inputs"]["order"]["total"] for row in rows)
    priors = collections.Counter(min(row["inputs"]["customer"]["priorRefunds"], 5) for row in rows)
    tiers = collections.Counter(row["inputs"]["customer"]["tier"] for row in rows)
    buckets = collections.Counter(
        "0-30" if age <= 30 else "31-60" if age <= 60 else "61-90" if age <= 90 else "91+"
        for age in ages
    )

    def quantile(values: Sequence[float], fraction: float) -> float:
        return values[int(fraction * (len(values) - 1))] if values else 0.0

    return {
        "counts": document["counts"],
        "ageDaysBuckets": dict(sorted(buckets.items())),
        "ageDaysQuantiles": [quantile(ages, f) for f in (0, 0.25, 0.5, 0.75, 0.95, 1)],
        "totalQuantiles": [quantile(totals, f) for f in (0, 0.25, 0.5, 0.75, 0.95, 1)],
        "priorRefundsCapped5": dict(sorted(priors.items())),
        "tiers": dict(sorted(tiers.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    document = extract_candidates(arguments.archive)
    Path(arguments.output).write_text(
        json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(summarize(document), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
