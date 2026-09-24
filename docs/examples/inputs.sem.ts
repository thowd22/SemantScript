// Interpolated inputs: bare identifiers only, with derived values named first.
import { sema } from "@semantscript/core";

interface Customer {
  createdAt: string;
  region: "eu" | "us" | "apac";
  tags: readonly string[];
}

export function assessRisk(customer: Customer, nowMs: number): "low" | "high" {
  // Property access and calls must be assigned to a const before interpolation.
  const accountAgeDays = Math.floor(
    (nowMs - Date.parse(customer.createdAt)) / 86_400_000,
  );
  return sema<"low" | "high">`
    Assess account risk.
    Customer: ${customer}
    Account age in days: ${accountAgeDays}
  `;
}
