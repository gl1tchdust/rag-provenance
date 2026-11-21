# Northwind Logistics - Approval Flow

## Who approves what

| Amount | Approver |
|---|---|
| Up to INR 10,000 | Line manager |
| INR 10,001 to INR 75,000 | Department head |
| Above INR 75,000 | Finance director and department head, both required |

## Sequence

Approvals are sequential, not parallel. A claim that skips the line manager is
rejected at the finance stage regardless of how senior the second approver is.

Approvers cannot approve their own claims. Where a line manager is also the
claimant, the claim goes to the department head directly.

## Delegation

An approver on leave may delegate authority for a named period. Delegation is
recorded in the expense system and expires automatically. Verbal delegation is
not recognised.
