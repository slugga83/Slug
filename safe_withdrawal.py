#!/usr/bin/env python3
"""Compute safe withdrawal amount for a bills checking account.

This script projects the balance of a specified account over a horizon and
returns how much can be withdrawn today without the balance dropping below
zero, given scheduled bills, paychecks, and transfers. Inputs can be provided
via a single JSON file or several CSV files in a Monarch/Tiller style format.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y")


@dataclass
class ScheduledEvent:
    """Represents a dated inflow or outflow."""

    when: date
    amount: float
    kind: str  # "inflow" or "outflow"
    description: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project an account balance and determine how much cash can be "
            "withdrawn today without the balance going negative."
        )
    )
    parser.add_argument("--input_json", type=Path, help="Path to combined JSON input file.")

    parser.add_argument("--transactions_csv", type=Path, help="CSV of transactions across accounts.")
    parser.add_argument("--accounts_csv", type=Path, help="CSV of current account balances.")
    parser.add_argument("--bills_csv", type=Path, help="CSV of upcoming bills.")
    parser.add_argument("--paychecks_csv", type=Path, help="CSV of expected paychecks.")
    parser.add_argument("--transfers_csv", type=Path, help="CSV of transfers into the account.")
    parser.add_argument("--account_name", help="Exact name of the account to simulate.")
    parser.add_argument(
        "--starting_balance",
        type=float,
        help="Starting balance for the account (used if no balance source available).",
    )
    parser.add_argument(
        "--horizon_days",
        type=int,
        default=90,
        help="Number of days (from today) to include in the projection (default: 90).",
    )
    parser.add_argument(
        "--today",
        type=str,
        help="Override today's date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--csv_out",
        type=Path,
        help="Optional path to write the daily projection table as CSV.",
    )

    args = parser.parse_args()

    if args.input_json:
        # JSON mode
        if any(
            getattr(args, field) is not None
            for field in (
                "transactions_csv",
                "accounts_csv",
                "bills_csv",
                "paychecks_csv",
                "transfers_csv",
                "account_name",
                "starting_balance",
            )
        ):
            parser.error("CSV-related arguments cannot be used with --input_json.")
    else:
        # CSV mode
        if args.transactions_csv is None:
            parser.error("--transactions_csv is required when --input_json is not provided.")
        if args.account_name is None:
            parser.error("--account_name is required when using CSV inputs.")
        if args.bills_csv is None:
            parser.error("--bills_csv is required when using CSV inputs.")

    return args


def parse_date(value: str) -> date:
    value = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date format: {value}")


def parse_amount(value: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = value.strip().replace("$", "").replace(",", "")
    negative = False
    if cleaned.startswith("(") and cleaned.endswith(")"):
        negative = True
        cleaned = cleaned[1:-1]
    if cleaned.startswith("-"):
        negative = True
        cleaned = cleaned[1:]
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    if cleaned == "":
        raise ValueError("Empty amount")
    amount = float(cleaned)
    return -amount if negative else amount


def parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    return value in {"true", "yes", "1", "y"}


def normalize_header(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file {path} is missing a header row.")
        rows = [row for row in reader]
    return rows


def extract_column(row: Dict[str, str], aliases: Iterable[str]) -> Optional[str]:
    for alias in aliases:
        if alias in row:
            return row[alias]
    return None


def map_headers(row: Dict[str, str]) -> Dict[str, str]:
    return {normalize_header(k): v for k, v in row.items()}


def get_column(row: Dict[str, str], aliases: Iterable[str]) -> Optional[str]:
    normalized = map_headers(row)
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def load_json_inputs(path: Path) -> Tuple[str, float, List[ScheduledEvent]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    try:
        account_name = payload["account_name"]
        starting_balance = float(payload["starting_balance"])
    except KeyError as exc:
        raise ValueError(f"JSON input missing required key: {exc}") from exc

    events: List[ScheduledEvent] = []

    for bill in payload.get("bills", []):
        if parse_bool(bill.get("paid", False)):
            continue
        due_date = parse_date(str(bill["due_date"]))
        amount = float(bill["amount"])
        events.append(ScheduledEvent(due_date, amount, "outflow", bill.get("payee")))

    for paycheck in payload.get("paychecks", []):
        pay_date = parse_date(str(paycheck["date"]))
        amount = float(paycheck["amount"])
        events.append(ScheduledEvent(pay_date, amount, "inflow", paycheck.get("description")))

    for transfer in payload.get("transfers", []):
        trans_date = parse_date(str(transfer["date"]))
        amount = float(transfer["amount"])
        events.append(ScheduledEvent(trans_date, amount, "inflow", transfer.get("description")))

    return account_name, starting_balance, events


def infer_sign(amount: float, category: Optional[str]) -> float:
    if amount < 0:
        return amount
    if category is None:
        return amount
    lowered = category.lower()
    income_indicators = {"income", "transfer in", "salary"}
    if any(token in lowered for token in income_indicators) or "paycheck" in lowered:
        return abs(amount)
    expense_keywords = {
        "expense",
        "utilities",
        "bill",
        "groceries",
        "dining",
        "food",
        "rent",
        "mortgage",
        "insurance",
        "payment",
        "transfer out",
        "loan",
        "credit card",
        "cc payment",
        "fuel",
        "gasoline",
        "shopping",
        "subscription",
    }
    if any(token in lowered for token in expense_keywords):
        return -abs(amount)
    return amount


def load_transactions(
    transactions_path: Path, account_name: str
) -> Tuple[List[Dict[str, str]], List[ScheduledEvent], Optional[float]]:
    rows = read_csv_rows(transactions_path)
    normalized_rows: List[Dict[str, str]] = [map_headers(row) for row in rows]

    date_aliases = {"date", "posted_date"}
    amount_aliases = {"amount"}
    account_aliases = {"account", "account_name"}
    description_aliases = {"description", "merchant", "payee", "memo"}
    category_aliases = {"category"}
    balance_aliases = {"balance", "current_balance"}

    filtered: List[Dict[str, str]] = []
    transfer_events: List[ScheduledEvent] = []
    latest_balance: Optional[Tuple[date, int, float]] = None

    for index, row in enumerate(normalized_rows):
        date_value = get_column(row, date_aliases)
        amount_value = get_column(row, amount_aliases)
        account_value = get_column(row, account_aliases)

        if not date_value or not amount_value:
            print(
                "Warning: skipping transaction row with missing date or amount:",
                row,
                file=sys.stderr,
            )
            continue

        if not account_value:
            continue

        if account_value.strip() != account_name:
            continue

        try:
            txn_date = parse_date(date_value)
        except Exception:
            continue
        try:
            amount = parse_amount(amount_value)
        except Exception:
            continue

        category_value = get_column(row, category_aliases)
        amount = infer_sign(amount, category_value)

        description_value = get_column(row, description_aliases) or ""

        filtered.append(
            {
                "date": txn_date,
                "amount": amount,
                "description": description_value,
                "category": category_value or "",
            }
        )

        balance_value = get_column(row, balance_aliases)
        if balance_value:
            try:
                balance = parse_amount(balance_value)
            except Exception:
                balance = None
            else:
                key = (txn_date, index)
                if balance is not None:
                    if latest_balance is None or key > (latest_balance[0], latest_balance[1]):
                        latest_balance = (txn_date, index, balance)

        if amount > 0:
            desc_lower = description_value.lower()
            cat_lower = (category_value or "").lower()
            if "transfer" in desc_lower or "xfer" in desc_lower or "transfer in" in cat_lower:
                transfer_events.append(
                    ScheduledEvent(txn_date, amount, "inflow", description_value)
                )

    balance_amount = latest_balance[2] if latest_balance else None
    return filtered, transfer_events, balance_amount


def load_accounts_balance(accounts_path: Path, account_name: str) -> float:
    rows = read_csv_rows(accounts_path)
    normalized_rows = [map_headers(row) for row in rows]
    account_aliases = {"account", "account_name"}
    balance_aliases = {"balance", "current_balance"}

    for row in normalized_rows:
        account_value = get_column(row, account_aliases)
        if account_value and account_value.strip() == account_name:
            balance_value = get_column(row, balance_aliases)
            if balance_value is None:
                continue
            return parse_amount(balance_value)
    raise ValueError(f"Account '{account_name}' not found in accounts CSV: {accounts_path}")


def load_bills(bills_path: Path) -> List[ScheduledEvent]:
    rows = read_csv_rows(bills_path)
    normalized = [map_headers(row) for row in rows]
    events: List[ScheduledEvent] = []
    for row in normalized:
        due_value = get_column(row, {"due_date"})
        amount_value = get_column(row, {"amount"})
        payee_value = get_column(row, {"payee", "description"})
        paid_value = get_column(row, {"paid"})
        if not due_value or not amount_value:
            continue
        if paid_value and parse_bool(paid_value):
            continue
        try:
            due = parse_date(due_value)
            amount = parse_amount(amount_value)
        except Exception:
            continue
        events.append(ScheduledEvent(due, abs(amount), "outflow", payee_value))
    return events


def load_simple_events(path: Optional[Path], kind: str) -> List[ScheduledEvent]:
    if path is None:
        return []
    rows = read_csv_rows(path)
    normalized = [map_headers(row) for row in rows]
    events: List[ScheduledEvent] = []
    for row in normalized:
        date_value = get_column(row, {"date"})
        amount_value = get_column(row, {"amount"})
        description_value = get_column(row, {"description"})
        if not date_value or not amount_value:
            continue
        try:
            when = parse_date(date_value)
            amount = abs(parse_amount(amount_value))
        except Exception:
            continue
        events.append(ScheduledEvent(when, amount, kind, description_value))
    return events


def build_daily_projection(
    today: date,
    horizon_days: int,
    starting_balance: float,
    events: Iterable[ScheduledEvent],
) -> Tuple[List[Tuple[date, float, float, float]], float, date]:
    ledger: Dict[date, Dict[str, float]] = {}
    for event in events:
        if event.when < today or event.when > today + timedelta(days=horizon_days):
            continue
        bucket = ledger.setdefault(event.when, {"inflow": 0.0, "outflow": 0.0})
        bucket[event.kind] += event.amount

    rows: List[Tuple[date, float, float, float]] = []
    running_balance = starting_balance
    min_balance = running_balance
    min_date = today

    for offset in range(horizon_days + 1):
        current_day = today + timedelta(days=offset)
        daily = ledger.get(current_day, {"inflow": 0.0, "outflow": 0.0})
        inflow = daily.get("inflow", 0.0)
        outflow = daily.get("outflow", 0.0)
        running_balance += inflow
        running_balance -= outflow
        rows.append((current_day, inflow, outflow, running_balance))
        if running_balance < min_balance:
            min_balance = running_balance
            min_date = current_day

    return rows, min_balance, min_date


def floor_to_cents(value: float) -> float:
    return math.floor(value * 100.0) / 100.0


def format_currency(value: float) -> str:
    return f"{value:.2f}"


def write_csv(path: Path, rows: List[Tuple[date, float, float, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["date", "inflows", "outflows", "running_balance"])
        for day, inflow, outflow, balance in rows:
            writer.writerow([day.isoformat(), f"{inflow:.2f}", f"{outflow:.2f}", f"{balance:.2f}"])


def main() -> None:
    args = parse_args()

    today = date.today()
    if args.today:
        try:
            today = parse_date(args.today)
        except ValueError as exc:
            raise SystemExit(f"Invalid --today value: {exc}")

    horizon_days = args.horizon_days
    if horizon_days < 0:
        raise SystemExit("--horizon_days must be non-negative")

    events: List[ScheduledEvent] = []
    account_name: Optional[str] = None
    starting_balance: Optional[float] = None

    if args.input_json:
        try:
            account_name, starting_balance, events = load_json_inputs(args.input_json)
        except Exception as exc:
            raise SystemExit(f"Failed to read JSON input: {exc}")
    else:
        account_name = args.account_name
        try:
            _, transfer_events, txn_balance = load_transactions(args.transactions_csv, account_name)
        except Exception as exc:
            raise SystemExit(f"Failed to read transactions CSV: {exc}")

        events.extend(transfer_events)

        try:
            bill_events = load_bills(args.bills_csv)
        except Exception as exc:
            raise SystemExit(f"Failed to read bills CSV: {exc}")
        events.extend(bill_events)

        events.extend(load_simple_events(args.paychecks_csv, "inflow"))
        events.extend(load_simple_events(args.transfers_csv, "inflow"))

        if args.accounts_csv:
            try:
                starting_balance = load_accounts_balance(args.accounts_csv, account_name)
            except Exception as exc:
                raise SystemExit(str(exc))
        elif txn_balance is not None:
            starting_balance = txn_balance
        elif args.starting_balance is not None:
            starting_balance = args.starting_balance
        else:
            raise SystemExit(
                "Unable to determine starting balance. Provide --accounts_csv, "
                "a Balance column in transactions CSV, or --starting_balance."
            )

    if account_name is None:
        raise SystemExit("Account name could not be determined from inputs.")
    if starting_balance is None:
        raise SystemExit("Starting balance could not be determined from inputs.")

    projection_rows, min_balance, min_date = build_daily_projection(
        today, horizon_days, starting_balance, events
    )

    safe_to_withdraw = max(0.0, floor_to_cents(min_balance))
    shortfall_needed = abs(min_balance) if min_balance < 0 else 0.0

    result_json = (
        "{"
        f"\"safe_to_withdraw\": {format_currency(safe_to_withdraw)}, "
        f"\"min_balance\": {format_currency(min_balance)}, "
        f"\"min_balance_date\": {json.dumps(min_date.isoformat())}, "
        f"\"starting_balance\": {format_currency(starting_balance)}, "
        f"\"horizon_days\": {horizon_days}, "
        f"\"shortfall_needed\": {format_currency(shortfall_needed)}, "
        f"\"account_name\": {json.dumps(account_name)}"
        "}"
    )

    print(result_json)

    print("DATE, INFLOWS, OUTFLOWS, RUNNING_BALANCE")
    for day, inflow, outflow, balance in projection_rows:
        print(
            f"{day.isoformat()}, {format_currency(inflow)}, "
            f"{format_currency(outflow)}, {format_currency(balance)}"
        )

    if args.csv_out:
        try:
            write_csv(args.csv_out, projection_rows)
        except Exception as exc:
            raise SystemExit(f"Failed to write CSV output: {exc}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        raise
    except Exception as exc:
        raise SystemExit(f"Unexpected error: {exc}")
