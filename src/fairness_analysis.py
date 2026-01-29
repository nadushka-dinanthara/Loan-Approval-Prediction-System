import pandas as pd


def approval_rate_by_group(df, group_col, prediction_col):
    """
    Calculate approval rate per group.
    Approval rate = mean of predicted approvals (1s)
    """
    rates = (
        df.groupby(group_col)[prediction_col]
          .mean()
          .reset_index()
          .rename(columns={prediction_col: "approval_rate"})
    )
    return rates


def print_gap_report(rates_df, group_col, mapping=None):
    """
    Print approval rates and gap in a readable way.
    """
    if mapping:
        rates_df[group_col] = rates_df[group_col].map(mapping)

    print("\nApproval Rates by", group_col)
    print(rates_df)

    if len(rates_df) == 2:
        gap = abs(
            rates_df.loc[0, "approval_rate"] -
            rates_df.loc[1, "approval_rate"]
        )
        print(f"\nApproval Rate Gap: {gap:.3f}")


def self_employed_fairness_report(df, group_col, prediction_col):
    """
    Calculate and print approval rates and gap for self-employed groups.
    """
    rates = (
        df.groupby(group_col)[prediction_col]
          .mean()
          .reset_index(name="approval_rate")
    )

    gap = abs(
        rates.loc[rates[group_col] == 1, "approval_rate"].values[0]
        -
        rates.loc[rates[group_col] == 0, "approval_rate"].values[0]
    )

    print(f"\nApproval Rates by {group_col}")
    print(rates)
    print(f"{group_col} Approval Gap: {gap:.3f}")

    return rates, gap
